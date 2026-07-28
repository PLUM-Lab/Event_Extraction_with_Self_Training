import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np

from graph import Graph
from transformers import BertModel, RobertaModel
from global_feature import generate_global_feature_vector, generate_global_feature_maps
from util import normalize_score
from gnn import FinalGNN


def log_sum_exp(tensor, dim=0, keepdim: bool = False):
    """LogSumExp operation used by CRF."""
    m, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - m
    else:
        stable_vec = tensor - m.unsqueeze(dim)
    return m + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def sequence_mask(lens, max_len=None):
    """Generate a sequence mask tensor from sequence lengths, used by CRF."""
    batch_size = lens.size(0)
    if max_len is None:
        max_len = lens.max().item()
    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp
    return mask


def token_lens_to_offsets(token_lens):
    """Map token lengths to first word piece indices, used by the sentence
    encoder.
    :param token_lens (list): token lengths (word piece numbers)
    :return (list): first word piece indices (offsets)
    """
    max_token_num = max([len(x) for x in token_lens])
    offsets = []
    for seq_token_lens in token_lens:
        seq_offsets = [0]
        for l in seq_token_lens[:-1]:
            seq_offsets.append(seq_offsets[-1] + l)
        offsets.append(seq_offsets + [-1] * (max_token_num - len(seq_offsets)))
    return offsets


def token_lens_to_idxs(token_lens):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):
    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]
    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):
    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


def graphs_to_node_idxs(graphs):
    """
    :param graphs (list): A list of Graph objects.
    :return: entity/trigger index matrix, mask tensor, max number, and max length
    """
    entity_idxs, entity_masks = [], []
    trigger_idxs, trigger_masks = [], []
    max_entity_num = max(max(graph.entity_num for graph in graphs), 1)
    max_trigger_num = max(max(graph.trigger_num for graph in graphs), 1)
    max_entity_len = max(max([e[1] - e[0] for e in graph.entities] + [1])
                         for graph in graphs)
    max_trigger_len = max(max([t[1] - t[0] for t in graph.triggers] + [1])
                          for graph in graphs)
    for graph in graphs:
        seq_entity_idxs, seq_entity_masks = [], []
        seq_trigger_idxs, seq_trigger_masks = [], []
        for entity in graph.entities:
            entity_len = entity[1] - entity[0]
            seq_entity_idxs.extend([i for i in range(entity[0], entity[1])])
            seq_entity_idxs.extend([0] * (max_entity_len - entity_len))
            seq_entity_masks.extend([1.0 / entity_len] * entity_len)
            seq_entity_masks.extend([0.0] * (max_entity_len - entity_len))
        seq_entity_idxs.extend([0] * max_entity_len * (max_entity_num - graph.entity_num))
        seq_entity_masks.extend([0.0] * max_entity_len * (max_entity_num - graph.entity_num))
        entity_idxs.append(seq_entity_idxs)
        entity_masks.append(seq_entity_masks)

        for trigger in graph.triggers:
            trigger_len = trigger[1] - trigger[0]
            seq_trigger_idxs.extend([i for i in range(trigger[0], trigger[1])])
            seq_trigger_idxs.extend([0] * (max_trigger_len - trigger_len))
            seq_trigger_masks.extend([1.0 / trigger_len] * trigger_len)
            seq_trigger_masks.extend([0.0] * (max_trigger_len - trigger_len))
        seq_trigger_idxs.extend([0] * max_trigger_len * (max_trigger_num - graph.trigger_num))
        seq_trigger_masks.extend([0.0] * max_trigger_len * (max_trigger_num - graph.trigger_num))
        trigger_idxs.append(seq_trigger_idxs)
        trigger_masks.append(seq_trigger_masks)

    return (
        entity_idxs, entity_masks, max_entity_num, max_entity_len,
        trigger_idxs, trigger_masks, max_trigger_num, max_trigger_len,
    )


def graphs_to_label_idxs(graphs, max_entity_num=-1, max_trigger_num=-1,
                         relation_directional=False,
                         symmetric_relation_idxs=None):
    """Convert a list of graphs to label index and mask matrices
    :param graphs (list): A list of Graph objects.
    :param max_entity_num (int) Max entity number (default = -1).
    :param max_trigger_num (int) Max trigger number (default = -1).
    """
    if max_entity_num == -1:
        max_entity_num = max(max([g.entity_num for g in graphs]), 1)
    if max_trigger_num == -1:
        max_trigger_num = max(max([g.trigger_num for g in graphs]), 1)
    (
        batch_entity_idxs, batch_entity_mask,
        batch_trigger_idxs, batch_trigger_mask,
        batch_relation_idxs, batch_relation_mask,
        batch_role_idxs, batch_role_mask
    ) = [[] for _ in range(8)]
    for graph in graphs:
        (
            entity_idxs, entity_mask, trigger_idxs, trigger_mask,
            relation_idxs, relation_mask, role_idxs, role_mask,
        ) = graph.to_label_idxs(max_entity_num, max_trigger_num,
                                relation_directional=relation_directional,
                                symmetric_relation_idxs=symmetric_relation_idxs)
        batch_entity_idxs.append(entity_idxs)
        batch_entity_mask.append(entity_mask)
        batch_trigger_idxs.append(trigger_idxs)
        batch_trigger_mask.append(trigger_mask)
        batch_relation_idxs.append(relation_idxs)
        batch_relation_mask.append(relation_mask)
        batch_role_idxs.append(role_idxs)
        batch_role_mask.append(role_mask)
    return (
        batch_entity_idxs, batch_entity_mask,
        batch_trigger_idxs, batch_trigger_mask,
        batch_relation_idxs, batch_relation_mask,
        batch_role_idxs, batch_role_mask
    )


def generate_pairwise_idxs(num1, num2):
    """Generate all pairwise combinations among entity mentions (relation) or
    event triggers and entity mentions (argument role).
    For example, if there are 2 triggers and 3 mentions in a sentence, num1 = 2,
    and num2 = 3. We generate the following vector:
    idxs = [0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4]
    Suppose `trigger_reprs` and `entity_reprs` are trigger/entity representation
    tensors. We concatenate them using:
    te_reprs = torch.cat([entity_reprs, entity_reprs], dim=1)
    After that we select vectors from `te_reprs` using (incomplete code) to obtain
    pairwise combinations of all trigger and entity vectors.
    te_reprs = torch.gather(te_reprs, 1, idxs)
    te_reprs = te_reprs.view(batch_size, -1, 2 * bert_dim)
    :param num1: trigger number (argument role) or entity number (relation)
    :param num2: entity number (relation)
    :return (list): a list of indices
    """
    idxs = []
    for i in range(num1):
        for j in range(num2):
            idxs.append(i)
            idxs.append(j + num1)
    return idxs


def tag_paths_to_spans(paths, token_nums, vocab):
    """Convert predicted tag paths to a list of spans (entity mentions or event
    triggers).
    :param paths: predicted tag paths.
    :return (list): a list (batch) of lists (sequence) of spans.
    """
    batch_mentions = []
    itos = {i: s for s, i in vocab.items()}
    for i, path in enumerate(paths):
        mentions = []
        cur_mention = None
        path = path.tolist()[:token_nums[i].item()]
        for j, tag in enumerate(path):
            tag = itos[tag]
            if tag == 'O':
                prefix = tag = 'O'
            else:
                prefix, tag = tag.split('-', 1)

            if prefix == 'B':
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = [j, j + 1, tag]
            elif prefix == 'I':
                if cur_mention is None:
                    # treat it as B-*
                    cur_mention = [j, j + 1, tag]
                elif cur_mention[-1] == tag:
                    cur_mention[1] = j + 1
                else:
                    # treat it as B-*
                    mentions.append(cur_mention)
                    cur_mention = [j, j + 1, tag]
            else:
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = None
        if cur_mention:
            mentions.append(cur_mention)
        batch_mentions.append(mentions)
    return batch_mentions


def generate_span_reprs(span_list, bert_input):
    # bert_input: (max_seq_len, bert_dim)
    # span_list: [[0,1], [2,4], ...]
    bert_dim = bert_input.shape[1]
    idxs, masks = [], []
    max_span_len = max([span[1]-span[0] for span in span_list])
    for span in span_list:
        span_len = span[1] - span[0]
        idxs.extend([i for i in range(span[0], span[1])])
        idxs.extend([0] * (max_span_len - span_len))
        masks.extend([1.0 / span_len] * span_len)
        masks.extend([0.0] * (max_span_len - span_len))
    tensor_idxs = bert_input.new_tensor(idxs, dtype=torch.long)
    tensor_masks = bert_input.new_tensor(masks)
    
    length = len(masks)

    selected_vecs = bert_input[tensor_idxs]
    expanded_masks = tensor_masks.unsqueeze(-1).expand(length, bert_dim)
    vecs = (selected_vecs * expanded_masks).reshape(len(span_list), max_span_len, bert_dim)
    out_vecs = torch.sum(vecs, 1)

    return out_vecs


class Linears(nn.Module):
    """Multiple linear layers with Dropout."""
    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs


class GraphScoreModel(nn.Module):
    def __init__(self,
                 config,
                 vocabs):
        super().__init__()

        self.config = config
        self.if_local = config.if_local
        self.device = config.gpu_device

        # vocabulary
        self.vocabs = vocabs
        self.entity_label_stoi = vocabs['entity_label']
        self.trigger_label_stoi = vocabs['trigger_label']
        self.mention_type_stoi = vocabs['mention_type']
        self.entity_type_stoi = vocabs['entity_type']
        self.event_type_stoi = vocabs['event_type']
        self.relation_type_stoi = vocabs['relation_type']
        self.role_type_stoi = vocabs['role_type']
        self.amr_edge_type_stoi = vocabs['amr_edge_type']
        self.entity_label_itos = {i:s for s, i in self.entity_label_stoi.items()}
        self.trigger_label_itos = {i:s for s, i in self.trigger_label_stoi.items()}
        self.entity_type_itos = {i: s for s, i in self.entity_type_stoi.items()}
        self.event_type_itos = {i: s for s, i in self.event_type_stoi.items()}
        self.relation_type_itos = {i: s for s, i in self.relation_type_stoi.items()}
        self.role_type_itos = {i: s for s, i in self.role_type_stoi.items()}
        self.entity_label_num = len(self.entity_label_stoi)
        self.trigger_label_num = len(self.trigger_label_stoi)
        self.mention_type_num = len(self.mention_type_stoi)
        self.entity_type_num = len(self.entity_type_stoi)
        self.event_type_num = len(self.event_type_stoi)
        self.relation_type_num = len(self.relation_type_stoi)
        self.role_type_num = len(self.role_type_stoi)
        self.amr_edge_type_num = len(self.amr_edge_type_stoi)
        # self.valid_event_entity_role = load_valid_pattern(config.valid_pattern_path, vocabs)
        self.use_graph_encoder = config.use_graph_encoder

        # bert
        bert_config = config.bert_config
        bert_config.output_hidden_states = True
        self.bert_dim = bert_config.hidden_size
        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.bert_dim *= 2
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy
        
        # local classifiers
        self.use_trigger_type = config.use_trigger_type
        self.binary_dim = self.bert_dim * 2
        linear_bias = config.linear_bias
        linear_dropout = config.linear_dropout
        role_hidden_num = config.role_hidden_num
        role_input_dim = self.binary_dim + (self.event_type_num if self.use_trigger_type else 0)
        self.classify_layer = Linears([role_input_dim, role_hidden_num,
                                      self.role_type_num],
                                     dropout_prob=linear_dropout,
                                     bias=linear_bias,
                                     activation=config.linear_activation)
        # graph encoder
        self.edge_type_num = self.amr_edge_type_num
        self.edge_type_dim = config.edge_type_dim
        gnn_layers = config.gnn_layers
        self.lamda = config.lamda
        self.graph_encoder = FinalGNN(self.bert_dim, self.edge_type_dim, self.edge_type_num, gnn_layers, self.lamda, config.gpu_device)
        
        self.use_bce_loss = config.use_bce_loss
        self.use_marginal_loss = config.use_marginal_loss
        self.loss_ratio = config.loss_ratio

        # loss
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.bce_loss = nn.BCELoss()


    def load_bert(self, name, cache_dir=None):
        """Load the pre-trained BERT model (used in training phrase)
        :param name (str): pre-trained BERT model name
        :param cache_dir (str): path to the BERT cache directory
        """
        print('Loading pre-trained BERT model {}'.format(name))
        # print(cache_dir)
        if name.startswith("roberta"):
            self.bert = RobertaModel.from_pretrained(name, output_hidden_states=True)
        if name.startswith("bert"):
            self.bert = BertModel.from_pretrained(name,
                                                  cache_dir=cache_dir,
                                                  output_hidden_states=True)

    def encode(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        bert_outputs = all_bert_outputs[0]
        # print('\n')
        # print("bert_init_dim", bert_outputs.shape)

        if self.use_extra_bert:
            extra_bert_outputs = all_bert_outputs[2][self.extra_bert]
            bert_outputs = torch.cat([bert_outputs, extra_bert_outputs], dim=2)

        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets)
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            masks = bert_outputs.new(masks).unsqueeze(-1)
            bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
            bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
            bert_outputs = bert_outputs.sum(2)
        else:
            raise ValueError('Unknown multi-piece token handling strategy: {}'
                             .format(self.multi_piece))
        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs

    def scores(self, bert_outputs, batch, graphs, amrs, aligns, exists):
        batch_size, _, bert_dim = bert_outputs.size()
        (
            entity_idxs, entity_masks, entity_num, entity_len,
            trigger_idxs, trigger_masks, trigger_num, trigger_len,
        ) = graphs_to_node_idxs(graphs)

        entity_idxs = bert_outputs.new_tensor(entity_idxs, dtype=torch.long)
        trigger_idxs = bert_outputs.new_tensor(trigger_idxs, dtype=torch.long)
        entity_masks = bert_outputs.new_tensor(entity_masks)
        trigger_masks = bert_outputs.new_tensor(trigger_masks)

        # entity type scores
        entity_idxs = entity_idxs.unsqueeze(-1).expand(-1, -1, bert_dim)
        entity_masks = entity_masks.unsqueeze(-1).expand(-1, -1, bert_dim)
        entity_words = torch.gather(bert_outputs, 1, entity_idxs)
        entity_words = entity_words * entity_masks
        entity_words = entity_words.view(batch_size, entity_num, entity_len, bert_dim)
        entity_reprs = entity_words.sum(2) # B. max_ent_num, H
        
        # trigger type scores
        trigger_idxs = trigger_idxs.unsqueeze(-1).expand(-1, -1, bert_dim)
        trigger_masks = trigger_masks.unsqueeze(-1).expand(-1, -1, bert_dim)
        trigger_words = torch.gather(bert_outputs, 1, trigger_idxs)
        trigger_words = trigger_words * trigger_masks
        trigger_words = trigger_words.view(batch_size, trigger_num, trigger_len, bert_dim)
        trigger_reprs = trigger_words.sum(2) # B. max_tri_num, H

        # must deal with them individually within a batch
        if self.use_graph_encoder:
            new_trigger_reprs = bert_outputs.new_zeros(trigger_reprs.shape)
            new_entity_reprs = bert_outputs.new_zeros(entity_reprs.shape)

            batch_entity_types = []
            batch_trigger_types = []
            for i in range(batch_size):
                graph_i = graphs[i]
                bert_i = bert_outputs[i]
                amr_i = amrs[i].clone()
                align_i = aligns[i]
                exist_i = exists[i]

                if amr_i.num_nodes() == 0:
                    trig_list = graph_i.triggers
                    trigger_types = []
                    for j, event in enumerate(trig_list):
                        trigger_types.append(event[2])
                    trigger_types += [0] * (trigger_num - len(trig_list))
                    batch_trigger_types.extend(trigger_types)
                    new_trigger_reprs[i, :, :] = trigger_reprs[i,:,:]
                    new_entity_reprs[i, :, :] = entity_reprs[i,:,:]
                    continue
                # entity and trigger representations    
                ent_repr_i = entity_reprs[i]
                trig_repr_i = trigger_reprs[i]

                span_list = amr_i.ndata["token_span"].tolist()

                amr_span_reprs = generate_span_reprs(span_list, bert_i)
                new_amr_reprs = amr_span_reprs.new_zeros(amr_span_reprs.shape)
                ent_list = graph_i.entities
                trig_list = graph_i.triggers

                ent_to_node_idx = []
                trig_to_node_idx = []

                if_amr_nodes_selected = [0 for _ in range(len(span_list))]
                ent_lines = []
                trig_lines = []
                
                # update span_list and span_reprs for gnn encoding
                trigger_types = []
                for j, event in enumerate(trig_list):
                    trigger_types.append(event[2])
                    head_event_idx = event[1] - 1
                    align_node_idx = align_i[head_event_idx]
                    if exist_i[head_event_idx] == 1 and if_amr_nodes_selected[align_node_idx] == 0:
                        if_amr_nodes_selected[align_node_idx] = 1
                        new_amr_reprs[align_node_idx] = trig_repr_i[j]
                        trig_lines.append(align_node_idx)
                    else:
                        amr_i.add_nodes(1)
                        curr_node_num = amr_i.num_nodes()
                        new_amr_reprs = torch.cat((new_amr_reprs, trig_repr_i[j:j+1]), 0)
                        trig_lines.append(curr_node_num - 1)
                        if_amr_nodes_selected.append(1)
                        amr_i.add_edge(align_node_idx, curr_node_num - 1)
                        amr_i.edata['type'][-1][0] = self.amr_edge_type_stoi['other']
                trigger_types += [0] * (trigger_num - len(trig_list))
                batch_trigger_types.extend(trigger_types)
                entity_types = []
                for j, entity in enumerate(ent_list):
                    entity_types.append(entity[2])
                    head_entity_idx = entity[1] - 1
                    align_node_idx = align_i[head_entity_idx]
                    if exist_i[head_entity_idx] == 1 and if_amr_nodes_selected[align_node_idx] == 0:
                        if_amr_nodes_selected[align_node_idx] = 1
                        new_amr_reprs[align_node_idx] = ent_repr_i[j]
                        ent_lines.append(align_node_idx)
                    else:
                        amr_i.add_nodes(1)
                        curr_node_num = amr_i.num_nodes()
                        new_amr_reprs = torch.cat((new_amr_reprs, ent_repr_i[j:j+1]), 0)
                        ent_lines.append(curr_node_num - 1)
                        if_amr_nodes_selected.append(1)
                        amr_i.add_edge(align_node_idx, curr_node_num - 1)
                        amr_i.edata['type'][-1][0] = self.amr_edge_type_stoi['other']
                        # the last edge type is new relation
                entity_types += [0] * (entity_num - len(ent_list))
                batch_entity_types.extend(entity_types)
                # fill in the blanks of other NODES
                for j in range(len(if_amr_nodes_selected)):
                    if if_amr_nodes_selected[j] == 0:
                        new_amr_reprs[j] = amr_span_reprs[j]
                assert (new_amr_reprs.shape[0] == amr_i.num_nodes())
                # now we have graph_i and new_amr_reprs
                # check_amr_t = new_amr_reprs[bert_outputs.new_tensor(trig_lines, dtype=torch.long)]
                # check_amr_e = new_amr_reprs[bert_outputs.new_tensor(ent_lines, dtype=torch.long)]
                output_embs = self.graph_encoder(amr_i, new_amr_reprs)

                tensor_trig_lines = bert_outputs.new_tensor(trig_lines, dtype=torch.long)
                tensor_ent_lines = bert_outputs.new_tensor(ent_lines, dtype=torch.long)

                new_trig_embs = output_embs[tensor_trig_lines]
                new_ent_embs = output_embs[tensor_ent_lines]

                new_trigger_reprs[i, 0:len(trig_lines), :] = new_trig_embs
                new_entity_reprs[i, 0:len(ent_lines), :] = new_ent_embs
        else:
            new_trigger_reprs = trigger_reprs
            new_entity_reprs = entity_reprs
        
        # relation type score
        # ee_idxs = generate_pairwise_idxs(entity_num, entity_num)
        # ee_idxs = entity_idxs.new(ee_idxs)
        # ee_idxs = ee_idxs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, bert_dim)
        # ee_reprs = torch.cat([new_entity_reprs, new_entity_reprs], dim=1)
        # ee_reprs = torch.gather(ee_reprs, 1, ee_idxs)
        # ee_reprs = ee_reprs.view(batch_size, -1, 2 * bert_dim)
        # role type score
        te_idxs = generate_pairwise_idxs(trigger_num, entity_num)
        te_idxs = entity_idxs.new(te_idxs)
        te_idxs = te_idxs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, bert_dim)
        te_reprs = torch.cat([new_trigger_reprs, new_entity_reprs], dim=1)
        te_reprs = torch.gather(te_reprs, 1, te_idxs)
        te_reprs = te_reprs.view(batch_size, -1, 2 * bert_dim)

        # if self.use_entity_type:
        #     batch_entity_types = bert_outputs.new_tensor(batch_entity_types, dtype=torch.long)
        #     batch_entity_types = batch_entity_types.view(batch_size, -1)
        #     batch_entity_types_onehot = nn.functional.one_hot(batch_entity_types, self.entity_type_num)
        #     batch_entity_types_onehot = batch_entity_types_onehot.repeat(1, trigger_num, 1)
        #     te_reprs = torch.cat([te_reprs, batch_entity_types_onehot], dim=2)
        if self.use_trigger_type:
            batch_trigger_types = bert_outputs.new_tensor(batch_trigger_types, dtype=torch.long)
            batch_trigger_types = batch_trigger_types.view(batch_size, -1)
            batch_trigger_types_onehot = nn.functional.one_hot(batch_trigger_types, self.event_type_num)
            batch_trigger_types_onehot = batch_trigger_types_onehot.unsqueeze(-2).repeat(1, 1, entity_num, 1)
            batch_trigger_types_onehot = batch_trigger_types_onehot.view(batch_size, -1, self.event_type_num)
            te_reprs = torch.cat([batch_trigger_types_onehot,te_reprs], dim=2)
        batch_logits = self.classify_layer(te_reprs) # batch,  max_trigger* max_entity, num_roles
        logit_list = []
        tag_list = []
        for i in range(batch_size):
            graph_i = graphs[i]
            role_dict = { (role[0],role[1]):role[2] for role in graph_i.roles}
            for tri_idx in range(graph_i.trigger_num):
                for ent_idx in range(graph_i.entity_num):
                    logit_list.append(batch_logits[i,tri_idx*entity_num+ent_idx,:])
                    if not batch.role_type_idxs is None:
                        tag_list.append(batch.role_type_idxs[i*trigger_num*entity_num+tri_idx*entity_num+ent_idx])
                    else:
                        role_type_idx = role_dict.get((tri_idx,ent_idx),self.role_type_stoi['O'])
                        tag_list.append(role_type_idx)
        if len(logit_list) == 0:
            return {'probs':None,'entity_num':entity_num,'trigger_num':trigger_num}
        logits = torch.stack(logit_list,dim=0) # B, num_role
        targets = bert_outputs.new_tensor(tag_list, dtype=torch.long)
        probs = self.sigmoid(logits)
        targets_one_hot = F.one_hot(targets,num_classes=self.role_type_num).to(torch.float)
        output = {'logits':logits,'targets':targets,'targets_one_hot':targets_one_hot,'probs':probs,'human_readable':None,'entity_num':entity_num,'trigger_num':trigger_num}
        return output
        
    def loss_function(self, output):
        total_loss = 0.0
        probs = output['probs']
        targets = output['targets']
        targets_one_hot = output['targets_one_hot']
        if self.use_bce_loss:
            # print('probs',probs.shape)
            # print('targets_one_hot',targets_one_hot.shape)
            loss = self.bce_loss(probs,targets_one_hot)
            total_loss = total_loss + loss
            # print('bce loss: ',loss.item())
        if self.use_marginal_loss:
            target_probs = torch.gather(probs,-1,targets.unsqueeze(-1)) # B
            max_probs, max_args = torch.max(probs,-1) # B
            masks = targets.new_tensor(torch.ones_like(targets),dtype=torch.float) # B
            masks[targets==max_args] = 0.0
            # print('target: ',targets.detach().cpu().tolist())
            # print('max_args: ',max_args.detach().cpu().tolist())
            # print('masks:,', masks.detach().cpu().tolist())
            loss = torch.max(1.0+max_probs-target_probs, targets.new_tensor([0.0],dtype=torch.float)) * masks
            loss = torch.mean(loss) * self.loss_ratio 
            # print('hinge loss: ',loss.item())
            total_loss = total_loss + loss
        return total_loss

    def forward(self, batch, epoch):
        # encoding
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        
        batch_size, _, _ = bert_outputs.size()
        output = self.scores(bert_outputs, batch, batch.graphs, batch.amr, batch.align, batch.exist)
        loss = self.loss_function(output)
        output['loss'] = loss
        return output

    def predict(self, batch):
        self.eval()
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()

        max_entity_num = max(max(graph.entity_num for graph in batch.graphs), 1)
        max_trigger_num = max(max(graph.trigger_num for graph in batch.graphs), 1)
        node_graphs = [Graph(graph.entities, graph.triggers, [], [], self.vocabs)
                       for graph in batch.graphs]
        output = self.scores(bert_outputs, batch, node_graphs, batch.amr, batch.align, batch.exist)
        if not output['probs'] is None:
            probs = output['probs'].detach().cpu().tolist()
        else:
            probs = output['probs']

        O_idx = self.role_type_stoi['O']
        role_idx = 0
        copy_graphs = []
        for graph in batch.graphs:
            copy_graph = graph.copy()
            # print('graph before: ',copy_graph.roles)
            copy_graph.roles = []
            copy_graph.role_scores = []
            for tri_idx in range(graph.trigger_num):
                for ent_idx in range(graph.entity_num):
                    prob = probs[role_idx]
                    for type_idx, p in enumerate(prob):
                        if not type_idx == O_idx and p>0.5:
                            copy_graph.add_role(tri_idx,ent_idx,type_idx,p)
                    role_idx+=1
            # print('graph after: ',copy_graph.roles)
            copy_graphs.append(copy_graph)
        return copy_graphs
        

    def feedback(self, batch):
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()
        output = self.scores(bert_outputs, batch.graphs, batch.amr, batch.align, batch.exist, train=False) # batch, max_entity * max_trigger, num_roles
        if not output['probs'] is None:
            probs = output['probs'].detach().cpu().tolist()
        else:
            probs = output['probs']
        # feedbacks = []
        # count_role = 0
        # # human_readable = {}
        # for b in range(batch_size):
        #     graph = batch.graphs[b]
        #     role_dict = { (role[0],role[1]):(role[2],role_score) for role, role_score in zip(graph.roles, graph.role_scores)}

        #     # print('# -------------------- new inst --------------------- #')
        #     # human_readable['tokens']=batch.tokens[b]
        #     # human_readable['roles']=[]
        #     # print(human_readable['tokens'])

        #     for t in range(graph.trigger_num):
        #         for e in range(graph.entity_num):
        #             count_role+=1
        #             feedback = heuristic(b,t,e,role_dict,role_type_scores)
        #             feedbacks.append(feedback)

        #             # role_type_idx, pred_role_score = role_dict.get((t,e),(self.role_type_stoi['O'],0.0))
        #             # _, score_role_type_idx = torch.max(role_type_scores[b,t*entity_num+e],dim=-1)
        #             # trigger_tokens = human_readable['tokens'][graph.triggers[t][0]:graph.triggers[t][1]]
        #             # trigger_type = self.event_type_itos[graph.triggers[t][2]]
        #             # entity_tokens = human_readable['tokens'][graph.entities[e][0]:graph.entities[e][1]]
        #             # entity_type = self.entity_type_itos[graph.entities[e][2]]
        #             # role_type = self.role_type_itos[role_type_idx]
        #             # human_readable['roles'].append((trigger_type,trigger_tokens,entity_tokens,entity_type,role_type,feedback))
        #             # print(trigger_type,trigger_tokens,entity_tokens,entity_type,'pred: ',role_type,'feedback: ',feedback,'score pred: ',self.role_type_itos[score_role_type_idx.item()],'ie score: ',pred_role_score)

        #         feedbacks.extend([0]*(entity_num-graph.entity_num))
        #     feedbacks.extend([0]*entity_num*(trigger_num-graph.trigger_num))
        role_feedbacks = []
        trigger_feedbacks = []
        count_role = 0
        count_trigger = 0
        entity_num = max(output['entity_num'],1)
        trigger_num = max(output['trigger_num'],1)
        for b in range(batch_size):
            graph = batch.graphs[b]
            role_dict = { (role[0],role[1]):(role[2],role_score) for role, role_score in zip(graph.roles, graph.role_scores)}
            for t in range(graph.trigger_num):
                trigger_feedback = 0.0
                count_trigger += 1
                for e in range(graph.entity_num):
                    ie_pred = role_dict.get((t,e),(self.role_type_stoi['O'],0.75))[0]
                    role_feedback = probs[count_role][ie_pred] * 2.0 -1.0
                    role_feedbacks.append(role_feedback)
                    count_role += 1
                    trigger_feedback+=role_feedback
                trigger_feedbacks.append(trigger_feedback/max(graph.entity_num,1))
                role_feedbacks.extend([0]*(entity_num-graph.entity_num))
            trigger_feedbacks.extend([0]*(trigger_num-graph.trigger_num))
            role_feedbacks.extend([0]*entity_num*(trigger_num-graph.trigger_num))
        return feedbacks, count_role