import copy
import random

import torch
import torch.nn as nn
import dgl
import numpy as np

from graph import Graph
from transformers import BertModel, RobertaModel
from global_feature import generate_global_feature_vector, generate_global_feature_maps
from util import normalize_score
from self_attention import SimpleSelfAttention, BilinearMatrixAttention


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
            seq_entity_idxs.extend([0] * (max_entity_len - entity_len)) # [0,1,2,0,0, 7,8,0,0,0]
            seq_entity_masks.extend([1.0 / entity_len] * entity_len) 
            seq_entity_masks.extend([0.0] * (max_entity_len - entity_len)) # [1/3,1/3,1/3,0,0, 1/2,1/2,0,0,0]
        seq_entity_idxs.extend([0] * max_entity_len * (max_entity_num - graph.entity_num)) # [0,1,2,0,0, 7,8,0,0,0, 0,0,0,0,0, 0,0,0,0,0]
        seq_entity_masks.extend([0.0] * max_entity_len * (max_entity_num - graph.entity_num)) # [1/3,1/3,1/3,0,0, 1/2,1/2,0,0,0， 0,0,0,0,0, 0,0,0,0,0]
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

def generate_amr_reprs(span_list, bert_input):
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
    out_vecs = torch.sum(vecs, 1) # num_amr_spans, bert_dim

    return out_vecs


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

# this function collect all nodes on the path between 2 nodes
# def get_nodes_on_path():

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




class ScoreGraph(nn.Module):
    def __init__(self,
                 config,
                 vocabs,
                 valid_patterns=None):
        super().__init__()

        self.config = config
        self.if_local = config.if_local
        # vocabularies
        self.vocabs = vocabs
        self.entity_label_stoi = vocabs['entity_label']
        self.trigger_label_stoi = vocabs['trigger_label']
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
        self.entity_type_num = len(self.entity_type_stoi)
        self.event_type_num = len(self.event_type_stoi)
        self.relation_type_num = len(self.relation_type_stoi)
        self.role_type_num = len(self.role_type_stoi)
        self.amr_edge_type_num = len(self.amr_edge_type_stoi)

        role_type_idx = []
        for _type, _idx in self.role_type_stoi.items():
            role_type_idx.append(_idx)
        relation_type_idx = []
        for _type, _idx in self.relation_type_stoi.items():
            relation_type_idx.append(_idx)
        self.role_type_idx = role_type_idx
        self.relation_type_idx = relation_type_idx


        # BERT encoder
        bert_config = config.bert_config
        # print(bert_config)
        bert_config.output_hidden_states = True
        self.bert_dim = bert_config.hidden_size
        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.bert_dim *= 2
        self.bert = BertModel(bert_config)
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy
        self.device = config.gpu_device

        # self.edge_type_dim = config.edge_type_dim

        self.ie_node_type_emb = nn.Embedding(self.event_type_num+self.entity_type_num, self.bert_dim)
        self.arg_role_emb = nn.Embedding(self.role_type_num, self.bert_dim)
        self.amr_edge_type_emb = nn.Embedding(self.amr_edge_type_num, self.bert_dim)
        self.pos_emb = nn.Embedding(128,self.bert_dim)
        self.amr_attention = nn.ModuleList([SimpleSelfAttention(self.bert_dim, config.intermediate_dim, config.num_attention_heads) for i in range(config.attention_layer_num)])
        # compute the similarity scores
        self.dropout_emb = nn.Dropout(p=config.emb_dropout)
        self.dropout_repr = nn.Dropout(p=config.repr_dropout)
        
        self.similarity_fc_type = config.similarity_fc
        self.use_node_type = config.use_node_type
        self.use_pos_emb = config.use_pos_emb
        if config.similarity_fc == 'linear':
            self.similarity_fc = nn.Linear(2*self.bert_dim, 1)
        elif config.similarity_fc == 'bilinear':
            self.similarity_fc = BilinearMatrixAttention(self.bert_dim, self.bert_dim, use_input_biases=True)



        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

        # loss
        self.bce_loss = nn.BCELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()


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

    def bert_encoder(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        :param amr_graphs (list): list of dgl amr graphs

        return word level representation
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

    def compute_scores(self, bert_outputs, batch, epoch):
        
        batch_loss = []
        batch_size, _, _ = bert_outputs.size()
        logits = []
        targets = []
        batch_info = []

        for b in range(batch_size):
            bert_i = bert_outputs[b]

            ie_i = batch.ie_graph[b] # get ie graph 
            amr_i = batch.amr[b] # get amr graph

            if amr_i.num_nodes() == 0:
                continue
            
            # get ie node representation
            ie_span_list = ie_i.ndata["node_span"].tolist()
            ie_repr_i = generate_amr_reprs(ie_span_list, bert_i)
            amr_span_list = amr_i.ndata["token_span"].tolist()
            amr_repr_i = generate_amr_reprs(amr_span_list, bert_i) # (num_span, bert_dim)

            ie2amr_edge_map = batch.ie_edge_dict[b]

            # print(ie2amr_edge_map)

            
            # print('num edge: ', len(ie_edge_dict))
            for ie_edge in  ie2amr_edge_map:
                # only event 
                # ie2amr_edge_map[tuple(_edge)] = {'amr_node':amr_node_path,'amr_rel':amr_rel_path,'is_event':is_event,'ie_rel':ie_egde_type}
                if not ie2amr_edge_map[ie_edge]['is_event']:
                    continue
                # labels[_idx][_idx] = 1.0
                batch_info.append(ie2amr_edge_map[ie_edge]['human_readable'])
                ie_startnode = ie_edge[0]
                ie_endnode =  ie_edge[1]
                # print('ie node has edge: ', ie_i.has_edges_between(begin_ie_node, end_ie_node))
                # begin_amr_node = ie2amr_map[begin_ie_node]
                # end_amr_node = ie2amr_map[end_ie_node]
                sub_amr_nodes = ie2amr_edge_map[ie_edge]['amr_node']
                sub_amr_rels = ie2amr_edge_map[ie_edge]['amr_rel']
                
                # get representations
                sub_amr_node_repr = torch.index_select(amr_repr_i,0,sub_amr_nodes) # num of amr nodes on path
                sub_amr_rel_repr = self.dropout_emb(self.amr_edge_type_emb(sub_amr_rels)) # num of amr rels
                # sub_ie_role_repr = self.arg_role_emb[ie2amr_edge_map[ie_edge]['ie_edge_type']]
                ie_startnode_repr = ie_repr_i[ie_startnode]
                ie_endnode_repr = ie_repr_i[ie_endnode]
                
                # may used latter
                ie_startnode_type_repr = self.dropout_emb(self.ie_node_type_emb(ie2amr_edge_map[ie_edge]['ie_node_type'][0]))
                ie_endnode_type_repr = self.dropout_emb(self.ie_node_type_emb(ie2amr_edge_map[ie_edge]['ie_node_type'][1]))

                # [start token] [rel] .... [rel] [end token]
                sub_amr_repr = []
                for _idx in range(sub_amr_rel_repr.size()[0]):
                    if _idx == 0:
                        sub_amr_repr.append(ie_startnode_repr)
                    else:
                        sub_amr_repr.append(sub_amr_node_repr[_idx])
                    sub_amr_repr.append(sub_amr_rel_repr[_idx])
                sub_amr_repr.append(ie_endnode_repr)

                if self.use_node_type:
                    sub_amr_repr = [ie_startnode_type_repr] + sub_amr_repr + [ie_endnode_type_repr]
                # sub_amr_repr = [sub_ie_startnode_type_repr] + sub_amr_repr + [sub_ie_endnode_type_repr]
                sub_amr_repr = torch.stack(sub_amr_repr,dim=0)
                if self.use_pos_emb:
                    pos_emb = self.pos_emb(bert_i.new_tensor(torch.arange(sub_amr_repr.shape[0]),dtype=torch.long))
                    sub_amr_repr = sub_amr_repr + pos_emb
                sub_amr_repr = sub_amr_repr.unsqueeze(0)
                # contextualized layer
                for _att_layer in self.amr_attention:
                    sub_amr_repr = _att_layer(sub_amr_repr) # 1, amr_path_length, H
                
                pooled_amr_repr = self.gelu(torch.mean(sub_amr_repr, 1)) # 1, H
                # print(ie_role_repr.shape)
                # print(ie_startnode_type_repr.shape)
                # print(pooled_amr_repr.shape)
                # print(ie_endnode_type_repr.shape)
                ie_startnode_type_repr = ie_startnode_type_repr.expand(self.role_type_num, -1)
                ie_endnode_type_repr = ie_endnode_type_repr.expand(self.role_type_num, -1)
                ie_role_repr = self.dropout_emb(self.arg_role_emb(bert_i.new_tensor(torch.arange(self.role_type_num),dtype=torch.long))) # arg role types, H
                # ff similarity function
                if self.similarity_fc_type == 'linear':
                    pooled_amr_repr = pooled_amr_repr.expand(self.role_type_num, -1)
                    ie_amr_repr = torch.cat([ie_role_repr,pooled_amr_repr],-1) # arg role, 2H
                    # ie_amr_repr = self.dropout_repr(ie_amr_repr)
                    logit = self.similarity_fc(ie_amr_repr) # arg role, 1
                elif self.similarity_fc_type == 'bilinear':
                    # print(ie_role_repr.shape, pooled_amr_repr.shape)
                    logit = self.similarity_fc(ie_role_repr, pooled_amr_repr.unsqueeze(1))
                    logit = logit.squeeze(0)

                
                logit = logit.squeeze(-1)
                logits.append(logit)
                assert ie2amr_edge_map[ie_edge]['ie_edge_type']<self.role_type_num
                targets.append(ie2amr_edge_map[ie_edge]['ie_edge_type'])
        
        logits = torch.stack(logits, dim=0)
        targets = bert_i.new_tensor(targets,dtype=torch.long)
        loss = self.loss_function(logits,targets)
        # print('batch loss: ',loss)

        return {'logits':logits, 'targets':targets, 'loss':loss, 'human_readable': batch_info}

    def loss_function(self, logits, targets):

        # instance_bce
        # probs = self.sigmoid(logits) # get probability
        # num_edge, _ = probs.size()
        #get labels
        # print('probs: ',probs)
        # print('labels: ',labels)
        # labels = torch.diag(torch.ones(num_edge), diagonal=0)
        # labels = logits.new_tensor(labels)
        # probs = probs.view(-1)
        # # print('probs: ',probs)
        # labels = labels.view(-1)
        # # print('labels: ',labels)
        # loss = self.bce_loss(probs, labels)

        # print('loss: ',loss.item())

        # max_margin
        # print(logits.shape)
        # print(targets.shape)
        
        # cross entropy loss
        # loss = self.cross_entropy_loss(logits, targets)
        # max margin loss

        # logits = self.sigmoid(logits) # bound between 0-1
        # scores, preds = torch.max(logits,dim=1)

        # # print(scores.shape)
        # # print(preds.shape)
        # # print(targets.shape)
        # # print(logits.shape)
        # # print(torch.gather(logits, 1, targets.unsqueeze(1)).shape)
        # _zeros = logits.new_tensor(torch.zeros_like(scores))

        # margins = 0.5+scores-torch.gather(logits, 1, targets.unsqueeze(1)).squeeze(-1)

        # # print(margins.shape)
        # # print(scores.shape)

        # margins = torch.stack([margins, _zeros],dim=1)

        # loss = torch.max(margins,1)[0]

        # loss = loss * (1.0 - preds==targets)

        # loss = loss.mean()

        # bce loss

        bin_targets = targets.new_tensor(torch.zeros_like(logits),dtype=torch.float)
        src_tensor = targets.new_tensor(torch.ones_like(logits),dtype=torch.float)
        targets = targets.unsqueeze(1)
        bin_targets = bin_targets.scatter_(1,targets,src_tensor)
        # print(bin_targets)
        probs = self.sigmoid(logits)
        loss = self.bce_loss(probs, bin_targets)
        return loss

    def forward(self, batch, epoch):
        # encoding
        bert_outputs = self.bert_encoder(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        
        # batch_size, _, _ = bert_outputs.size() # batch, seq_length, bert_dim
        output = self.compute_scores(bert_outputs, batch, epoch)
        return output['loss']

    def cal_f1(self, logit, target):
        tp = 0.0
        fp = 0.0
        fn = 0.0
        for i, v in enumerate(logit):
            if i == target:
                if v >= 0.5:
                    tp+=1
                else:
                    fn+=1
            else:
                if v >=0.5:
                    fp+=1
        r = tp/(tp+fp)
        p = tp/(tp+fn)
        f1 = 2*r*p/(r+p)
        return f1

    def make_human_readable(self, logit, target, topk, _readable):
        _readable['target'] = (self.role_type_itos[target], logit[target])
        _readable['fp'] = []
        _readable['topk'] = []
        for i , v in enumerate(logit):
            if not i == target:
                if v >=0.5:
                    _readable['fp'].append((self.role_type_itos[i], v))
            if i in topk:
                _readable['topk'].append((self.role_type_itos[i], v))
        _readable['topk'].sort(key=lambda x: x[1], reverse=True)
        return _readable

    def predict(self, batch, epoch):
        self.eval()

        # identification
        # tp = 0.0
        # fp = 0.0
        # fb = 
        # total = 0.0
        # topk_count = 0
        f1 = []
        readable_list = []
        metrics = {}
        with torch.no_grad():
            # for i in range(batch_size):
            #     bert_outputs = self.bert_encoder(batch.piece_idxs,
            #                        batch.attention_masks,
            #                        batch.token_lens)
        
                # batch_size, _, _ = bert_outputs.size() # batch, seq_length, bert_dim
            bert_outputs = self.bert_encoder(batch.piece_idxs,
                                batch.attention_masks,
                                batch.token_lens)
            output = self.compute_scores(bert_outputs, batch, epoch)
            logits = output['logits']
            targets = output['targets'].unsqueeze(1)
            _, max_arg = torch.max(logits, 1)
            top1 = torch.sum(max_arg==targets.squeeze(1))
            human_readable = output['human_readable']
            # pred = torch.argmax(logits, dim=-1)
            topks = torch.topk(logits,3,dim=-1)[1]
            probs = self.sigmoid(logits)
            # print(probs)
            # print(targets)
            # tp = torch.sum(torch.gather(probs,1,targets)>=0.5)
            # fn = torch.sum(torch.gather(probs,1,targets)<0.5)
            # fp = torch.sum(probs>=0.5) - tp
            # total = targets.size()[0]
            tp = 0.0
            fn = 0.0
            fp = 0.0
            total = 0.0

            # for b in range(topk.shape[0]):
            #     for k in range(topk.shape[1]):
            #         topk_count += topk[b][k] == targets[b]
            # print()
            # print('tp: ',tp.item(),'fn: ',fn.item(),'fp: ',fp.item(),'top1: ',top1.item(),'total:', total)
            # print(logits)
            # print(targets)
            probs = probs.detach().cpu().tolist()
            targets = targets.squeeze(1).detach().cpu().tolist()
            topks = topks.detach().cpu().tolist()
            for prob, target, topk,  _readable in zip(probs, targets, topks, human_readable):
                if not target ==0:
                    total+=1
                f1.append(0.0)
                _readable = self.make_human_readable(prob, target, topk, _readable)
                # print(_readable)
                readable_list.append(_readable)
                for t, p in enumerate(prob):
                    if not _readable['ie_edge_type'] in metrics:
                        metrics[_readable['ie_edge_type']]={'tp':0.0, 'fn':0.0, 'fp':0.0}
                    if p >= 0.5 and not t == target:
                        metrics[_readable['ie_edge_type']]['fp']+=1
                    if t == target:
                        if p >= 0.5:
                            metrics[_readable['ie_edge_type']]['tp']+=1
                        else:
                            metrics[_readable['ie_edge_type']]['fn']+=1
                    
                    
                    if p >= 0.5 and not t == target:
                        if not t ==0:
                            fp+=1
                    if not target == 0:
                        if t == target:
                            if p >= 0.5:
                                tp+=1
                            else:
                                fn+=1

            # print('metrics: ', metrics)
            print()
            print('tp: ',tp,'fn: ',fn,'fp: ',fp,'total:', total)



            # print(f1)

        self.train()

        output = {'tp':tp,'fn':fn,'fp':fp,'top1':top1,'total':total,'metrics':metrics,'human_readable':readable_list}

        return output

    def measure_compat(self, batch):
        
        bert_outputs = self.bert_encoder(batch.piece_idxs,
                                batch.attention_masks,
                                batch.token_lens)

        batch_size, _, _ = bert_outputs.size()

        scores = []
        for b in range(batch_size):
            bert_i = bert_outputs[b]

            
            # get ie node representation
            ie_span_list = batch.graphs[b].triggers + batch.graphs[b].entities
            amr_node_span_dic = batch.amr_node_span_lists[b]
            amr_span_list = [amr_node_span_dic[i] for i in range(len(amr_node_span_dic))]
            ie2amr_edge_map = batch.ie2amr_edge_maps[b]
            if len(amr_span_list) <=0 or len(ie_span_list) <=0:
                scores.extend([0]*len(ie2amr_edge_map))
                continue

            ie_repr_i = generate_amr_reprs(ie_span_list, bert_i)
            amr_repr_i = generate_amr_reprs(amr_span_list, bert_i) # (num_span, bert_dim)


            # print('num edge: ', len(ie_edge_dict))
            for ie_edge in  ie2amr_edge_map:
                # only event 
                # ie2amr_edge_map[tuple(_edge)] = {'amr_node':amr_node_path,'amr_rel':amr_rel_path,'is_event':is_event,'ie_rel':ie_egde_type}
                if len(ie2amr_edge_map[ie_edge]) == 0:
                    scores.append(0.0)
                    continue
                # labels[_idx][_idx] = 1.0
                ie_startnode = ie_edge[0]
                ie_endnode =  ie_edge[1]
                # print('ie node has edge: ', ie_i.has_edges_between(begin_ie_node, end_ie_node))
                # begin_amr_node = ie2amr_map[begin_ie_node]
                # end_amr_node = ie2amr_map[end_ie_node]
                sub_amr_nodes = ie2amr_edge_map[ie_edge]['amr_node']
                sub_amr_rels = ie2amr_edge_map[ie_edge]['amr_rel']
                
                # get representations
                sub_amr_node_repr = torch.index_select(amr_repr_i,0,sub_amr_nodes) # num of amr nodes on path
                sub_amr_rel_repr = self.dropout_emb(self.amr_edge_type_emb(sub_amr_rels)) # num of amr rels
                # sub_ie_role_repr = self.arg_role_emb[ie2amr_edge_map[ie_edge]['ie_edge_type']]
                ie_startnode_repr = ie_repr_i[ie_startnode]
                ie_endnode_repr = ie_repr_i[ie_endnode]
                
                # may used latter
                # ie_startnode_type_repr = self.dropout_emb(self.ie_node_type_emb(ie2amr_edge_map[ie_edge]['ie_node_type'][0]))
                # ie_endnode_type_repr = self.dropout_emb(self.ie_node_type_emb(ie2amr_edge_map[ie_edge]['ie_node_type'][1]))

                # [start token] [rel] .... [rel] [end token]
                sub_amr_repr = []
                for _idx in range(sub_amr_rel_repr.size()[0]):
                    if _idx == 0:
                        sub_amr_repr.append(ie_startnode_repr)
                    else:
                        sub_amr_repr.append(sub_amr_node_repr[_idx])
                    sub_amr_repr.append(sub_amr_rel_repr[_idx])
                sub_amr_repr.append(ie_endnode_repr)

                
                # sub_amr_repr = [sub_ie_startnode_type_repr] + sub_amr_repr + [sub_ie_endnode_type_repr]
                sub_amr_repr = torch.stack(sub_amr_repr,dim=0)
                if self.use_pos_emb:
                    pos_emb = self.pos_emb(bert_i.new_tensor(torch.arange(sub_amr_repr.shape[0]),dtype=torch.long))
                    sub_amr_repr = sub_amr_repr + pos_emb
                sub_amr_repr = sub_amr_repr.unsqueeze(0)
                # contextualized layer
                for _att_layer in self.amr_attention:
                    sub_amr_repr = _att_layer(sub_amr_repr) # 1, amr_path_length, H
                
                pooled_amr_repr = self.gelu(torch.mean(sub_amr_repr, 1)) # 1, H
                # print(ie_role_repr.shape)
                # print(ie_startnode_type_repr.shape)
                # print(pooled_amr_repr.shape)
                # print(ie_endnode_type_repr.shape)
                # ie_startnode_type_repr = ie_startnode_type_repr.expand(self.role_type_num, -1)
                # ie_endnode_type_repr = ie_endnode_type_repr.expand(self.role_type_num, -1)
                ie_role_idx = ie2amr_edge_map[ie_edge]['ie_edge_type']
                ie_role_repr = self.dropout_emb(self.arg_role_emb(bert_i.new_tensor(ie_role_idx,dtype=torch.long))) # arg role types, H
                # ff similarity function
                if self.similarity_fc_type == 'linear':
                    pooled_amr_repr = pooled_amr_repr.expand(self.role_type_num, -1)
                    ie_amr_repr = torch.cat([ie_role_repr,pooled_amr_repr],-1) # arg role, 2H
                    # ie_amr_repr = self.dropout_repr(ie_amr_repr)
                    logit = self.similarity_fc(ie_amr_repr) # arg role, 1
                elif self.similarity_fc_type == 'bilinear':
                    # print(ie_role_repr.shape, pooled_amr_repr.shape)
                    logit = self.similarity_fc(ie_role_repr, pooled_amr_repr.unsqueeze(1))
                    prob = self.sigmoid(logit.squeeze(0))
                    scores.append(prob.item()*2.0-1.0)
                    ie2amr_edge_map[ie_edge]['human_readable']['score'] = prob.item()*2.0-1.0
                    print('####')
                    print(batch.gold_events[b])
                    human = ie2amr_edge_map[ie_edge]['human_readable']
                    print((human['trigger_type'],human['trigger_token'][0],human['entity_token'][0],human['entity_type'],human['role_type'],human['score']))
                    # print all scores
                    with torch.no_grad():
                        ie_role_repr = self.dropout_emb(self.arg_role_emb(bert_i.new_tensor(torch.arange(self.role_type_num),dtype=torch.long)))
                        logit = self.similarity_fc(ie_role_repr, pooled_amr_repr.unsqueeze(1)) # 23, H  *  1,1,H
                        all_scores = (self.sigmoid(logit.squeeze(0))*2.0 -1.0).detach().cpu().tolist()
                        print([ (self.role_type_itos[_i],round(score[0],5)) for _i, score in enumerate(all_scores)])





        # print(scores)
                
                
        # print('batch loss: ',loss)

        return scores




"""
    def predict(self, batch, epoch):
        self.eval()

        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens,
                                   batch.amr)
        batch_size, _, _ = bert_outputs.size()

        # identification
        entity_label_scores = self.entity_label_ffn(bert_outputs)
        entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
        trigger_label_scores = self.trigger_label_ffn(bert_outputs)
        trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
        _, entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores,
                                                               batch.token_nums)
        _, trigger_label_preds = self.trigger_crf.viterbi_decode(trigger_label_scores,
                                                                 batch.token_nums)
        entities = tag_paths_to_spans(entity_label_preds,
                                      batch.token_nums,
                                      self.entity_label_stoi)
        triggers = tag_paths_to_spans(trigger_label_preds,
                                      batch.token_nums,
                                      self.trigger_label_stoi)
        node_graphs = [Graph(e, t, [], [], self.vocabs)
                       for e, t in zip(entities, triggers)]
        scores = self.scores(bert_outputs, node_graphs, batch.amr, batch.align, batch.exist, epoch, predict=True)
        max_entity_num = max(max(len(seq_entities) for seq_entities in entities), 1)

        batch_graphs = []
        # Decode each sentence in the batch
        for i in range(batch_size):
            seq_entities, seq_triggers = entities[i], triggers[i]
            amr_i = batch.amr[i]
            spans = sorted([(*i, True) for i in seq_entities] + [(*i, False) for i in seq_triggers], key=lambda x: (x[0], x[1], not x[-1]))
            entity_num, trigger_num = len(seq_entities), len(seq_triggers)
            if entity_num == 0 and trigger_num == 0:
                # skip decoding
                batch_graphs.append(Graph.empty_graph(self.vocabs))
                continue
            graph = self.decode(spans, amr_i,
                                entity_type_scores=scores[0][i],
                                mention_type_scores=scores[1][i],
                                event_type_scores=scores[2][i],
                                relation_type_scores=scores[3][i],
                                role_type_scores=scores[4][i],
                                entity_num=max_entity_num)
            batch_graphs.append(graph)

        self.train()
        return batch_graphs

    def compute_graph_scores(self, graphs, scores):
        (
            entity_type_scores, _mention_type_scores,
            trigger_type_scores, relation_type_scores,
            role_type_scores
        ) = scores
        label_idxs = graphs_to_label_idxs(graphs)
        label_idxs = [entity_type_scores.new_tensor(idx,
                                               dtype=torch.long if i % 2 == 0
                                               else torch.float)
                      for i, idx in enumerate(label_idxs)]
        (
            entity_idxs, entity_mask, trigger_idxs, trigger_mask,
            relation_idxs, relation_mask, role_idxs, role_mask
        ) = label_idxs
        # Entity score
        entity_idxs = entity_idxs.unsqueeze(-1)
        entity_scores = torch.gather(entity_type_scores, 2, entity_idxs)
        entity_scores = entity_scores.squeeze(-1) * entity_mask
        entity_score = entity_scores.sum(1)
        # Trigger score
        trigger_idxs = trigger_idxs.unsqueeze(-1)
        trigger_scores = torch.gather(trigger_type_scores, 2, trigger_idxs)
        trigger_scores = trigger_scores.squeeze(-1) * trigger_mask
        trigger_score = trigger_scores.sum(1)
        # Relation score
        relation_idxs = relation_idxs.unsqueeze(-1)
        relation_scores = torch.gather(relation_type_scores, 2, relation_idxs)
        relation_scores = relation_scores.squeeze(-1) * relation_mask
        relation_score = relation_scores.sum(1)
        # Role score
        role_idxs = role_idxs.unsqueeze(-1)
        role_scores = torch.gather(role_type_scores, 2, role_idxs)
        role_scores = role_scores.squeeze(-1) * role_mask
        role_score = role_scores.sum(1)

        score = entity_score + trigger_score + role_score + relation_score

        global_vectors = [generate_global_feature_vector(g, self.global_feature_maps, features=self.global_features)
                          for g in graphs]
        global_vectors = entity_scores.new_tensor(global_vectors)
        global_weights = self.global_feature_weights.unsqueeze(0).expand_as(global_vectors)
        global_score = (global_vectors * global_weights).sum(1)
        score = score + global_score

        return score

    def generate_locally_top_graphs(self, graphs, scores):
        (
            entity_type_scores, _mention_type_scores,
            trigger_type_scores, relation_type_scores,
            role_type_scores
        ) = scores
        max_entity_num = max(max([g.entity_num for g in graphs]), 1)
        top_graphs = []
        for graph_idx, graph in enumerate(graphs):
            entity_num = graph.entity_num
            trigger_num = graph.trigger_num
            _, top_entities = entity_type_scores[graph_idx].max(1)
            top_entities = top_entities.tolist()[:entity_num]
            top_entities = [(i, j, k) for (i, j, _), k in
                            zip(graph.entities, top_entities)]
            _, top_triggers = trigger_type_scores[graph_idx].max(1)
            top_triggers = top_triggers.tolist()[:trigger_num]
            top_triggers = [(i, j, k) for (i, j, _), k in
                            zip(graph.triggers, top_triggers)]
            # _, top_relations = relation_type_scores[graph_idx].max(1)
            # top_relations = top_relations.tolist()
            # top_relations = [(i, j, top_relations[i * max_entity_num + j])
            #                  for i in range(entity_num) for j in
            #                  range(entity_num)
            #                  if i < j and top_relations[i * max_entity_num + j] != 'O']
            top_relation_scores, top_relation_labels = relation_type_scores[graph_idx].max(1)
            top_relation_scores = top_relation_scores.tolist()
            top_relation_labels = top_relation_labels.tolist()
            top_relations = [(i, j) for i, j in zip(top_relation_scores, top_relation_labels)]
            top_relation_list = []
            for i in range(entity_num):
                for j in range(entity_num):
                    if i < j:
                        score_1, label_1 = top_relations[i * max_entity_num + j]
                        score_2, label_2 = top_relations[j * max_entity_num + i]
                        if score_1 > score_2 and label_1 != 'O':
                            top_relation_list.append((i, j, label_1))
                        if score_2 > score_1 and label_2 != 'O': 
                            top_relation_list.append((j, i, label_2))

            _, top_roles = role_type_scores[graph_idx].max(1)
            top_roles = top_roles.tolist()
            top_roles = [(i, j, top_roles[i * max_entity_num + j])
                         for i in range(trigger_num) for j in range(entity_num)
                         if top_roles[i * max_entity_num + j] != 'O']
            top_graphs.append(Graph(
                entities=top_entities,
                triggers=top_triggers,
                # relations=top_relations,
                relations=top_relation_list,
                roles=top_roles,
                vocabs=graph.vocabs
            ))
        return top_graphs

    def trim_beam_set(self, beam_set, beam_size):
        if len(beam_set) > beam_size:
            beam_set.sort(key=lambda x: self.compute_graph_score(x), reverse=True)
            beam_set = beam_set[:beam_size]
        return beam_set

    def compute_graph_score(self, graph):
        score = graph.graph_local_score
        if self.use_global_features:
            global_vector = generate_global_feature_vector(graph,
                                                           self.global_feature_maps,
                                                           features=self.global_features)
            global_vector = self.global_feature_weights.new_tensor(global_vector)
            global_score = global_vector.dot(self.global_feature_weights).item()
            score = score + global_score
        return score

    def decode(self, spans, amr, entity_type_scores, mention_type_scores, event_type_scores,
               relation_type_scores, role_type_scores, entity_num):


        prior_list = amr.ndata["priority"].squeeze(1).tolist()
        token_pos_list = amr.ndata["token_pos"].squeeze(1).tolist()

        split_order_list = []
        ent_num, trig_num = 0, 0
        for i in range(len(spans)):
            if spans[i][3]:
                split_order_list.append(ent_num)
                ent_num += 1
            else:
                split_order_list.append(trig_num)
                trig_num += 1
        # print("split", split_order_list)
        align_list = [len(prior_list)+10 for _ in range(len(spans))]
        for i in range(len(spans)):
            start_i = spans[i][0]
            end_i = spans[i][1] - 1
            for j in range(len(token_pos_list)):
                if token_pos_list[j] >= start_i and token_pos_list[j] <= end_i:
                    align_list[i] = prior_list[j]
                    break
        # after we get an alignment list, we try to get the priority and sort the spans
        align_list = np.array(align_list)
        index_list = np.argsort(align_list)
        new_spans = []
        for i in range(len(spans)):
            new_spans.append(spans[index_list[i]])

        new_ent_score = entity_type_scores.new_zeros(entity_type_scores.size())
        new_trig_score = event_type_scores.new_zeros(event_type_scores.size())
        new_rela_score = relation_type_scores.new_zeros(relation_type_scores.size())
        new_role_score = role_type_scores.new_zeros(role_type_scores.size())

        # build up an individual entity and trigger list
        entity_index_list, trig_index_list = [], []

        for i in range(len(new_spans)):
            if new_spans[i][3]:
                entity_index_list.append(split_order_list[index_list[i]])
            else:
                trig_index_list.append(split_order_list[index_list[i]])

        # change the order of ent and trig score
        for i in range(ent_num):
            new_ent_score[i] = entity_type_scores[entity_index_list[i]]
        for i in range(trig_num):
            new_trig_score[i] = event_type_scores[trig_index_list[i]]

        # change the order of relation matrix
        for i in range(ent_num):
            for j in range(ent_num):
                new_rela_score[i * entity_num + j] = relation_type_scores[entity_index_list[i] * entity_num + entity_index_list[j]]

        # cahnge the order of role matrix
        for i in range(trig_num):
            for j in range(ent_num):
                new_role_score[i * entity_num + j] = role_type_scores[trig_index_list[i] * entity_num + entity_index_list[j]]


        beam_set = [Graph.empty_graph(self.vocabs)]
        entity_idx, trigger_idx = 0, 0

        for start, end, _, is_entity_node in new_spans:
            # 1. node step
            if is_entity_node:
                node_scores = new_ent_score[entity_idx].tolist()
                # print(node_scores)
            else:
                node_scores = new_trig_score[trigger_idx].tolist()
            node_scores_norm = normalize_score(node_scores)
            # node_scores = [(s, i) for i, s in enumerate(node_scores)]
            node_scores = [(s, i, n) for i, (s, n) in enumerate(zip(node_scores,
                                                                node_scores_norm))]
            node_scores.sort(key=lambda x: x[0], reverse=True)
            top_node_scores = node_scores[:self.beta_v]

            beam_set_ = []
            for graph in beam_set:
                for score, label, score_norm in top_node_scores:
                    graph_ = graph.copy()
                    if is_entity_node:
                        graph_.add_entity(start, end, label, score, score_norm)
                    else:
                        graph_.add_trigger(start, end, label, score, score_norm)
                    beam_set_.append(graph_)
            beam_set = beam_set_

            # 2. edge step
            if is_entity_node:
                # add a new entity: new relations, new argument roles
                for i in range(entity_idx):
                    # add relation edges
                    # edge_scores_1 = relation_type_scores[i * entity_num + entity_idx].tolist()
                    # edge_scores_2 = relation_type_scores[entity_idx * entity_num + i].tolist()
                    edge_scores_1 = new_rela_score[i * entity_num + entity_idx].tolist()
                    edge_scores_2 = new_rela_score[entity_idx * entity_num + i].tolist()
                    # print(relation_type_scores[i * entity_num + entity_idx])

                    # edge_scores_norm_1 = (edge_scores_1 / 10.0).softmax(0).tolist()
                    # edge_scores_norm_2 = (edge_scores_2 / 10.0).softmax(0).tolist()
                    # edge_scores_1 = edge_scores_1.tolist()
                    # edge_scores_2 = edge_scores_2.tolist()
                    edge_scores_norm_1 = normalize_score(edge_scores_1)
                    edge_scores_norm_2 = normalize_score(edge_scores_2)

                    if self.relation_directional:
                        edge_scores = [(max(s1, s2), n2 if s1 < s2 else n1, i, s1 < s2)
                                       for i, (s1, s2, n1, n2)
                                       in enumerate(zip(edge_scores_1, edge_scores_2,
                                                        edge_scores_norm_1,
                                                        edge_scores_norm_2))]
                        null_score = edge_scores[0][0]
                        edge_scores.sort(key=lambda x: x[0], reverse=True)
                        top_edge_scores = edge_scores[:self.beta_e]
                    else:
                        edge_scores = [(max(s1, s2), n2 if s1 < n2 else n1, i, False)
                                       for i, (s1, s2, n1, n2)
                                       in enumerate(zip(edge_scores_1, edge_scores_2,
                                                        edge_scores_norm_1,
                                                        edge_scores_norm_2))]
                        null_score = edge_scores[0][0]
                        edge_scores.sort(key=lambda x: x[0], reverse=True)
                        top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, score_norm, label, inverse in top_edge_scores:
                            rel_cur_ent = label * 100 + graph.entities[-1][-1]
                            rel_pre_ent = label * 100 + graph.entities[i][-1]
                            if label == 0 or (rel_pre_ent in self.valid_relation_entity and
                                              rel_cur_ent in self.valid_relation_entity):
                                graph_ = graph.copy()
                                if self.relation_directional and inverse:
                                    graph_.add_relation(entity_idx, i, label, score, score_norm)
                                else:
                                    graph_.add_relation(i, entity_idx, label, score, score_norm)
                                beam_set_.append(graph_)
                                has_valid_edge = True
                        if not has_valid_edge:
                            graph_ = graph.copy()
                            graph_.add_relation(i, entity_idx, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 200:
                        beam_set = self.trim_beam_set(beam_set, self.beam_size)

                for i in range(trigger_idx):
                    # add argument role edges
                    edge_scores = new_role_score[i * entity_num + entity_idx].tolist()
                    edge_scores_norm = normalize_score(edge_scores)
                    edge_scores = [(s, i, n) for i, (s, n) in enumerate(zip(edge_scores, edge_scores_norm))]
                    null_score = edge_scores[0][0]
                    edge_scores.sort(key=lambda x: x[0], reverse=True)
                    top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, label, score_norm in top_edge_scores:
                            role_entity = label * 100 + graph.entities[-1][-1]
                            event_role = graph.triggers[i][-1] * 100 + label
                            if label == 0 or (event_role in self.valid_event_role and
                                              role_entity in self.valid_role_entity):
                                graph_ = graph.copy()
                                graph_.add_role(i, entity_idx, label, score, score_norm)
                                beam_set_.append(graph_)
                                has_valid_edge = True
                        if not has_valid_edge:
                            graph_ = graph.copy()
                            graph_.add_role(i, entity_idx, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 100:
                        beam_set = self.trim_beam_set(beam_set, self.beam_size)
                beam_set = self.trim_beam_set(beam_set_, self.beam_size)

            else:
                # add a new trigger: new argument roles
                for i in range(entity_idx):
                    edge_scores = new_role_score[trigger_idx * entity_num + i].tolist()
                    edge_scores_norm = normalize_score(edge_scores)
                    edge_scores = [(s, i, n) for i, (s, n) in enumerate(zip(edge_scores,
                                                                            edge_scores_norm))]
                    null_score = edge_scores[0][0]
                    edge_scores.sort(key=lambda x: x[0], reverse=True)
                    top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, label, score_norm in top_edge_scores:
                            event_role = graph.triggers[-1][-1] * 100 + label
                            role_entity = label * 100 + graph.entities[i][-1]
                            if label == 0 or (event_role in self.valid_event_role
                                              and role_entity in self.valid_role_entity):
                                graph_ = graph.copy()
                                graph_.add_role(trigger_idx, i, label, score, score_norm)
                                beam_set_.append(graph_)
                                has_valid_edge = True
                        if not has_valid_edge:
                            graph_ = graph.copy()
                            graph_.add_role(trigger_idx, i, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 100:
                        beam_set = self.trim_beam_set(beam_set, self.beam_size)

                beam_set = self.trim_beam_set(beam_set_, self.beam_size)

            if is_entity_node:
                entity_idx += 1
            else:
                trigger_idx += 1
        beam_set.sort(key=lambda x: self.compute_graph_score(x), reverse=True)
        graph = beam_set[0]

        # predict mention types
        _, mention_types = mention_type_scores.max(dim=1)
        mention_types = mention_types[:entity_idx]
        mention_list = [(i, j, l.item()) for (i, j, k), l
                        in zip(graph.entities, mention_types)]
        graph.mentions = mention_list

        return graph
"""

    
    



