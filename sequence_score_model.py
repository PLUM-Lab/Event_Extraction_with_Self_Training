"This version allows to enforce valid trigger,entity -> role patterns"


import copy
import random
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np

from graph import Graph
from transformers import BertModel, RobertaModel
from global_feature import generate_global_feature_vector, generate_global_feature_maps
from util import normalize_score
from self_attention_encoder import SimpleSelfAttention, BilinearMatrixAttention
from score_util import get_ie2amr_role

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

def load_valid_pattern(pattern_path, vocabs):
    event_type_vocab = vocabs['event_type']
    entity_type_vocab = vocabs['entity_type']
    relation_type_vocab = vocabs['relation_type']
    role_type_vocab = vocabs['role_type']

    valid_event_role = {}
    event_role = json.load(open(os.path.join(pattern_path, 'event_role.json'), 'r', encoding='utf-8'))
    for event, roles in event_role.items():
        if event not in event_type_vocab:
            continue
        event_type_idx = event_type_vocab[event]
        if not event_type_idx in valid_event_role:
            valid_event_role[event_type_idx] = set()
        for role in roles:
            if role not in role_type_vocab:
                continue
            role_type_idx = role_type_vocab[role]
            valid_event_role[event_type_idx].add(role_type_idx)
    
    valid_entity_role = {}
    role_entity = json.load(open(os.path.join(pattern_path, 'role_entity.json'), 'r', encoding='utf-8'))
    for role, entities in role_entity.items():
        if role not in role_type_vocab:
            continue
        role_type_idx = role_type_vocab[role]
        for entity in entities:
            # print(entity, entity_type_vocab)
            if not entity in entity_type_vocab:
                continue
            entity_type_idx = entity_type_vocab[entity]
            # print(entity_type_idx, valid_entity_role)
            if not entity_type_idx in valid_entity_role:
                valid_entity_role[entity_type_idx] = set()
            valid_entity_role[entity_type_idx].add(role_type_idx)

    # print('entity ',valid_entity_role)
    # print('event ',valid_event_role)
    valid_event_entity_role = {}
    for event in valid_event_role:
        for entity in valid_entity_role:
            if not (event,entity) in valid_event_entity_role:
                valid_event_entity_role[(event,entity)] = [0.0]*len(role_type_vocab)
                valid_event_entity_role[(event,entity)][role_type_vocab['O']] = 1.0
            for role in valid_event_role[event].intersection(valid_entity_role[entity]):
                valid_event_entity_role[(event,entity)][role] = 1.0
    return valid_event_entity_role
          
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


class SeqScoreGraph(nn.Module):
    def __init__(self,
                 config,
                 vocabs):
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

        # embeddings
        self.trigger_emb = nn.Embedding(self.event_type_num, self.bert_dim)
        self.role_emb = nn.Embedding(self.role_type_num, self.bert_dim)
        self.amr_rel_emb = nn.Embedding(self.amr_edge_type_num, self.bert_dim)
        self.pos_emb = nn.Embedding(128,self.bert_dim)
        self.token_type_emb = nn.Embedding(3,self.bert_dim)
        
        # pooler
        self.attention_layers = nn.ModuleList([SimpleSelfAttention(self.bert_dim, config.intermediate_dim, config.num_attention_heads,hidden_dropout_prob=config.hidden_drop,attention_probs_dropout_prob=config.attention_drop) for i in range(config.num_attention_layers)])
                
        self.use_trigger_type = config.use_trigger_type
        self.use_token_type = config.use_token_type
        # self.use_entity_type = config.use_entity_type
        self.use_pos_emb = config.use_pos_emb
        self.amr_path_reduce = config.amr_path_reduce
        self.use_bce_loss = config.use_bce_loss
        self.use_marginal_loss = config.use_marginal_loss
        self.loss_ratio = config.loss_ratio

        # functions
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.bce_loss = nn.BCELoss()

        # classify layer
        if self.amr_path_reduce == 'token':
            linear_dim = 2 * self.bert_dim
        elif self.amr_path_reduce == 'token_role':
            linear_dim = 3 * self.bert_dim
        elif self.amr_path_reduce == 'token_trigger_role':
            linear_dim = 4 * self.bert_dim
        else:
            linear_dim = self.bert_dim
        self.classify_layer = nn.Linear(linear_dim,1)

        # predict
        self.use_valid_pattern = config.use_valid_pattern
        self.valid_event_entity_role = load_valid_pattern(config.valid_pattern_path, vocabs)

        

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

    def scores(self, bert_outputs, batch, ie2amr_edge_map_list=None):
        batch_loss = []
        batch_size, _, _ = bert_outputs.size()
        logit_list = []
        tag_list = []
        human_readable = []
        entity_num = 0
        trigger_num = 0
        for b in range(batch_size):
            bert_b = bert_outputs[b]
            ie_graph = batch.graphs[b]
            entity_num = ie_graph.entity_num if ie_graph.entity_num > entity_num else entity_num
            trigger_num = ie_graph.trigger_num if ie_graph.trigger_num > trigger_num else trigger_num
            if ie_graph.trigger_num == 0 or ie_graph.entity_num == 0:
                continue
            if ie2amr_edge_map_list is None:
                ie2amr_edge_map = batch.ie2amr_edge_map[b]
            else:
                ie2amr_edge_map = ie2amr_edge_map_list[b]
            amr_graph = batch.amr[b]
            amr_span_list = amr_graph.ndata["token_span"].tolist()
            amr_repr = generate_amr_reprs(amr_span_list, bert_b) # (num_span, bert_dim)
            ie_span_list = [[span[0],span[1]] for span in ie_graph.triggers] + [[span[0],span[1]] for span in ie_graph.entities]
            ie_repr = generate_amr_reprs(ie_span_list, bert_b) # (num_span, bert_dim)
            num_trigger = ie_graph.trigger_num

            for ie_edge, amr_info in  ie2amr_edge_map.items():
                human_readable.append(amr_info['human_readable'])
                trigger_idx, entity_idx = ie_edge
                amr_node_path = amr_info['amr_node']
                amr_node_path = bert_b.new_tensor(amr_node_path,dtype=torch.long)
                amr_rel_path = amr_info['amr_rel']
                amr_rel_path = bert_b.new_tensor(amr_rel_path,dtype=torch.long)
                trigger_type = amr_info['trigger_type']
                entity_type = amr_info['entity_type']
                role_type = amr_info['role_type']
                amr_node_repr = torch.index_select(amr_repr,0,amr_node_path)
                amr_rel_repr = self.amr_rel_emb(amr_rel_path)
                trigger_repr = ie_repr[trigger_idx]
                entity_repr = ie_repr[num_trigger+entity_idx]

                amr_path_repr = []
                token_type = []
                for _idx in range(amr_rel_repr.size()[0]):
                    amr_path_repr.append(amr_node_repr[_idx])
                    token_type.append(0)
                    amr_path_repr.append(amr_rel_repr[_idx])
                    token_type.append(1)
                amr_path_repr.append(amr_node_repr[-1])
                token_type.append(0)
                amr_path_repr[0] = trigger_repr
                amr_path_repr[-1] = entity_repr
                # amr_path_repr.append(self.role_emb[role_type]) 
                # token_type.append(2)
                if self.use_trigger_type:
                    trigger_type_t = bert_b.new_tensor([trigger_type],dtype=torch.long).squeeze(0)
                    amr_path_repr = [self.trigger_emb(trigger_type_t)] + amr_path_repr
                    token_type = [2] + token_type
                amr_path_repr = torch.stack(amr_path_repr,dim=0).unsqueeze(0) #1, S, H
                amr_path_repr = amr_path_repr.expand(self.role_type_num,-1,-1) # num_role, S, H
                # print('amr_path_repr',amr_path_repr.shape)
                all_emb_idx = torch.arange(self.role_type_num).to(dtype=torch.long, device=bert_b.device)
                role_repr = self.role_emb(all_emb_idx).unsqueeze(1) # num_role, 1, H
                # print('role_repr',role_repr.shape)
                amr_path_repr = torch.cat([amr_path_repr,role_repr],dim=1)
                # print('amr_path_repr',amr_path_repr.shape)
                token_type.append(2)
                token_type = bert_b.new_tensor(token_type,dtype=torch.long)
                token_type_repr = self.token_type_emb(token_type).unsqueeze(0) # 1, S, H
                amr_path_len = amr_path_repr.shape[1]
                pos_repr = self.pos_emb(torch.arange(amr_path_len).to(dtype=torch.long, device=bert_b.device)).unsqueeze(0) # 1, S, H
                
                if self.use_pos_emb and self.use_token_type:
                    context_amr_repr = amr_path_repr + token_type_repr + pos_repr
                elif self.use_pos_emb:
                    context_amr_repr = amr_path_repr + pos_repr
                elif self.use_token_type:
                    context_amr_repr = amr_path_repr + token_type_repr
                else:
                    context_amr_repr = amr_path_repr
                for att_layer in self.attention_layers:
                    context_amr_repr = att_layer(context_amr_repr)
                
                if self.amr_path_reduce == "mean":
                    pooled_amr_repr = torch.mean(context_amr_repr,dim=1)
                elif self.amr_path_reduce == "trigger":
                    assert self.use_trigger_type
                    pooled_amr_repr = context_amr_repr[:,0,:] # num_role, H
                elif self.amr_path_reduce == "sum":
                    pooled_amr_repr = torch.sum(context_amr_repr,dim=1) # num_role, H
                elif self.amr_path_reduce == "token":
                    if self.use_trigger_type:
                        trigger_repr = context_amr_repr[:,1,:]
                    else:
                        trigger_repr = context_amr_repr[:,0,:]
                    entity_repr = context_amr_repr[:,-2,:]
                    pooled_amr_repr = torch.cat([trigger_repr,entity_repr],dim=-1) # num_role, 2H
                elif self.amr_path_reduce == "role":
                    pooled_amr_repr = context_amr_repr[:,-1,:]
                elif self.amr_path_reduce == "token_role":
                    if self.use_trigger_type:
                        trigger_repr = context_amr_repr[:,1,:]
                    else:
                        trigger_repr = context_amr_repr[:,0,:]
                    entity_repr = context_amr_repr[:,-2,:]
                    role_repr = context_amr_repr[:,-1,:]
                    pooled_amr_repr = torch.cat([trigger_repr,entity_repr,role_repr],dim=-1) # num_role, 3H
                elif self.amr_path_reduce == "token_trigger_role":
                    assert self.use_trigger_type
                    trigger_repr = context_amr_repr[:,1,:]
                    trigger_type_repr = context_amr_repr[:,0,:]
                    entity_repr = context_amr_repr[:,-2,:]
                    role_repr = context_amr_repr[:,-1,:]
                    pooled_amr_repr = torch.cat([trigger_repr,entity_repr,trigger_type_repr,role_repr],dim=-1) # num_role, 3H
                else:
                    raise NotImplementedError
                logit = self.classify_layer(pooled_amr_repr).squeeze(-1) # num_role
                # print('logit',logit.shape)
                logit_list.append(logit)
                tag_list.append(role_type)
        targets = bert_b.new_tensor(tag_list, dtype=torch.long) # B
        if len(logit_list) == 0:
            return {'probs':None,'entity_num':entity_num,'trigger_num':trigger_num, 'human_readable':None}
        logits = torch.stack(logit_list,dim=0) # B, num_role
        probs = self.sigmoid(logits)
        targets_one_hot = F.one_hot(targets,num_classes=self.role_type_num).to(torch.float)
        output = {'logits':logits,'targets':targets,'targets_one_hot':targets_one_hot,'probs':probs,'human_readable':human_readable,'entity_num':entity_num,'trigger_num':trigger_num}
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

        output = self.scores(bert_outputs, batch)
        loss = self.loss_function(output)
        output['loss'] = loss
        return output

    def predict(self,batch):
        self.eval()
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size = bert_outputs.shape[0]
        ie2amr_edge_map_list = []
        for i in range(batch_size):
            ie2amr_node_map, ie2amr_edge_map = get_ie2amr_role(batch.amr[i],batch.align[i],batch.exist[i],batch.amr_bidir_graph[i],batch.amr_edge2type[i],batch.graphs[i],self.vocabs,batch.tokens[i])
            ie2amr_edge_map_list.append(ie2amr_edge_map)
        output = self.scores(bert_outputs,batch,ie2amr_edge_map_list=ie2amr_edge_map_list)
        if not output['probs'] is None:
            probs = output['probs'].detach().cpu().tolist()
        else:
            probs = output['probs']
        # max_probs, max_args = torch.max(probs,-1) # B
        # max_probs = max_probs.detach().cpu().tolist()
        # max_args = max_args.detach().cpu().tolist()
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
    
    def evaluate(self,batch, gold_graphs):

        accu_result = {'total_num':0.0, 'tp':0.0, 'fp':0.0,'fn':0.0, 'gtp':0.0, 'gfp': 0.0, 'gfn': 0.0}

        # for role_type in self.role_type_stoi:
        #     each_type_result[role_type] = {'total':0,'total_correct':0,'total_false':0,'tp':0,'tp_correct':0.0,'tp_false':0.0,'score_correct':0,'score_correct_accu':0.0,'score_false':0,'score_false_accu':0.0,'pred_score':0.0,'gold_score':0.0,'instances_correct':[],'instances_false':[]}

        def convert_arguments(triggers, entities, roles):
            args = set()
            args_id = {}
            for trigger_idx, entity_idx, role in roles:
                arg_start, arg_end, _ = entities[entity_idx]
                trigger_label = triggers[trigger_idx][-1]
                args.add((arg_start, arg_end, trigger_label, role))
                args_id[(arg_start, arg_end, trigger_label)] = role
            return args, args_id

        self.eval()
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size = bert_outputs.shape[0]
        ie2amr_edge_map_list = []
        for i in range(batch_size):
            ie2amr_node_map, ie2amr_edge_map = get_ie2amr_role(batch.amr[i],batch.align[i],batch.exist[i],batch.amr_bidir_graph[i],batch.amr_edge2type[i],batch.graphs[i],self.vocabs,batch.tokens[i])
            ie2amr_edge_map_list.append(ie2amr_edge_map)
        output = self.scores(bert_outputs,batch,ie2amr_edge_map_list=ie2amr_edge_map_list)
        if not output['probs'] is None:
            probs = output['probs'].detach().cpu().tolist()
        else:
            probs = output['probs']
        human_readable = output['human_readable']
        # max_probs, max_args = torch.max(probs,-1) # B
        # max_probs = max_probs.detach().cpu().tolist()
        # max_args = max_args.detach().cpu().tolist()
        O_idx = self.role_type_stoi['O']
        role_idx = 0
        tp = 0
        total_gold_score = 0.0
        total_pred_score = 0.0
        total_pred_role_num = 0
        for pred_graph, gold_graph in zip(batch.graphs, gold_graphs):
            gold_triggers = gold_graph.triggers
            pred_triggers = pred_graph.triggers
            gold_entities = gold_graph.entities
            pred_entities = pred_graph.entities
            gold_args, gold_args_ids = convert_arguments(gold_triggers, gold_entities,
                                      gold_graph.roles)
            
            # pred_args = {(role[0],role[1]):role[2] for role in pred_graph.roles}
            pred_args = { (role[0],role[1]):(role[2],role_score) for role, role_score in zip(pred_graph.roles, pred_graph.role_scores)}
            for tri_idx in range(pred_graph.trigger_num):
                for ent_idx in range(pred_graph.entity_num):
                    human = human_readable[role_idx]
                    # if len(human['amr_path'])<=5:
                    #     role_idx += 1
                    #     continue
                    prob = probs[role_idx]
                    pred_role_type, pred_score = pred_args.get((tri_idx,ent_idx))
                    pred_arg_start = pred_graph.entities[ent_idx][0]
                    pred_arg_end = pred_graph.entities[ent_idx][1]
                    pred_trigger_type = pred_graph.triggers[tri_idx][2]
                    pred_arg = (pred_arg_start,pred_arg_end,pred_trigger_type,pred_role_type)
                    pred_arg_id = (pred_arg_start,pred_arg_end,pred_trigger_type)
                    in_gold = False
                    in_gold_id = False
                    if pred_arg in gold_args:
                        in_gold = True
                        gold_role = self.role_type_itos[pred_role_type]
                    elif pred_arg_id in gold_args_ids:
                        in_gold_id = True
                        gold_role = self.role_type_itos[gold_args_ids[pred_arg_id]]
                    else:
                        gold_role = 'O'

                    human = human_readable[role_idx]
                    out_str = "tokens: {},\n pred_event_type: {}, pred_event_token: {}, pred_entity: {}, \n gold_role: {}, pred_role: {}, pred_score:{}, \nscore_pred:{}, score_gold:{}, \n amr_path: {}".format(" ".join(human['tokens']),human['trigger_type'],human['trigger_token'][0], human['entity_token'][0],gold_role,human['role_type'],pred_score,prob[pred_role_type],prob[self.role_type_stoi[gold_role]],human['amr_path'])
                    # print(out_str)
                    # accu_result = {'corr_corr_num':0 'false_corr_num':0, 'both_corr_num':0 ,'total_corr_num':0, 'total_false_num':, 'total_num':0}
                    pred_role = self.role_type_itos[pred_role_type]
                    accu_result['total_num'] += 1

                    if pred_role_type == O_idx:
                        if in_gold_id: # false
                            if prob[pred_role_type] < 0.5: # false
                                pass
                            else: # true
                                accu_result['fp'] += 1
                                

                        elif not in_gold_id: # true
                            if prob[pred_role_type] > 0.5: # true
                                accu_result['tp'] += 1
                            else: # false
                                accu_result['fn'] += 1

                    else:
                        if in_gold: #true
                            if prob[pred_role_type] > 0.5: # true
                                accu_result['tp']
                            else: #false
                                accu_result['fn'] += 1
                                
                        elif not in_gold: # false
                            if prob[pred_role_type] < 0.5: # false
                                pass
                            else: # true
                                accu_result['fp'] += 1
                    
                    if prob[self.role_type_stoi[gold_role]] >= 0.5:
                        accu_result['gtp'] += 1
                    else:
                        accu_result['gfn'] += 1
                    for s,i in self.role_type_stoi.items():
                        if not s == gold_role:
                            # print(gold_role, s,i, prob[i])
                            if prob[i] >= 0.5:
                                accu_result['gfp'] += 1
                                
                    
                    # # ----------------------- debug ------------------------- #
                    if prob[self.role_type_stoi[gold_role]]> prob[pred_role_type]:
                        human = human_readable[role_idx]
                        print('# ----------------------------------------------- #')
                        print('token: ',human['tokens'])
                        print('pred: ',(human['trigger_type'],human['trigger_token'][0],human['entity_type'], human['entity_token'][0]))
                        print('amr path: ',human['amr_path'])
                        print('gold role: ',gold_role,'pred role: ',human['role_type'],'pred score: ',pred_score ,'score pred: ',prob[pred_role_type], 'score gold: ', prob[self.role_type_stoi[gold_role]])
                    # ----------------------------------------------------------------- #

                    total_gold_score+=prob[self.role_type_stoi[gold_role]]*2.0 - 1
                    total_pred_score+=prob[pred_role_type]*2.0 - 1
                    # each_type_result[gold_role]['gold_score']+=prob[self.role_type_stoi[gold_role]]
                    # each_type_result[gold_role]['pred_score']+=prob[pred_role_type]
                    role_idx+=1
                    total_pred_role_num+=1
                    # each_type_result[gold_role]['total']+=1
                    # if prob[self.role_type_stoi[gold_role]] >= 0.5:
                    #     accu_result['gold_corr_num'] += 1


            # print('graph after: ',copy_graph.roles)
            # pred_role_num = role_idx
            # print(each_type_result)
        return total_pred_role_num, total_gold_score, total_pred_score, accu_result

    def feedback(self,batch, use_heuristic=False, margin=0.1,threshold=None):
        def heuristic(p_s, p_e, margin, label):
            # c_s = abs(p_s-0.5) * 2
            # c_e = p_e
            # if p_s > 0.5:
            #     if p_e > 0.5:
            #         f = 2 * p_s - 1
            #     else:
            #         f = p_s + p_s - 1
            # elif p_s <=0.5:
            #     if c_s - c_e >=margin:
            #         f = 2*p_s-1
            #     elif c_e - c_s >=margin:
            #         f = 0
            #     elif abs(c_e-c_s) < margin:
            #         f = p_s+p_e - 1
            #     else:
            #         raise NotImplementedError
            # print(2*p_s-1,p_s,p_e,f,label)
            f = (p_s*2.0 -1.0)*p_e
            print('prob',p_s, p_e, f, self.role_type_itos[label])
            return f
                    
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size = bert_outputs.shape[0]
        ie2amr_edge_map_list = []
        for i in range(batch_size):
            ie2amr_node_map, ie2amr_edge_map = get_ie2amr_role(batch.amr[i],batch.align[i],batch.exist[i],batch.amr_bidir_graph[i],batch.amr_edge2type[i],batch.graphs[i],self.vocabs,batch.tokens[i])
            ie2amr_edge_map_list.append(ie2amr_edge_map)
        output = self.scores(bert_outputs,batch,ie2amr_edge_map_list=ie2amr_edge_map_list)
        if not output['probs'] is None:
            probs = output['probs'].detach().cpu().tolist()
        else:
            probs = output['probs']

        role_feedbacks = []
        trigger_feedbacks = []
        count_role = 0
        count_trigger = 0
        entity_num = max(output['entity_num'],1)
        trigger_num = max(output['trigger_num'],1)
        # print('entity num', entity_num)
        # print('trigger num',trigger_num)
        num_total_role = 0
        for b in range(batch_size):
            graph = batch.graphs[b]
            role_dict = { (role[0],role[1]):(role[2],role_score) for role, role_score in zip(graph.roles, graph.role_scores)}
            for t in range(graph.trigger_num):
                trigger_feedback = 0.0
                count_trigger += 1
                count_trigger_role = 0
                for e in range(graph.entity_num):
                    ie_pred = role_dict[(t,e)]
                    role_feedback = probs[num_total_role][ie_pred[0]] * 2.0 -1.0
                    if use_heuristic:
                        role_feedback = heuristic(probs[num_total_role][ie_pred[0]],ie_pred[1],margin,ie_pred[0])
                    if threshold:
                        prob = probs[num_total_role][ie_pred[0]]
                        if prob >= threshold or prob <= 1.0 - threshold:
                            role_feedbacks.append(role_feedback)
                            count_role += 1
                            count_trigger_role += 1
                        else:
                            role_feedbacks.append(0.0)
                    else:
                        role_feedbacks.append(role_feedback)
                        count_role += 1
                        count_trigger_role += 1
                    # ----------------------- debug ------------------------- #
                    # human = human_readable_list[count_role]
                    # print('# ----------------------------------------------- #')
                    # print('token: ',human['tokens'])
                    # print('pred: ',(human['trigger_type'],human['trigger_token'][0],human['entity_type'], human['entity_token'][0]))
                    # print('amr path: ',human['amr_path'])
                    # print('pred role: ',human['role_type'],'pred score: ',ie_pred[1] ,'score score: ',role_feedback)
                    # ----------------------------------------------------------------- #
                    num_total_role += 1
                    trigger_feedback+=role_feedbacks[-1]
                trigger_feedback/=max(count_trigger_role,1)
                # if use_heuristic:
                #     trigger_feedback = heuristic(trigger_feedback,graph.trigger_scores[t],margin,'O')
                trigger_feedbacks.append(trigger_feedback)
                role_feedbacks.extend([0]*(entity_num-graph.entity_num))
            trigger_feedbacks.extend([0]*(trigger_num-graph.trigger_num))
            role_feedbacks.extend([0]*entity_num*(trigger_num-graph.trigger_num))
        threshold_ratio = count_role/max(1,num_total_role)
        # print('total num of role in score model: ',count_role)
        return role_feedbacks, count_role, trigger_feedbacks, count_trigger, threshold_ratio
    
    def label_data(self,batch, use_heuristic=False, margin=0.1):
        # the margin may be tricky here
        # TODO: fix the logic for maring here 
        def heuristic(p_s, p_e, margin, label):
            c_s = abs(p_s-0.5) * 2
            c_e = p_e
            if p_s > 0.5:
                if p_e > 0.5:
                    f = 2 * p_s - 1
                else:
                    f = p_s + p_s - 1
            elif p_s <=0.5:
                if c_s - c_e >=margin:
                    f = 2*p_s-1
                elif c_e - c_s >=margin:
                    f = 0
                elif abs(c_e-c_s) < margin:
                    f = p_s+p_e - 1
                else:
                    raise NotImplementedError
            # print(2*p_s-1,p_s,p_e,f,label)
            return f
                    
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size = bert_outputs.shape[0]
        ie2amr_edge_map_list = []
        for i in range(batch_size):
            ie2amr_node_map, ie2amr_edge_map = get_ie2amr_role(batch.amr[i],batch.align[i],batch.exist[i],batch.amr_bidir_graph[i],batch.amr_edge2type[i],batch.graphs[i],self.vocabs,batch.tokens[i])
            ie2amr_edge_map_list.append(ie2amr_edge_map)
        output = self.scores(bert_outputs,batch,ie2amr_edge_map_list=ie2amr_edge_map_list)
        if not output['probs'] is None:
            probs = output['probs'].detach().cpu().tolist()
        else:
            probs = output['probs']
        human_readable_list = output['human_readable']

        role_feedbacks = []
        trigger_feedbacks = []
        count_role = 0
        count_trigger = 0
        entity_num = max(output['entity_num'],1)
        trigger_num = max(output['trigger_num'],1)
        # print('entity num', entity_num)
        # print('trigger num',trigger_num)
        output = []
        for b in range(batch_size):
            graph = batch.graphs[b]
            role_dict = { (role[0],role[1]):(role[2],role_score) for role, role_score in zip(graph.roles, graph.role_scores)}
            one_sent_score = 0.0
            one_sent_num = 0
            inst = {"sent_id":batch.sent_ids[b],'tokens':batch.tokens[b],'pred':[]}
            for t in range(graph.trigger_num):
                trigger_feedback = 0.0
                count_trigger += 1
                for e in range(graph.entity_num):
                    ie_pred = role_dict[(t,e)]
                    role_feedback = probs[count_role][ie_pred[0]] * 2.0 -1.0
                    if use_heuristic:
                        role_feedback = heuristic(probs[count_role][ie_pred[0]],ie_pred[1],margin,ie_pred[0])
                    role_feedbacks.append(role_feedback)
                    human = human_readable_list[count_role]

                    inst['pred'].append({'trigger':human['trigger_token'],'trigger_type':human['trigger_type'],'entity':human['entity_token'],'entity_type':human['entity_type'],'amr_path':human['amr_path'],'pred_role':human['role_type'],'ie_score':ie_pred[1],'trigger_score':graph.trigger_scores[t],'score':role_feedback})
                    # ----------------------- debug ------------------------- #
                    # if (role_feedback + ie_pred[1])/2 > 0.95:
                    #     print('# ----------------------------------------------- #')
                    #     print("sent id: ",batch.sent_ids[b])
                    #     assert batch.tokens[b] == human['tokens']
                    #     print('token: ',human['tokens'])
                    #     print('pred: ',(human['trigger_type'],human['trigger_token'],human['entity_type'], human['entity_token'][0]))
                    #     print('amr path: ',human['amr_path'])
                    #     print('pred role: ',human['role_type'],'ie pred score: ',ie_pred[1],'trigger score':graph.trigger_scores[t],'score model score: ',role_feedback)
                    # ----------------------------------------------------------------- #
                    one_sent_num += 1
                    one_sent_score += (role_feedback + ie_pred[1]+graph.trigger_scores[t])/3 if role_feedback>0 else (abs(role_feedback) - ie_pred[1]+graph.trigger_scores[t])/2

                    count_role += 1
                    trigger_feedback+=role_feedback
                trigger_feedback/=max(graph.entity_num,1)
                if use_heuristic:
                    trigger_feedback = heuristic(trigger_feedback,graph.trigger_scores[t],margin)
                trigger_feedbacks.append(trigger_feedback)
                role_feedbacks.extend([0]*(entity_num-graph.entity_num))
            trigger_feedbacks.extend([0]*(trigger_num-graph.trigger_num))
            role_feedbacks.extend([0]*entity_num*(trigger_num-graph.trigger_num))
            # print('#------------- avg score ',one_sent_score/max(one_sent_num,1),'------------#')
            inst['avg_score']=one_sent_score/max(one_sent_num,1)
            # if one_sent_score/max(one_sent_num,1) > 0.95:
            #     print('# ------------------------------------------------------- #')
            #     print(inst)
            output.append(inst)
        return output
