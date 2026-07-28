import os
import json
import glob
import lxml.etree as et
from nltk import word_tokenize, sent_tokenize
from copy import deepcopy

import copy
import torch
from graph import Graph
from torch.utils.data import Dataset
from collections import Counter, namedtuple, defaultdict
import torch.nn as nn

# --------------------------- self train ----------------------------- #
batch_fields = [
    'sent_ids', 'tokens', 'piece_idxs', 'token_lens', 'attention_masks',
    'entity_label_idxs', 'trigger_label_idxs',
    'entity_type_idxs', 'event_type_idxs', 'mention_type_idxs',
    'relation_type_idxs', 'role_type_idxs',
    'graphs', 'token_nums', 'amr', 'align', 'exist',
    'amr_infos', 'ie2amr_edge_maps', 'amr_node_span_lists', 'gold_events'
]
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))

def pred2input(pred_graphs, aux_batch, vocabs, margin=0.9):
    max_entity_num = max([graph.entity_num for graph in pred_graphs] + [1])
    max_trigger_num = max([graph.trigger_num for graph in pred_graphs] + [1])
    batch_role_types = []
    batch_event_types = []
    graphs = []
    role_masks = []
    trigger_masks = []
    count_role = 0
    for graph in pred_graphs:
        graph = copy.deepcopy(graph)
        graphs.append(graph)
        entities = graph.entities
        triggers = graph.triggers
        entity_num = len(graph.entities)
        trigger_num = len(graph.triggers)
        role_labels = { (role[0],role[1]):role[2] for role in graph.roles}
        labels = [[vocabs['role_type']['O']] * len(entities) for _ in range(len(triggers))]
        role_dict = { (role[0],role[1]):(role[2],role_score) for role, role_score in zip(graph.roles, graph.role_scores)}
        event_types = []
        for tri_idx, tri in enumerate(triggers):
            event_types.append(tri[2])
            trigger_score = graph.trigger_scores[tri_idx]
            trigger_masks.append(1.0 if trigger_score>margin else 0.0)
            for ent_idx, ent in enumerate(entities):
                pred_score = role_dict[(tri_idx,ent_idx)][1]
                role_masks.append(1.0 if pred_score>margin else 0.0)
                # if (tri_idx,ent_idx) in role_labels:
                labels[tri_idx][ent_idx] = role_labels[(tri_idx,ent_idx)]
                # --------- debug code ---------------- 
                # role_type_itos = {i: s for s, i in vocabs['role_type'].items()} 
                # print('pred 2 input: ',role_type_itos[labels[tri_idx][ent_idx]])
                count_role+=1
            batch_role_types.extend(labels[tri_idx]+[-100] * (max_entity_num - entity_num))
            role_masks.extend([0]*(max_entity_num - entity_num))
        batch_role_types.extend([-100] * max_entity_num * (max_trigger_num - trigger_num))
        role_masks.extend([0] * max_entity_num * (max_trigger_num - trigger_num))
        trigger_masks.extend([0]*(max_trigger_num - trigger_num))
        batch_event_types.extend(event_types+[-100]*(max_trigger_num-trigger_num))
    batch_event_types = torch.cuda.LongTensor(batch_event_types)
    batch_role_types = torch.cuda.LongTensor(batch_role_types)
    
    return Batch(
        sent_ids=copy.deepcopy(aux_batch.sent_ids),
        tokens=copy.deepcopy(aux_batch.tokens),
        piece_idxs=copy.deepcopy(aux_batch.piece_idxs),
        token_lens=copy.deepcopy(aux_batch.token_lens),
        attention_masks=copy.deepcopy(aux_batch.attention_masks),
        token_nums=copy.deepcopy(aux_batch.token_nums),
        graphs=graphs,
        role_type_idxs=batch_role_types,
        event_type_idxs=batch_event_types,
        amr=copy.deepcopy(aux_batch.amr),
        align=copy.deepcopy(aux_batch.align),
        exist=copy.deepcopy(aux_batch.exist)
    ), role_masks, trigger_masks, count_role
    
def pred2amr(pred_graphs, aux_batch, vocabs, ie_vocabs):
    
    def BFS(startnode, endnode, amr_graph):
        if not startnode in amr_graph:
            return -1, None
        queue = []
        visited = set()
        previous = {}
        queue.append((startnode,0))
        visited.add(startnode)
        while len(queue)>0:
            # print(queue)
            node, dist = queue.pop(0)
            if node == endnode:
                return dist, previous
            for neighbour in amr_graph[node]:
                if not neighbour in visited:
                    assert not neighbour in previous
                    previous[neighbour] = node
                    queue.append((neighbour,dist+1))
                    visited.add(neighbour)
        return -1, None
    
    def find_amr_path(startnode, endnode, amr_graph, edge2type):
        dist, previous=BFS(startnode, endnode, amr_graph)
        if dist < 0:
            return None, None
        node_path = []
        rel_path = []
        # print('distance: ',dist)
        while not endnode == startnode:
            node_path.append(endnode)
            next_node = previous[endnode]
            if not (next_node, endnode) in edge2type:
                rel_path.append(edge2type[(endnode,next_node)])
            else:
                rel_path.append(edge2type[(next_node,endnode)])
            endnode = next_node
        node_path.append(startnode)
        node_path.reverse()
        rel_path.reverse()

        return node_path, rel_path

    def find_ie2amr_edge(pred_graph, amr_info, tokens, vocabs, ie_vocabs):
        # print('entity vocab in find: ',vocabs['entity_type'])
        max_entity_num = max([graph.entity_num for graph in pred_graphs] + [1])
        max_trigger_num = max([graph.trigger_num for graph in pred_graphs] + [1])
        # vocabs
        entity_type_stoi = vocabs['entity_type']
        event_type_stoi = vocabs['event_type']
        relation_type_stoi = vocabs['relation_type']
        role_type_stoi = vocabs['role_type']
        amr_edge_type_stoi = vocabs['amr_edge_type']
        _node_type_offset = len(event_type_stoi)

        #ie vocabs
        ie_entity_type_stoi = ie_vocabs['entity_type']
        ie_event_type_stoi = ie_vocabs['event_type']
        ie_relation_type_stoi = ie_vocabs['relation_type']
        ie_role_type_stoi = ie_vocabs['role_type']
        ie_entity_type_itos={}
        for _s, _i in ie_entity_type_stoi.items():
            ie_entity_type_itos[_i] = _s
        ie_event_type_itos={}
        for _s, _i in ie_event_type_stoi.items():
            ie_event_type_itos[_i] = _s
        # relation_type_itos={}
        # for _s, _i in relation_type_stoi.items():
        #     relation_type_itos[_i] = _s
        ie_role_type_itos={}
        for _s, _i in ie_role_type_stoi.items():
            ie_role_type_itos[_i] = _s
        # _edge_type_offset = len(role_type_stoi)
        # _node_type_offset = len(event_type_stoi)
        # convert index back to str label
        ie_roles = {(i, j): ie_role_type_itos[k] for (i, j, k) in pred_graph.roles} # ie role label
        # ie_triggers = {(i, j): ie_event_type_itos[k] for (i, j, k) in pred_graph.triggers} # ie trigger label
        # ie_entities = {(i, j): ie_entity_type_itos[k] for (i, j, k) in pred_graph.entities} # ie entity label


        amr_node_span_dic = amr_info['node_idx_to_offset_whole']
        amr_edge2type = amr_info['edge2type']
        align_list = amr_info['align_dict']
        exist_list = amr_info['exist_dict']
        bidir_amr_graph = amr_info['bidir_graph']

        amr_node_span_list = [amr_node_span_dic[i] for i in range(len(amr_node_span_dic))]
        amr_nodes_selected = [0 for _ in range(len(amr_node_span_list))]
        
        ie_node_span_list = pred_graph.triggers + pred_graph.entities

        # first align ie node with amr node and add missing node into amr
        # print('amrgraph before: ',bidir_amr_graph)
        ie2amr_node_map = {} # ie node idx : amr node idx
        node_added = 0
        # print(ie_node_span_list)
        for ie_node_idx, node_span in enumerate(ie_node_span_list):
            head_node_offset = node_span[1] - 1
            # print(head_node_offset)
            # print(align_list)
            amr_node_idx = align_list[head_node_offset]

            # if in the amr list 
            if exist_list[head_node_offset] == 1 and amr_nodes_selected[amr_node_idx] ==0:
                amr_nodes_selected[amr_node_idx] = 1
                ie2amr_node_map[ie_node_idx] = amr_node_idx
            # if can not find it add a new node in amr
            else:
                node_added += 1
                amr_node_span_list.append(node_span)
                amr_node_num = len(amr_node_span_list)
                amr_nodes_selected.append(1)
                if not amr_node_idx in bidir_amr_graph:
                    bidir_amr_graph[amr_node_idx] = [amr_node_num - 1]
                else:
                    bidir_amr_graph[amr_node_idx].append(amr_node_num - 1)
                assert not amr_node_num - 1 in bidir_amr_graph
                bidir_amr_graph[amr_node_num - 1]=[amr_node_idx]
                amr_edge2type[(amr_node_idx, amr_node_num - 1)] = 'amr2ie'
                ie2amr_node_map[ie_node_idx] = amr_node_num - 1
        
        
        # second align ie edge to amr edge
        ie2amr_edge_map = {}  # (start node idx , end node idx): [amr nodes on path], [amr relation on path]
        # print('amr edges: ', amr_graph.edges())
        # print('amr edge types : ', amr_graph.edata['type'].tolist())

        # print('amr edge to type: ',amr_edge2type)
        # print('node map: ',ie2amr_node_map)
        # ie_edges = ie_graph.edata.tolist()
        # print('ie edges: ',ie_edge2type)
        # print('amrgraph middle: ',bidir_amr_graph)
        for tri_idx, trigger in enumerate(pred_graph.triggers):
            for ent_idx, entity in enumerate(pred_graph.entities):

                # for debug: get ie subgraph
                # amr_node_idx = ie2amr_node_map[ie_node_idx]
                if (tri_idx,ent_idx) in ie_roles:
                    ie_egde_type = ie_roles[(tri_idx,ent_idx)] # str label
                else:
                    ie_egde_type = 'O'

                startnode =  ie2amr_node_map[tri_idx]
                endnode =  ie2amr_node_map[len(pred_graph.triggers)+ent_idx]
                trigger_type = ie_event_type_itos[trigger[2]]
                entity_type = ie_entity_type_itos[entity[2]]
                # print('start, end node: ',startnode,endnode)
                # print('bidir amr graph',bidir_amr_graph)
                temp_startnode = copy.deepcopy(startnode)
                temp_endnode = copy.deepcopy(endnode)
                temp_bidir_amr_graph = copy.deepcopy(bidir_amr_graph)
                temp_amr_edge2type = copy.deepcopy(amr_edge2type)
                amr_node_path, amr_rel_path_label = find_amr_path(temp_startnode, temp_endnode, temp_bidir_amr_graph, temp_amr_edge2type)

                if amr_node_path is None:
                    # print('start end node after: ',startnode, endnode)
                    amr_node_path = [startnode, endnode]
                    amr_rel_path_label = ['amr2ie']
                    # if not startnode in bidir_amr_graph: 
                    #     bidir_amr_graph[startnode]=[endnode]  
                    # else: 
                    #     bidir_amr_graph[startnode].append(endnode)
                    # if not endnode in bidir_amr_graph:
                    #     bidir_amr_graph[endnode]=[startnode] 
                    # else:
                    #     bidir_amr_graph[endnode].append(startnode)
                    # amr_edge2type[(startnode, endnode)] = 'amr2ie'

                # human readable
                human_readable = {'tokens':tokens}
                human_readable['trigger_type'] = trigger_type
                human_readable['entity_type'] = entity_type
                human_readable['role_type'] = ie_egde_type
                human_readable['trigger_token'] = tokens[trigger[0]:trigger[1]]
                human_readable['entity_token'] = tokens[entity[0]:entity[1]]

                amr_tokens = []
                for amr_node in amr_node_path:
                    amr_token = tokens[amr_node_span_list[amr_node][0]:amr_node_span_list[amr_node][1]]
                    amr_tokens.append(amr_token)
                amr_edges = []
                human_readable['amr_path'] = []
                for amr_token, amr_rel in zip(amr_tokens, amr_rel_path_label):
                    human_readable['amr_path'].append(' '.join(amr_token)) if isinstance(amr_token, list) else human_readable['amr_path'].append(_amr_token)
                    if not amr_rel in amr_edge_type_stoi:
                        human_readable['amr_path'].append('amr2ie')
                    else:
                        human_readable['amr_path'].append(amr_rel)
                human_readable['amr_path'].append(' '.join(amr_tokens[-1])) if isinstance(amr_tokens[-1], list) else human_readable['amr_path'].append(amr_tokens[-1])

                # print(human_readable)
                # now convert lables to index
                ie_egde_type = [role_type_stoi[ie_egde_type]]
                trigger_type = [event_type_stoi[trigger_type]]
                entity_type = [entity_type_stoi[entity_type]+_node_type_offset]
                
                amr_rel_path = []
                for _rel in  amr_rel_path_label:
                    if not _rel in amr_edge_type_stoi:
                        amr_rel_path.append(amr_edge_type_stoi['amr2ie'])
                        # print('miss one relation: ', _rel)
                    else:
                        amr_rel_path.append(amr_edge_type_stoi[_rel])
                
                # amr_rel_path = [amr_edge_type_stoi[_rel] for _rel in  amr_rel_path]
                amr_node_path = torch.cuda.LongTensor(amr_node_path)
                amr_rel_path = torch.cuda.LongTensor(amr_rel_path)
                ie_egde_type = torch.cuda.LongTensor(ie_egde_type)
                trigger_type = torch.cuda.LongTensor(trigger_type)
                entity_type = torch.cuda.LongTensor(entity_type)
                ie2amr_edge_map[(tri_idx,ent_idx)] = {'amr_node':amr_node_path,'amr_rel':amr_rel_path,'ie_edge_type':ie_egde_type,'ie_node_type':[trigger_type, entity_type],'human_readable':human_readable}
                # print(ie2amr_edge_map)
            for ent_idx in range(max_entity_num-len(graph.entities)):
                ie2amr_edge_map[(tri_idx,10000+ent_idx)] = {}
        for tri_idx in range(max_trigger_num-len(graph.triggers)):
            for ent_idx in range(max_entity_num):
                ie2amr_edge_map[((tri_idx+1)*10000,ent_idx)] = {}
        # print('node added: ',node_added)
        return amr_node_span_list, ie2amr_edge_map


    ie2amr_edge_maps = []
    amr_node_span_lists = []
    for graph, tokens, amr_info in zip(pred_graphs, aux_batch.tokens, aux_batch.amr_infos):
        amr_node_span_list, ie2amr_edge_map = find_ie2amr_edge(copy.deepcopy(graph), copy.deepcopy(amr_info), tokens, vocabs, ie_vocabs)
        ie2amr_edge_maps.append(ie2amr_edge_map)
        amr_node_span_lists.append(amr_node_span_list)

    return Batch(
        sent_ids=copy.deepcopy(aux_batch.sent_ids),
        tokens=copy.deepcopy(aux_batch.tokens),
        piece_idxs=copy.deepcopy(aux_batch.piece_idxs),
        token_lens=copy.deepcopy(aux_batch.token_lens),
        attention_masks=copy.deepcopy(aux_batch.attention_masks),
        token_nums=copy.deepcopy(aux_batch.token_nums),
        graphs=copy.deepcopy(pred_graphs),
        ie2amr_edge_maps=ie2amr_edge_maps,
        amr_node_span_lists=amr_node_span_lists,
        gold_events=copy.deepcopy(aux_batch.gold_events)
    )

# ----------------------------------------------------------------------------------------- #

def generate_vocabs(datasets, coref=False,
                    relation_directional=False,
                    symmetric_relations=None):
    """Generate vocabularies from a list of data sets
    :param datasets (list): A list of data sets
    :return (dict): A dictionary of vocabs
    """
    entity_type_set = set()
    event_type_set = set()
    relation_type_set = set()
    role_type_set = set()
    for dataset in datasets:
        entity_type_set.update(dataset.entity_type_set)
        event_type_set.update(dataset.event_type_set)
        relation_type_set.update(dataset.relation_type_set)
        role_type_set.update(dataset.role_type_set)

    # add inverse relation types for non-symmetric relations
    if relation_directional:
        if symmetric_relations is None:
            symmetric_relations = []
        relation_type_set_ = set()
        for relation_type in relation_type_set:
            relation_type_set_.add(relation_type)
            if relation_directional and relation_type not in symmetric_relations:
                relation_type_set_.add(relation_type + '_inv')

    # entity and trigger labels
    prefix = ['B', 'I']
    entity_label_stoi = {'O': 0}
    trigger_label_stoi = {'O': 0}
    for t in entity_type_set:
        for p in prefix:
            entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)
    for t in event_type_set:
        for p in prefix:
            trigger_label_stoi['{}-{}'.format(p, t)] = len(trigger_label_stoi)

    entity_type_stoi = {k: i for i, k in enumerate(entity_type_set, 1)}
    entity_type_stoi['O'] = 0

    event_type_stoi = {k: i for i, k in enumerate(event_type_set, 1)}
    event_type_stoi['O'] = 0

    relation_type_stoi = {k: i for i, k in enumerate(relation_type_set, 1)}
    relation_type_stoi['O'] = 0
    if coref:
        relation_type_stoi['COREF'] = len(relation_type_stoi)

    role_type_stoi = {k: i for i, k in enumerate(role_type_set, 1)}
    role_type_stoi['O'] = 0

    mention_type_stoi = {'NAM': 0, 'NOM': 1, 'PRO': 2, 'UNK': 3}

    return {
        'entity_type': entity_type_stoi,
        'event_type': event_type_stoi,
        'relation_type': relation_type_stoi,
        'role_type': role_type_stoi,
        'mention_type': mention_type_stoi,
        'entity_label': entity_label_stoi,
        'trigger_label': trigger_label_stoi,
    }


def load_valid_patterns(path, vocabs):
    event_type_vocab = vocabs['event_type']
    entity_type_vocab = vocabs['entity_type']
    relation_type_vocab = vocabs['relation_type']
    role_type_vocab = vocabs['role_type']

    # valid event-role
    valid_event_role = set()
    event_role = json.load(
        open(os.path.join(path, 'event_role.json'), 'r', encoding='utf-8'))
    for event, roles in event_role.items():
        if event not in event_type_vocab:
            print('event {} is not in the valid pattern')
            continue
        event_type_idx = event_type_vocab[event]
        for role in roles:
            if role not in role_type_vocab:
                print('role {} is not in the valid pattern')
                continue
            role_type_idx = role_type_vocab[role]
            valid_event_role.add(event_type_idx * 100 + role_type_idx)

    # valid relation-entity
    valid_relation_entity = set()
    relation_entity = json.load(
        open(os.path.join(path, 'relation_entity.json'), 'r', encoding='utf-8'))
    for relation, entities in relation_entity.items():
        if not relation in relation_type_vocab:
            print('relation {} is not in the valid pattern')
            continue
        relation_type_idx = relation_type_vocab[relation]
        for entity in entities:
            if not entity in entity_type_vocab:
                print('relation {} is not in the valid pattern')
                continue
            entity_type_idx = entity_type_vocab[entity]
            valid_relation_entity.add(
                relation_type_idx * 100 + entity_type_idx)

    # valid role-entity
    valid_role_entity = set()
    role_entity = json.load(
        open(os.path.join(path, 'role_entity.json'), 'r', encoding='utf-8'))
    for role, entities in role_entity.items():
        if role not in role_type_vocab:
            print('relation {} is not in the valid pattern')
            continue
        role_type_idx = role_type_vocab[role]
        for entity in entities:
            entity_type_idx = entity_type_vocab[entity]
            valid_role_entity.add(role_type_idx * 100 + entity_type_idx)

    return {
        'event_role': valid_event_role,
        'relation_entity': valid_relation_entity,
        'role_entity': valid_role_entity
    }


def read_ltf(path):
    root = et.parse(path, et.XMLParser(
        dtd_validation=False, encoding='utf-8')).getroot()
    doc_id = root.find('DOC').get('id')
    doc_tokens = []
    for seg in root.find('DOC').find('TEXT').findall('SEG'):
        seg_id = seg.get('id')
        seg_tokens = []
        seg_start = int(seg.get('start_char'))
        seg_text = seg.find('ORIGINAL_TEXT').text
        for token in seg.findall('TOKEN'):
            token_text = token.text
            start_char = int(token.get('start_char'))
            end_char = int(token.get('end_char'))
            assert seg_text[start_char - seg_start:
                            end_char - seg_start + 1
                            ] == token_text, 'token offset error'
            seg_tokens.append((token_text, start_char, end_char))
        doc_tokens.append((seg_id, seg_tokens))

    return doc_tokens, doc_id


def read_txt(path, language='english'):
    doc_id = os.path.basename(path)
    data = open(path, 'r', encoding='utf-8').read()
    data = [s.strip() for s in data.split('\n') if s.strip()]
    sents = [l for ls in [sent_tokenize(line, language=language) for line in data]
             for l in ls]
    doc_tokens = []
    offset = 0
    for sent_idx, sent in enumerate(sents):
        sent_id = '{}-{}'.format(doc_id, sent_idx)
        tokens = word_tokenize(sent)
        tokens = [(token, offset + i, offset + i + 1)
                  for i, token in enumerate(tokens)]
        offset += len(tokens)
        doc_tokens.append((sent_id, tokens))
    return doc_tokens, doc_id


def read_json(path):
    with open(path, 'r', encoding='utf-8') as r:
        data = [json.loads(line) for line in r]
    doc_id = data[0]['doc_id']
    offset = 0
    doc_tokens = []

    for inst in data:
        tokens = inst['tokens']
        tokens = [(token, offset + i, offset + i + 1)
                  for i, token in enumerate(tokens)]
        offset += len(tokens)
        doc_tokens.append((inst['sent_id'], tokens))
    return doc_tokens, doc_id


def read_json_single(path):
    with open(path, 'r', encoding='utf-8') as r:
        data = [json.loads(line) for line in r]
    doc_id = os.path.basename(path)
    doc_tokens = []
    for inst in data:
        tokens = inst['tokens']
        tokens = [(token, i, i + 1) for i, token in enumerate(tokens)]
        doc_tokens.append((inst['sent_id'], tokens))
    return doc_tokens, doc_id


def save_result(output_file, gold_graphs, pred_graphs, sent_ids, tokens=None):
    with open(output_file, 'w', encoding='utf-8') as w:
        for i, (gold_graph, pred_graph, sent_id) in enumerate(
                zip(gold_graphs, pred_graphs, sent_ids)):
            output = {'sent_id': sent_id,
                      'gold': gold_graph.to_dict(),
                      'pred': pred_graph.to_dict()}
            if tokens:
                output['tokens'] = tokens[i]
            w.write(json.dumps(output) + '\n')


def mention_to_tab(start, end, entity_type, mention_type, mention_id, tokens, token_ids, score=1):
    tokens = tokens[start:end]
    token_ids = token_ids[start:end]
    span = '{}:{}-{}'.format(token_ids[0].split(':')[0],
                             token_ids[0].split(':')[1].split('-')[0],
                             token_ids[1].split(':')[1].split('-')[1])
    mention_text = tokens[0]
    previous_end = int(token_ids[0].split(':')[1].split('-')[1])
    for token, token_id in zip(tokens[1:], token_ids[1:]):
        start, end = token_id.split(':')[1].split('-')
        start, end = int(start), int(end)
        mention_text += ' ' * (start - previous_end) + token
        previous_end = end
    return '\t'.join([
        'json2tab',
        mention_id,
        mention_text,
        span,
        'NIL',
        entity_type,
        mention_type,
        str(score)
    ])


def json_to_mention_results(input_dir, output_dir, file_name,
                            bio_separator=' '):
    mention_type_list = ['nam', 'nom', 'pro', 'nam+nom+pro']
    file_type_list = ['bio', 'tab']
    writers = {}
    for mention_type in mention_type_list:
        for file_type in file_type_list:
            output_file = os.path.join(output_dir, '{}.{}.{}'.format(file_name,
                                                                     mention_type,
                                                                     file_type))
            writers['{}_{}'.format(mention_type, file_type)
                    ] = open(output_file, 'w')

    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    for f in json_files:
        with open(f, 'r', encoding='utf-8') as r:
            for line in r:
                result = json.loads(line)
                doc_id = result['doc_id']
                tokens = result['tokens']
                token_ids = result['token_ids']
                bio_tokens = [[t, tid, 'O']
                              for t, tid in zip(tokens, token_ids)]
                # separate bio output
                for mention_type in ['NAM', 'NOM', 'PRO']:
                    tokens_tmp = deepcopy(bio_tokens)
                    for start, end, enttype, mentype in result['graph']['entities']:
                        if mention_type == mentype:
                            tokens_tmp[start] = 'B-{}'.format(enttype)
                            for token_idx in range(start + 1, end):
                                tokens_tmp[token_idx] = 'I-{}'.format(
                                    enttype)
                    writer = writers['{}_bio'.format(mention_type.lower())]
                    for token in tokens_tmp:
                        writer.write(bio_separator.join(token) + '\n')
                    writer.write('\n')
                # combined bio output
                tokens_tmp = deepcopy(bio_tokens)
                for start, end, enttype, _ in result['graph']['entities']:
                    tokens_tmp[start] = 'B-{}'.format(enttype)
                    for token_idx in range(start + 1, end):
                        tokens_tmp[token_idx] = 'I-{}'.format(enttype)
                writer = writers['nam+nom+pro_bio']
                for token in tokens_tmp:
                    writer.write(bio_separator.join(token) + '\n')
                writer.write('\n')
                # separate tab output
                for mention_type in ['NAM', 'NOM', 'PRO']:
                    writer = writers['{}_tab'.format(mention_type.lower())]
                    mention_count = 0
                    for start, end, enttype, mentype in result['graph']['entities']:
                        if mention_type == mentype:
                            mention_id = '{}-{}'.format(doc_id, mention_count)
                            tab_line = mention_to_tab(
                                start, end, enttype, mentype, mention_id, tokens, token_ids)
                            writer.write(tab_line + '\n')
                # combined tab output
                writer = writers['nam+nom+pro_tab']
                mention_count = 0
                for start, end, enttype, mentype in result['graph']['entities']:
                    mention_id = '{}-{}'.format(doc_id, mention_count)
                    tab_line = mention_to_tab(
                        start, end, enttype, mentype, mention_id, tokens, token_ids)
                    writer.write(tab_line + '\n')
    for w in writers:
        w.close()


def normalize_score(scores):
    scores_tensor = torch.tensor(scores)
    scores_tensor = nn.functional.softmax(scores_tensor,dim=-1)
    scores = scores_tensor.tolist()
    return scores


def best_score_by_task(log_file, task, max_epoch=1000):
    with open(log_file, 'r', encoding='utf-8') as r:
        config = r.readline()

        best_scores = []
        best_dev_score = 0
        for line in r:
            record = json.loads(line)
            dev = record['dev']
            test = record['test']
            epoch = record['epoch']
            if epoch > max_epoch:
                break
            if dev[task]['f'] > best_dev_score:
                best_dev_score = dev[task]['f']
                best_scores = [dev, test, epoch]

        print('Epoch: {}'.format(best_scores[-1]))
        tasks = ['entity', 'mention', 'relation', 'trigger_id', 'trigger',
                 'role_id', 'role']
        for t in tasks:
            print('{}: dev: {:.2f}, test: {:.2f}'.format(t,
                                                         best_scores[0][t][
                                                             'f'] * 100.0,
                                                         best_scores[1][t][
                                                             'f'] * 100.0))
