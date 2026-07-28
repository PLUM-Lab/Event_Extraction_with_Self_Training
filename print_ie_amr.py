import os
import json
import time
import copy
from argparse import ArgumentParser
import random
import numpy as np
import torch

        
parser = ArgumentParser()
parser.add_argument('--input_path', default='')
parser.add_argument('--amr_path', default='')
args = parser.parse_args()





def get_role_list(entities, events, id_map):
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(events))]
    role_list = []
    for i, event in enumerate(events):
        for arg in event['arguments']:
            entity_idx = entity_idxs[id_map.get(
                arg['entity_id'], arg['entity_id'])]
            if visited[i][entity_idx] == 0:
                role_list.append((i, entity_idx, arg['role']))
                visited[i][entity_idx] = 1
    role_list.sort(key=lambda x: (x[0], x[1]))
    return role_list

def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    for entity in entities:
        start, end = entity['start'], entity['end']
        for i in range(start, end):
            if tokens[i]:
                id_map[entity['id']] = tokens[i]
                continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity['id']
    return entities_, id_map

def process_amr(raw_amr):
    amr_split_list = raw_amr.split('\n')
    node_to_idx, node_to_offset, node_to_offset_whole = {}, {}, {}
    node_num = 0
    # first to fill in the node list
    for line in amr_split_list:
        if line.startswith('# ::node'):
            node_split = line.split('\t') # ['# ::node','2','multiple','1-2']
            if len(node_split) != 4:
                # check if the alignment text spans exist
                continue
            else:
                align_span = node_split[3].split('-') #['1','2']
                if not align_span[0].isdigit():
                    continue
                head_word_idx = int(align_span[1]) - 1 # 1
                try:
                    start = int(align_span[0]) # 1
                except:
                    # print(amr_list[i])
                    raise ValueError
                end = int(align_span[1]) #2
                if (start, end) not in list(node_to_offset_whole.values()):
                    node_to_offset.update({node_split[1]: head_word_idx}) # {'node_id 2': end_idx-1}
                    node_to_offset_whole.update({node_split[1]: (start, end)}) # {'node_id 2': (start_idx, end_idx)}
                    node_to_idx.update({node_split[1]: node_num}) # {'node_id 2': node_index}
                    node_num += 1
        else:
            continue

    # node_idx_list.append(node_to_idx)
    # change node_id2offset to idx2offset
    node_idx_to_offset = {}
    for key in node_to_idx.keys():
        node_idx_to_offset.update({node_to_idx[key]: node_to_offset[key]}) # {node_index: end_idx-1}

    node_idx_to_offset_whole = {}
    for key in node_to_idx.keys():
        node_idx_to_offset_whole.update({node_to_idx[key]: node_to_offset_whole[key]}) # {node_index: (start_idx, end_idx)}

    edge_type_dict = {}

    # add amr prior order
    for line in amr_split_list:
        if line.startswith('# ::root'):
            root_split = line.split('\t')
            root = root_split[1]

    prior_dict = {root:[]}

    start_list = []
    end_list = []

    for line in amr_split_list:
        if line.startswith('# ::edge'):
            edge_split = line.split('\t') # ['# ::edge','multiple','op1','1000000','2','3']
            amr_edge_type = edge_split[2] #'op1'
            edge_start = edge_split[4] # '2' this is node id
            edge_end = edge_split[5] # '3' this is node id
            # check if the start and end nodes exist
            if (edge_start in node_to_idx) and (edge_end in node_to_idx):
                # check if the edge type is "ARGx-of", if so, reverse the direction of the edge
                if amr_edge_type.endswith("-of"):
                    edge_start, edge_end = edge_end, edge_start
                    amr_edge_type = amr_edge_type[0:-3]
                start_idx = node_to_idx[edge_start] # node_index 
                end_idx = node_to_idx[edge_end]
                edge_type_dict.update({(start_idx, end_idx): amr_edge_type}) #{(start_node_idx,end_node_idx): edge type}
            else:
                continue
            if edge_end != root and (not ((edge_start in end_list) and (edge_end in start_list))):
                start_list.append(edge_start)
                end_list.append(edge_end)
            if edge_start not in prior_dict:
                prior_dict.update({edge_start:[edge_end]})
            else:
                prior_dict[edge_start].append(edge_end)
        else:
            continue
    final_order_list = []
    # output orders
    candidate_nodes = node_to_idx.copy()
    while len(candidate_nodes) != 0:
        current_level_nodes = []
        for key in candidate_nodes:
            if key not in end_list:
                final_order_list.append(candidate_nodes[key])
                current_level_nodes.append(key)
        # Remove current level nodes from the dictionary
        for node in current_level_nodes:
            candidate_nodes.pop(node)

        # deleting from start lists the current level nodes
        for node in current_level_nodes:
            indices_list = [i for i, x in enumerate(start_list) if x == node]
            start_list = [x for x in start_list if x != node]
            new_end_list = []
            for i in range(len(end_list)):
                if i not in indices_list:
                    new_end_list.append(end_list[i])
            end_list = new_end_list
    return {'node_to_idx':node_to_idx, 'node_idx_to_offset':node_idx_to_offset,'node_idx_to_offset_whole':node_idx_to_offset_whole, 'edge_type_dict':edge_type_dict,'decode_order':final_order_list}



data = []

def main():
    with open(args.input_path, 'r', encoding='utf-8') as r:
        amr_list = torch.load(args.amr_path)
        for i, line in enumerate(r):
            inst = json.loads(line)
            inst_len = len(inst['pieces'])
            is_title = inst['sent_id'].endswith('-3') \
                and inst['tokens'][-1] != '.' \
                and len(inst['entity_mentions']) == 0
            if not sum(inst['token_lens']) == inst_len:
                continue
            if not '# ::root' in amr_list[i]:
                continue
            amr_info = process_amr(amr_list[i])
            inst['amr_info']=(amr_list[i])
            data.append(inst)
            
    for inst in data:
        tokens = inst['tokens']
        pieces = inst['pieces']
        entities = inst['entity_mentions']
        entities, entity_id_map = remove_overlap_entities(entities)
        entities.sort(key=lambda x: x['start'])
        events = inst['event_mentions']
        events.sort(key=lambda x: x['trigger']['start'])
        role_list = get_role_list(entities, events,
                                        entity_id_map)
        amr = inst['amr_info']
        if tokens == ['Australian', 'commandos', ',', 'who', 'have', 'been', 'operating', 'deep', 'in', 'Iraq', ',', 'destroyed', 'a', 'command', 'and', 'control', 'post', 'and', 'killed', 'a', 'number', 'of', 'soldiers', ',', 'according', 'to', 'the', 'country', "'s", 'defense', 'chief', ',', 'Gen.', 'Peter', 'Cosgrove', '.']:
            print('-------------------------------------------')
            print('tokens: ',tokens)
            print('pieces: ',pieces)
            print('AMR :',amr)
            
            # print(amr['node_idx_to_offset_whole'])
            # print(amr['edge_type_dict'])
            for role in role_list:
                print(events[role[0]], entities[role[1]], role)
                print()
            
            # print(role_list)
            print()
        
# ['Australian', 'commandos', ',', 'who', 'have', 'been', 'operating', 'deep', 'in', 'Iraq', ',', 'destroyed', 'a', 'command', 'and', 'control', 'post', 'and', 'killed', 'a', 'number', 'of', 'soldiers', ',', 'according', 'to', 'the', 'country', "'s", 'defense', 'chief', ',', 'Gen.', 'Peter', 'Cosgrove', '.']
main()
    
        