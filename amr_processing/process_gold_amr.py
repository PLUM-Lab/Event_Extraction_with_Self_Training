import os
import sys
import json
from tqdm import tqdm

"""
This code is to process gold amr 3.0 dataset
"""


# read from all amr files and normalize files
def merge_dir(dir, outfile):

    # collect amrs
    amrs = []
    for filename in sorted(os.listdir(dir)):
        if not filename.startswith("training"):
            continue
        with open(os.path.join(dir, filename), encoding='utf-8') as f:
            print(filename)
            for i,line in enumerate(f):
                if i in [0, 1]:
                    continue
                # if line.startswith('# ::align'):
                #     continue
                amrs.append(line)
            amrs.append('\n')

    # normalization
    amrs = ''.join(amrs)
    amrs = amrs.replace('\r','')
    amrs = amrs.replace('\n\n\n','\n\n')
    amrs = amrs.replace('\u0092',"'")
    amrs = amrs.replace('\u0085'," ")

    # write data
    with open(outfile,'w+', encoding='utf-8') as f:
        f.write(amrs)
        print(amrs.count('# ::snt'))

# get fields:  sent_id, tokens, nodes, edges
def process_amr(raw_amr):
    amr_split_list = raw_amr.split('\n')
    node_to_idx, node_to_offset, node_to_offset_whole = {}, {}, {}
    node_num = 0
    nodes = {}
    
    for line in amr_split_list:
        if line.startswith('# ::id'):
            sent_id = line.split(' ')[2]
        if line.startswith('# ::tok'):
            tokens = line.split(' ')[2:]
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
                    # node id to start token index end token idx
                    nodes[node_split[1]] = (start, end)
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
                
                # if amr_edge_type.startswith("ARG") and amr_edge_type.endswith("-of"):
                #     edge_start, edge_end = edge_end, edge_start
                #     amr_edge_type = amr_edge_type[0:4]
                # self.amr_edge_type_set.add(amr_edge_type)

                start_idx = node_to_idx[edge_start] # node_index 
                end_idx = node_to_idx[edge_end]
                # start node idx, end node idx : edge type
                edge_type_dict.update({ edge_start+'\t'+edge_end: amr_edge_type})
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
    return {'sent_id': sent_id, 'tokens': tokens, 'nodes': nodes, 'edges': edge_type_dict,'decode_order':final_order_list}

if __name__ == '__main__':
    input_path = 'dataset/aligned_amr3.0.amr-ie/training_all.txt.raw'
    output_path = 'dataset/aligned_amr3.0.amr-ie/training_all.txt'
    # merge_dir(input_dir, output_path)
    input_amr = open(input_path).read().strip().split('\n\n')
    inst_list = []
    for raw_amr in tqdm(input_amr):
        inst = process_amr(raw_amr)
        inst_list.append(inst)
    with open(output_path, 'w') as fout:
        for inst in inst_list:
            fout.write(json.dumps(inst)+'\n')


# tokenize the sentece again

# convert the start and end idx in nodes



 