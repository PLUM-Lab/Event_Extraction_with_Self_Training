"""
this code is to convert any tokenization to roberta,
"""

import math
import json
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import RobertaTokenizerFast
from nltk.tokenize import word_tokenize

def tokens_to_offset_list(tokens_list):
    tokens_num = len(tokens_list)
    curr_len = 0
    offsets_list = []

    for i in range(tokens_num):
        start = curr_len
        end = curr_len + len(tokens_list[i])
        curr_len = end + 1

        offsets_list.append((start, end))
    
    return offsets_list

def span_to_offset(offset_list, start, end):
    # offset_list: list of tuples, span: [s1, s2]
    start_offset = offset_list[start][0]
    end_offset = offset_list[end - 1][1]
    return start_offset, end_offset

def transform_for_amrie(data_dict, tokenizer):
    new_data_dict = data_dict.copy()

    tokens = data_dict["tokens"]
    recomb_sent = " ".join(tokens)

    tmp_nodes = []
    for i, node in enumerate(data_dict["nodes"]):
        span = data_dict["nodes"][node]
        tmp_node = {}
        tmp_node['node_id'] = node
        tmp_node["start"] = span[0]
        tmp_node["end"] = span[1]
        # print(tmp_node)
        tmp_nodes.append(tmp_node)

        
    # generate the string offsets for each token
    offset_list = tokens_to_offset_list(tokens)

    tokens_offset_list = []
    for i, token in enumerate(tokens):
        new_start, new_end = span_to_offset(offset_list, i, i+1)
        tokens_offset_list.append({'token':token,'start':new_start,'end':new_end})
    
    
    # do roberta tokenization
    pieces = tokenizer.tokenize(recomb_sent)
    offset_mappings = tokenizer(recomb_sent, return_offsets_mapping=True)["offset_mapping"][1:-1].copy()
    offset_mappings.insert(0, (-math.inf, -math.inf))
    offset_mappings.append((math.inf, math.inf))

    tokenslist = []
    for i in range(len(pieces)):
        if pieces[i].startswith('\u0120'):
            tokenslist.append(pieces[i][1:])
        else:
            tokenslist.append(pieces[i])
    assert (len(tokenslist) == len(pieces))
    
    
    # map the spans back to the offset mappings
    token_lens = []
    start_end = []
    for i, value in enumerate(tokens_offset_list):
        token_start = value["start"]
        token_end = value["end"]
        token = value['token']

        # first find out the minimum start
        for j in range(1, len(offset_mappings)):
            if offset_mappings[j][0] == offset_mappings[j][1]:
                if  offset_mappings[j][0] <= token_start and offset_mappings[j+1][0] == token_start:
                    break
            # if offset_mappings[j][0] <= token_start and offset_mappings[j+1][0] > token_start:
            #     if offset_mappings[j] == offset_mappings[j+1]:
            #         j-=1
            #         break
            #     elif offset_mappings[j] == offset_mappings[j-1]:
            #         j-=1
            #         break
            if offset_mappings[j][0] <= token_start and offset_mappings[j+1][0] > token_start:
                break
        if offset_mappings[j] == offset_mappings[j-1]:
            j-=1
        
        span_start = j - 1

        # then find out the minimum end
        for j in range(0, len(offset_mappings)-1):
            # if offset_mappings[j][1] < token_end and offset_mappings[j+1][1] >= token_end:
            #     if offset_mappings[j] == offset_mappings[j+1]:
            #         j+=1
            #         break
                # elif offset_mappings[j] == offset_mappings[j-1]:
                #     j+=1
                #     break
            if offset_mappings[j][1] < token_end and offset_mappings[j+1][1] >= token_end:
                break
        if offset_mappings[j] == offset_mappings[j-1]:
            j+=1 
        # elif offset_mappings[j] == offset_mappings[j+1]:
        #     j+=1
        span_end = j + 1
        
        token_lens.append(span_end-span_start)
        start_end.append((span_start,span_end))
        
    new_data_dict["pieces"] = pieces
    new_data_dict["token_lens"] = token_lens
    new_data_dict["sentence"] = recomb_sent
    new_data_dict["nodes"] = tmp_nodes

    return new_data_dict


def transform_dataset(data_dir, output_dir, tokenizer):
    # NYT
    # with open(data_dir, "r", encoding="utf-8") as f:
    #     with open(output_dir, "w", encoding="utf-8") as f1:
    #         sent_id = 1
    #         for line in f:
    #             num_problem = 0
    #             line = json.loads(line)
    #             if line['sentence'] != "":
    #                 line['tokens'] = word_tokenize(line['sentence'])
    #                 output_dict = transform_for_amrie(line, tokenizer)
    #                 if output_dict is None:
    #                     num_problem+=1
    #                     continue
    #                 output_line = json.dumps(output_dict) + '\n'
    #                 f1.write(output_line)
    #                 sent_id+=1
    #         print('instance that has mismatch token length and word pieces: ',num_problem)
    
    # AMR 3.0
    with open(data_dir, "r", encoding="utf-8") as f:
        with open(output_dir, "w", encoding="utf-8") as f1:
            for line in tqdm(f):
                num_problem = 0
                line = json.loads(line)
                output_dict = transform_for_amrie(line, tokenizer)
                if output_dict is None:
                    num_problem+=1
                    continue
                output_line = json.dumps(output_dict) + '\n'
                f1.write(output_line)
            print('instance that has mismatch token length and word pieces: ',num_problem)

if __name__ == "__main__":
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default='./dataset/aligned_amr3.0.amr-ie/training_all.txt')
    parser.add_argument('-o', '--output', default='./dataset/aligned_amr3.0.amr-ie/aux_amr3.0.jsonl')
    args = parser.parse_args()
    transform_dataset(args.input, args.output, tokenizer)


