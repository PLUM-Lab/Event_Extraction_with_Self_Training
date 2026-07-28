import math
import json
from argparse import ArgumentParser
from tqdm import tqdm

"""
This code should only compute the token len for pieces, the node spans should remain untouched.
This code is mainly for gold amr 3.0

"""
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

def transform_for_roberta(data_dict, tokenizer):
    new_data_dict = data_dict.copy()

    tokens = data_dict["tokens"]
    recomb_sent = " ".join(tokens)
    # generate the string offsets for each token
    offset_list = tokens_to_offset_list(tokens)
    # modify the dictionarys for entities, event triggers
    tmp_nodes = []
    for i, node in enumerate(data_dict["nodes"]):
        span = data_dict["nodes"][node]
        new_start, new_end = span_to_offset(offset_list, span[0], span[1])
        tmp_node = {}
        tmp_node['node_id'] = node
        tmp_node["start"] = new_start
        tmp_node["end"] = new_end
        # print(tmp_node)
        tmp_nodes.append(tmp_node)
    
    
    # do roberta tokenization
    pieces = tokenizer.tokenize(recomb_sent)
    offset_mappings = tokenizer(recomb_sent, return_offsets_mapping=True)["offset_mapping"][1:-1].copy()
    offset_mappings.insert(0, (-math.inf, -math.inf))
    offset_mappings.append((math.inf, math.inf))

    # map the spans back to the offset mappings
    for i, node in enumerate(tmp_nodes):
        node_start = node["start"]
        node_end = node["end"]

        # first find out the minimum start
        for j in range(1, len(offset_mappings)):
            if offset_mappings[j][0] <= node_start and offset_mappings[j+1][0] > node_start:
                break
        span_start = j - 1

        # then find out the minimum end
        for j in range(0, len(offset_mappings)-1):
            if offset_mappings[j][1] < node_end and offset_mappings[j+1][1] >= node_end:
                break
        span_end = j + 1

        tmp_nodes[i]['start'] = span_start
        tmp_nodes[i]['end'] = span_end

    new_data_dict["nodes"] = tmp_nodes
    
    # map the spans back to offset mappings
    # for i, event in enumerate(tmp_events):
    #     ent_start = event["start"]
    #     ent_end = event["end"]

    #     # first find out the minimum start
    #     for j in range(1, len(offset_mappings)):
    #         if offset_mappings[j][0] <= ent_start and offset_mappings[j+1][0] > ent_start:
    #             break
    #     span_start = j - 1

    #     # then find out the minimum end
    #     for j in range(0, len(offset_mappings)-1):
    #         if offset_mappings[j][1] < ent_end and offset_mappings[j+1][1] >= ent_end:
    #             break

    #     span_end = j + 1

    #     new_data_dict["event_mentions"][i]["trigger"]["start"] = span_start
    #     new_data_dict["event_mentions"][i]["trigger"]["end"] = span_end
    
    tokenslist = []

    for i in range(len(pieces)):
        if pieces[i].startswith('\u0120'):
            tokenslist.append(pieces[i][1:])
        else:
            tokenslist.append(pieces[i])
    
    assert (len(tokenslist) == len(pieces))

    # print(tokenslist)
    # for node in tmp_nodes:
    #     if node['end'] - node['start'] > 1:
            # print(tokenslist[node['start']:node['end']], node['start'],node['end'])


    new_data_dict["tokens"] = tokenslist
    new_data_dict["pieces"] = pieces
    new_data_dict["token_lens"] = [1 for _ in range(len(pieces))]
    new_data_dict["sentence"] = recomb_sent
    # print(new_data_dict)

    return new_data_dict


def transform_dataset(data_dir, output_dir, tokenizer):
    with open(data_dir, "r", encoding="utf-8") as fin:
        with open(output_dir, "w", encoding="utf-8") as fout:
            # done = 0
            # while not done:
            #     line = f.readline()
            #     if line != "":
            #         data_dict = json.loads(line)
            #         output_dict = transform_for_amrie(data_dict, tokenizer)
            #         output_line = json.dumps(output_dict) + '\n'
            #         f1.write(output_line)
            #     else:
            #         done = 1
            for line in tqdm(fin):
                data_dict = json.loads(line)
                output_dict = transform_for_roberta(data_dict, tokenizer)
                fout.write(json.dumps(output_dict)+'\n')


if __name__ == "__main__":
    from transformers import RobertaTokenizerFast
    t = RobertaTokenizerFast.from_pretrained("roberta-large")
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', default='dataset/aligned_amr3.0.amr-ie/aux_amr3.0.jsonl')
    parser.add_argument('-i', '--input', default='dataset/aligned_amr3.0.amr-ie/training_all.txt')
    args = parser.parse_args()
    transform_dataset(args.input, args.output, t)