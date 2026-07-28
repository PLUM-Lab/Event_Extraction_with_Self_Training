"""
Load in pretrained event extraction model and compute per role type performance.
"""

import os
import json
import time
import copy
from argparse import ArgumentParser

import tqdm
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from transformers import (RobertaTokenizer, RobertaConfig, AdamW,
                          get_linear_schedule_with_warmup)
from transformers import (BertTokenizer, BertConfig, AdamW,
                          get_linear_schedule_with_warmup)
from baseline_model import OneIE # for now let's use roberta model
from graph import Graph
from config import Config
from data import IEDataset
from scorer_detailed import score_graphs
from util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task, pred2input


# --------------------------------- functions ------------------------------------ #
# this function is to load ie model
def load_model(model_path, device=0, gpu=True):
    # cur_dir = os.path.dirname(os.path.realpath(__file__))
    print('Loading the IE model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    config.bert_cache_dir = 'bert_cache'
    vocabs = state['vocabs']
    valid_patterns = state['valid']

    # recover the model
    model = OneIE(config, vocabs, valid_patterns)
    model.load_bert(config.bert_model_name, cache_dir=config.bert_cache_dir)
    model.load_state_dict(state['model'])
    if gpu:
        model.cuda(device)
    if config.bert_model_name.startswith("roberta"):
        tokenizer = RobertaTokenizer.from_pretrained(config.bert_model_name,
                                              do_lower_case=False)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.bert_model_name,
                                                cache_dir=config.bert_cache_dir,
                                                do_lower_case=False)

    return model, tokenizer, config, valid_patterns

# -------------------------------------------------------------------------------------- #


# ----------------------------- configuration ------------------------------------#
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
# parser.add_argument('-n', '--name', default='temp')
parser.add_argument('-g', '--gpu', type=int, default=0)
# parser.add_argument('-l', '--load', default=False, action='store_true')
parser.add_argument('-m', '--load_model_path', default='')

args = parser.parse_args()
config = Config.from_json_file(args.config)
config.gpu_device = args.gpu
config.load_model_path = args.load_model_path
print("Run training on GPU " + str(config.gpu_device))
# -------------------------------------------------------------------------------#

# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# load IE dataset
dev_set = IEDataset(config.dev_file, gpu=use_gpu,
                    relation_mask_self=config.relation_mask_self,
                    relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations)
test_set = IEDataset(config.test_file, gpu=use_gpu,
                    relation_mask_self=config.relation_mask_self,
                    relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations)


model, tokenizer, config, valid_patterns = load_model(config.load_model_path, device=config.gpu_device, gpu=use_gpu)
vocabs = model.vocabs
print('Load model config: ',config.to_dict())

print('IE model vocab')
print('entity: ',vocabs['entity_type'])
print('trigger: ',vocabs['event_type'])
print('role: ',vocabs['role_type'])
print('relation: ',vocabs['relation_type'])


dev_set.numberize(tokenizer, vocabs)
test_set.numberize(tokenizer, vocabs)
dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)



model.beam_size = config.test_beam_size
print('mode beam size ',model.beam_size)

epoch = 0
# dev set
# progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
#                         desc='Dev {}'.format(epoch+1))
# dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
# for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
#                         shuffle=False, collate_fn=dev_set.collate_fn):
#     progress.update(1)
#     with torch.no_grad():
#         graphs = model.predict(batch)
#     if config.ignore_first_header:
#         for inst_idx, sent_id in enumerate(batch.sent_ids):
#             if int(sent_id.split('-')[-1]) < 4:
#                 graphs[inst_idx] = Graph.empty_graph(vocabs)
#     for graph in graphs:
#         graph.clean(relation_directional=config.relation_directional,
#                     symmetric_relations=config.symmetric_relations)
#         graph.clean_role()
#     dev_gold_graphs.extend(batch.graphs)
#     dev_pred_graphs.extend(graphs)
#     dev_sent_ids.extend(batch.sent_ids)
#     dev_tokens.extend(batch.tokens)
# progress.close()
# dev_scores = score_graphs(dev_gold_graphs, dev_pred_graphs, vocabs,
#                             relation_directional=config.relation_directional)

# test set
progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                        desc='Test {}'.format(epoch+1))
test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []
for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False,
                        collate_fn=test_set.collate_fn):
    progress.update(1)
    with torch.no_grad():
        graphs = model.predict(batch)
    if config.ignore_first_header:
        for inst_idx, sent_id in enumerate(batch.sent_ids):
            if int(sent_id.split('-')[-1]) < 4:
                graphs[inst_idx] = Graph.empty_graph(vocabs)
    for graph in graphs:
        graph.clean(relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations)
        graph.clean_role()
    test_gold_graphs.extend(batch.graphs)
    test_pred_graphs.extend(graphs)
    test_sent_ids.extend(batch.sent_ids)
    test_tokens.extend(batch.tokens)
progress.close()
test_scores = score_graphs(test_gold_graphs, test_pred_graphs, vocabs,
                            relation_directional=config.relation_directional)