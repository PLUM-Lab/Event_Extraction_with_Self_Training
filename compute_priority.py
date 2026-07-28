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
from graph_score_model import GraphScoreModel
from sequence_score_model import SeqScoreGraph
from graph import Graph
from config import Config
from data import IEDataset
from score_data import AuxDataset
from scorer import score_graphs
from util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task, pred2input 
from score_util import pred2score


# --------------------------------- functions ------------------------------------ #
# this function is to load ie model
def load_model(model_path, device=0, gpu=False, beam_size=1, use_global_features=False):
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
    model.beam_size = beam_size
    model.use_global_features = use_global_features
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

# this function is to load score model
def load_score_model(model_path, device=0, gpu=True):
    print('Loading the score model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)
    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    config.bert_cache_dir = 'bert_cache'
    vocabs = state['vocabs']
    model = SeqScoreGraph(config, vocabs)
    model.load_bert(config.bert_model_name, cache_dir=config.bert_cache_dir)
    model.load_state_dict(state['model'])
    if gpu:
        model.cuda(device)
    if config.bert_model_name.startswith("roberta"):
        tokenizer = RobertaTokenizer.from_pretrained(config.bert_model_name,
                                              do_lower_case=False)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name,
                                                cache_dir=config.bert_cache_dir,
                                                do_lower_case=False)
    return model, tokenizer, config, vocabs

# this function will perform one training step for both labeled and unlabeled dataset 
def train_batch(model, config, score_model=None, aux_batch=None, feedback=True):
    
    model.eval()
    model.beam_size = config.beam_size
    with torch.no_grad():
        pred_graphs = model.predict(aux_batch)
        for graph in pred_graphs:
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
    # convert prediction to training data
    ie_batch, role_masks, trigger_masks = pred2input(pred_graphs, aux_batch, model.vocabs, margin=config.score_threshold)
    model.train()
    if feedback:
        # # convert prediction to amr input
        # amr_batch = pred2amr(pred_graphs, aux_batch, score_model.vocabs, model.vocabs)
        score_batch = pred2score(pred_graphs,aux_batch,model.vocabs,score_model.vocabs)
        with torch.no_grad():
            output = score_model.label_data(score_batch,use_heuristic=config.use_heuristic,margin=config.score_threshold)
                
    return output 

# -------------------------------------------------------------------------------------- #


# ----------------------------- configuration ------------------------------------#
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('-n', '--name', default='temp')
parser.add_argument('-g', '--gpu', type=int, default=0)
parser.add_argument('-l', '--load', default=False, action='store_true')
parser.add_argument('-i', '--exp_idx', default='1')

args = parser.parse_args()
config = Config.from_json_file(args.config)
config.gpu_device = args.gpu
print("Run training on GPU " + str(config.gpu_device))
# -------------------------------------------------------------------------------#

# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)
# output
output_dir = os.path.join(config.log_path, args.name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
log_file = os.path.join(output_dir, 'labeled_dataset_'+args.exp_idx+'.txt')
print('save file to: ',log_file)
config.load_model_path = config.load_model_path.replace('2',args.exp_idx)
config.score_model_path = config.score_model_path.replace('2',args.exp_idx)
config.aux2train_ratio = -1

# TODO: load unlabled dataset (Done)
print("loading aux train file")
aux_train_set = AuxDataset(config, gpu=use_gpu)
print("Finish loading aux train file")

# TODO: load score model (Done)
score_model, score_tokenizer, score_config, score_vocabs = load_score_model(config.score_model_path, device=config.gpu_device, gpu=use_gpu)
score_model.eval()
print("Finish loading score model")
# aux_vocabs = score_model.vocabs


"""
batch.piece_idxs,
batch.attention_masks,
batch.token_lens
batch.token_nums
batch.amr_graphs a dictionary {}
"""

new_config = copy.deepcopy(config)
model, tokenizer, config, valid_patterns = load_model(config.load_model_path, device=config.gpu_device, gpu=use_gpu, beam_size=config.beam_size, use_global_features=config.use_global_features)
vocabs = model.vocabs
config = new_config
print('Load model config: ',config.to_dict())
print('IE model vocab')
print('entity: ',len(vocabs['entity_type']),[ k for k in vocabs['entity_type']])
print('trigger: ',len(vocabs['event_type']),[ k for k in vocabs['event_type']])
print('role: ',len(vocabs['role_type']),[ k for k in vocabs['role_type']])
print('relation: ',len(vocabs['relation_type']),[ k for k in vocabs['relation_type']])

print("Numberizing the aux dataset")
aux_train_set.numberize(tokenizer, score_vocabs)
print("Finsh numberizing the aux dataset")

batch_num = len(aux_train_set) // config.batch_size


out = []
for epoch in range(config.max_epoch):
    print('Epoch: {}'.format(epoch))

    # dev set
    model.beam_size = config.beam_size
    progress = tqdm.tqdm(total=batch_num, ncols=75,
                         desc='Dev {}'.format(epoch+1))

    for aux_batch in DataLoader(aux_train_set, batch_size=config.batch_size // config.accumulate_step,
            shuffle=True, drop_last=True, collate_fn=aux_train_set.collate_fn):
        progress.update(1)
        with torch.no_grad():
            output = train_batch(model, config, score_model=score_model, aux_batch=aux_batch)    
        out.extend(output)
    progress.close()

    break

with open(log_file,'w') as fout:
    for inst in out:
        fout.write(json.dumps(inst)+'\n')
    
    # result = json.dumps(
    #     {'epoch': epoch+1, 'dev': dev_scores, 'test': test_scores})
    # with open(log_file, 'a', encoding='utf-8') as w:
    #     w.write(result + '\n')
