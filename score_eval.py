"""This code evaluate the score model's performance of identifying the correctness of the IE prediction
"""

import os
import json
import time
import dgl
from argparse import ArgumentParser

import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (RobertaTokenizer, RobertaConfig, AdamW,
                          get_linear_schedule_with_warmup)
from transformers import (BertTokenizer, BertConfig, AdamW,
                          get_linear_schedule_with_warmup)

from graph_score_model import GraphScoreModel
from sequence_score_model import SeqScoreGraph
from baseline_score_model import BaselineScoreModel
from graph import Graph
from config import Config
from score_data import IEGraphDataset
from scorer import score_graphs
from score_util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task, pred2score
from baseline_model import OneIE # for now let's use roberta model


def load_model(model_path, device=3, beam_size=1):
    # cur_dir = os.path.dirname(os.path.realpath(__file__))
    print('Loading the IE model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device)
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
    print('ie model beam size: ',model.beam_size)
    model.cuda(device)
    if config.bert_model_name.startswith("roberta"):
        tokenizer = RobertaTokenizer.from_pretrained(config.bert_model_name,
                                              do_lower_case=False)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.bert_model_name,
                                                cache_dir=config.bert_cache_dir,
                                                do_lower_case=False)
    return model, tokenizer, config, valid_patterns

def load_score_model(model_path, config, device=0, gpu=True):
    print('Loading the score model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)
    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    config.bert_cache_dir = 'bert_cache'
    vocabs = state['vocabs']
    if config.model_type == 'baseline_score':
        print('eval baseline score model')
        model = BaselineScoreModel(config, vocabs)
    elif config.model_type == 'seq_score':
        print('eval sequence score model')
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

def eval_batch(model, ie_model, batch, epoch):
    model.eval()
    ie_model.eval()
    with torch.no_grad():
        pred_graphs = ie_model.predict(batch)
        for graph in pred_graphs:
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
        score_batch = pred2score(pred_graphs,batch,ie_model.vocabs,model.vocabs)
        batch_role_num, batch_gold_score, batch_pred_score, batch_each_type_result = model.evaluate(score_batch, batch.graphs)
    return batch_role_num, batch_gold_score, batch_pred_score, batch_each_type_result

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('-n', '--name', default='experiment')
parser.add_argument('-g', '--gpu', type=int, default=0)
parser.add_argument('-m', '--ie_model', default='')
parser.add_argument('-s', '--score_model', default='')
args = parser.parse_args()
config = Config.from_json_file(args.config)
config.gpu_device = args.gpu
print("Run training on GPU " + str(config.gpu_device))
config.ie_model_path = args.ie_model
config.score_model_path = args.score_model

# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# output
# output_dir = os.path.join(config.log_path, args.name)
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
# log_file = os.path.join(output_dir, 'log.txt')
# with open(log_file, 'w', encoding='utf-8') as w:
#     w.write(json.dumps(config.to_dict()) + '\n')
#     print('Log file: {}'.format(log_file))
# best_role_model = os.path.join(output_dir, 'best.role.mdl')
# final_model = os.path.join(output_dir, 'final.mdl')
# dev_role_result_file = os.path.join(output_dir, 'role.dev.json')
# test_role_result_file = os.path.join(output_dir, 'role.test.json')

# datasets
model_name = config.bert_model_name
if config.bert_model_name.startswith("roberta"):
    tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=False)
else:
    tokenizer = BertTokenizer.from_pretrained(model_name,
                                              cache_dir=config.bert_cache_dir,
                                              do_lower_case=False)
train_set = IEGraphDataset(config.train_file, config.train_amr, config, gpu=use_gpu,
                      relation_mask_self=config.relation_mask_self,
                      relation_directional=config.relation_directional,
                      symmetric_relations=config.symmetric_relations,
                      ignore_title=config.ignore_title)
dev_set = IEGraphDataset(config.dev_file, config.dev_amr, config, gpu=use_gpu,
                    relation_mask_self=config.relation_mask_self,
                    relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations,train=False)
test_set = IEGraphDataset(config.test_file, config.test_amr, config, gpu=use_gpu,
                     relation_mask_self=config.relation_mask_self,
                     relation_directional=config.relation_directional,
                     symmetric_relations=config.symmetric_relations,train=False)
# vocabs = generate_vocabs([train_set, dev_set, test_set])
model, tokenizer, _, vocabs = load_score_model(config.score_model_path, config, device=config.gpu_device, gpu=use_gpu)
model.eval()
print("Finish loading score model")

train_set.numberize(tokenizer, vocabs)
dev_set.numberize(tokenizer, vocabs)
test_set.numberize(tokenizer, vocabs)
batch_num = len(train_set) // config.batch_size
dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
        'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert')],
        'lr': config.learning_rate, 'weight_decay': config.weight_decay
    }
    # {
    #     'params': graph_params, 'lr': config.learning_rate, 'weight_decay': 0
    # }
]
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * config.warmup_epoch,
                                           num_training_steps=batch_num * config.max_epoch)
# model state
state = dict(model=model.state_dict(),
             config=config.to_dict(),
             vocabs=vocabs)

# load ie model
ie_model, _, _, _ = load_model(config.ie_model_path, device=config.gpu_device,beam_size=config.beam_size)

best_dev = 0.0
for epoch in range(1):
    print('Epoch: {}'.format(epoch))

    total_accu_result = {'total_num':0.0, 'tp':0.0, 'fp':0.0,'fn':0.0,'gtp':0.0, 'gfp':0.0,'gfn':0.0 }
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                         desc='Dev {}'.format(epoch))
    # best_dev_role_model = False
    # dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
    total_role_num = 0
    total_tp = 0
    total_gold_score = 0
    total_pred_score = 0
    total_role_num = 0
    for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=dev_set.collate_fn):
        progress.update(1)
        batch_role_num, batch_gold_score, batch_pred_score, batch_accur_result = eval_batch(model, ie_model, batch, epoch)
        total_role_num += batch_role_num
        total_gold_score += batch_gold_score
        total_pred_score += batch_pred_score
        for cri, value in batch_accur_result.items():
            total_accu_result[cri] += value
    
    print('total num {}'.format(total_accu_result['total_num']))
    tp = total_accu_result['tp']
    fp = total_accu_result['fp']
    fn = total_accu_result['fn']
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f1 = 2*p*r/(p+r)

    gtp = total_accu_result['gtp']
    gfp = total_accu_result['gfp']
    gfn = total_accu_result['gfn']
    gp = gtp/(gtp+gfp)
    gr = gtp/(gtp+gfn)
    gf1 = 2*gp*gr/(gp+gr)
    print('precision {}'.format(p))
    print('recall {}'.format(r))
    print('F1 {}'.format(f1))
    # print('gprecision {}'.format(gp))
    # print('grecall {}'.format(gr))
    # print('gF1 {}'.format(gf1))

    # print('correct accuracy: ',total_accu_result['corr_corr_num']/total_accu_result['total_corr_num'])
    # print('false accuracy: ',total_accu_result['false_corr_num']/total_accu_result['total_false_num'])
    # print('total accuracy: ',total_accu_result['both_corr_num']/total_accu_result['total_num'])
    # print('gold accuracy: ',total_accu_result['gold_corr_num']/total_accu_result['total_num'])
    # print('gold score: ',total_gold_score/total_accu_result['total_num'])
    # print('pred score: ',total_pred_score/total_accu_result['total_num'])
    # print('---------------------------------------------------------------')
    # print()

    # test set
    total_accu_result = {'total_num':0.0, 'tp':0.0, 'fp':0.0,'fn':0.0,'gtp':0.0, 'gfp':0.0,'gfn':0.0 }
    progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                         desc='Test {}'.format(epoch))
    # test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []
    total_role_num = 0
    total_tp = 0
    total_gold_score = 0
    total_pred_score = 0
    total_role_num = 0
    for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False,
                            collate_fn=test_set.collate_fn):
        progress.update(1)
        batch_role_num, batch_gold_score, batch_pred_score, batch_accur_result = eval_batch(model, ie_model, batch, epoch)
        total_role_num += batch_role_num
        total_gold_score += batch_gold_score
        total_pred_score += batch_pred_score
        for cri, value in batch_accur_result.items():
            total_accu_result[cri] += value
        
    # print('correct accuracy: ',total_accu_result['corr_corr_num']/total_accu_result['total_corr_num'])
    # print('false accuracy: ',total_accu_result['false_corr_num']/total_accu_result['total_false_num'])
    # print('total accuracy: ',total_accu_result['both_corr_num']/total_accu_result['total_num'])
    # print('gold accuracy: ',total_accu_result['gold_corr_num']/total_accu_result['total_num'])
    # print('gold score: ',total_gold_score/total_accu_result['total_num'])
    # print('pred score: ',total_pred_score/total_accu_result['total_num'])
    # print('---------------------------------------------------------------')
    # print()
    print('total num {}'.format(total_accu_result['total_num']))
    tp = total_accu_result['tp']
    fp = total_accu_result['fp']
    fn = total_accu_result['fn']
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f1 = 2*p*r/(p+r)

    gtp = total_accu_result['gtp']
    gfp = total_accu_result['gfp']
    gfn = total_accu_result['gfn']
    gp = gtp/(gtp+gfp)
    gr = gtp/(gtp+gfn)
    gf1 = 2*gp*gr/(gp+gr)
    print('precision {}'.format(p))
    print('recall {}'.format(r))
    print('F1 {}'.format(f1))
    # print('gprecision {}'.format(gp))
    # print('grecall {}'.format(gr))
    # print('gF1 {}'.format(gf1))
