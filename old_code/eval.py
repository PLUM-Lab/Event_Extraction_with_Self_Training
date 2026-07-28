import os
import json
import time
import copy
from argparse import ArgumentParser

import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (RobertaTokenizer, RobertaConfig, AdamW,
                          get_linear_schedule_with_warmup)
from transformers import (BertTokenizer, BertConfig, AdamW,
                          get_linear_schedule_with_warmup)
from baseline_model import OneIE # for now let's use roberta model
from score_model_v2 import ScoreGraph
from graph import Graph
from config import Config
from data import IEDataset, AuxDataset
from scorer import score_graphs
from util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task, pred2input, pred2amr


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


def copy2load_config(config, new_config):
    # model
    config.beam_size = new_config.beam_size

    # data
    config.ignore_title = new_config.ignore_title
    config.ignore_first_header = new_config.ignore_first_header
    config.valid_pattern_path = new_config.valid_pattern_path

    # train
    config.accumulate_step = new_config.accumulate_step
    config.batch_size = new_config.batch_size
    config.eval_batch_size = new_config.eval_batch_size
    config.max_epoch = new_config.max_epoch
    config.learning_rate = new_config.learning_rate
    config.bert_learning_rate = new_config.bert_learning_rate
    config.weight_decay = new_config.weight_decay
    config.bert_weight_decay = new_config.bert_weight_decay
    config.warmup_epoch = new_config.warmup_epoch
    config.grad_clipping = new_config.grad_clipping

    # device
    config.use_gpu = new_config.use_gpu
    config.gpu_device = new_config.gpu_device

    #self-train
    config.test_beam_size = new_config.test_beam_size
    config.mix_ratio = new_config.mix_ratio
    config.feedback = new_config.feedback
    config.further_train = new_config.further_train

    return config

def sanity_check(ie_config, score_config):
    # the tokennizer must be the same
    assert ie_config.bert_model_name == score_config.bert_model_name

def calculate_F1(tp,fp,fn):
    if isinstance(tp, list):
        tp = float(sum(tp))
        fp = float(sum(fp))
        fn = float(sum(fn))
    if tp+fp == 0:
        return 0.0, 0.0, 0.0
    precision = tp/(tp+fp)
    if tp+fn == 0:
        return 0.0, 0.0, 0.0
    recall = tp/(tp+fn)
    if precision+recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2*precision*recall/(precision+recall)

    return round(f1,4), round(recall,4), round(precision,4)

# -------------------------------------------------------------------------------------- #


# ----------------------------- configuration ------------------------------------#
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('-n', '--name', default='temp')
parser.add_argument('-g', '--gpu', type=int, default=0)
parser.add_argument('-l', '--load', default=False, action='store_true')
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
log_file = os.path.join(output_dir, 'log.txt')
with open(log_file, 'w', encoding='utf-8') as w:
    w.write(json.dumps(config.to_dict()) + '\n')
    print('Log file: {}'.format(log_file))
best_role_model = os.path.join(output_dir, 'best.role.mdl')
dev_result_file = os.path.join(output_dir, 'result.dev.json')
test_result_file = os.path.join(output_dir, 'result.test.json')

# load IE dataset
train_set = IEDataset(config.train_file, gpu=use_gpu,
                        relation_mask_self=config.relation_mask_self,
                        relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations,
                        ignore_title=config.ignore_title)
dev_set = IEDataset(config.dev_file, gpu=use_gpu,
                    relation_mask_self=config.relation_mask_self,
                    relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations)
test_set = IEDataset(config.test_file, gpu=use_gpu,
                    relation_mask_self=config.relation_mask_self,
                    relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations)


# now check if load IE model
if args.load:
    # use some of current config
    new_config = copy.deepcopy(config)
    model, tokenizer, config, valid_patterns = load_model(config.load_model_path, device=config.gpu_device, gpu=use_gpu, beam_size=config.beam_size, use_global_features=config.use_global_features)
    vocabs = model.vocabs
    config = copy2load_config(config, new_config)
    print('load model config: ',config.to_dict())

else:
    # datasets
    model_name = config.bert_model_name
    if config.bert_model_name.startswith("roberta"):
        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=False)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name,
                                                cache_dir=config.bert_cache_dir,
                                                do_lower_case=False)
    vocabs = generate_vocabs([train_set, dev_set, test_set])
    valid_patterns = load_valid_patterns(config.valid_pattern_path, vocabs)
    # initialize the model
    model = OneIE(config, vocabs, valid_patterns)
    model.load_bert(model_name, cache_dir=config.bert_cache_dir)
    if use_gpu:
        model.cuda(device=config.gpu_device)


train_set.numberize(tokenizer, vocabs)
dev_set.numberize(tokenizer, vocabs)
test_set.numberize(tokenizer, vocabs)

dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

tasks = ['entity', 'trigger', 'relation', 'role']
best_dev = {k: 0 for k in tasks}
for epoch in range(config.max_epoch):
    print('Epoch: {}'.format(epoch))

    model.beam_size = config.test_beam_size

    # given gold event span and entity span
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                         desc='Dev gold even entity {}'.format(epoch))
    # best_dev_role_model = False
    # dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
    total_per_label_metric = {}
    for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=dev_set.collate_fn):
        progress.update(1)
        per_label_metric = model.role_predict(batch)
        for k, v in per_label_metric.items():
            if not k in total_per_label_metric:
                total_per_label_metric[k] = v
            else:
                total_per_label_metric[k]['tp']+=v['tp']
                total_per_label_metric[k]['fp']+=v['fp']
                total_per_label_metric[k]['fn']+=v['fn']
    for k, v in total_per_label_metric.items():
        f1 , recall , precision = calculate_F1(v['tp'],v['fp'],v['fn'])
        total_per_label_metric[k]['F1'] = f1
        total_per_label_metric[k]['recall'] = recall
        total_per_label_metric[k]['precision'] = precision
        print(total_per_label_metric[k])
    with open(os.path.join(output_dir, 'per_label.json'), 'w') as fout:
        json.dump(total_per_label_metric,fout)

    #     if config.ignore_first_header:
    #         for inst_idx, sent_id in enumerate(batch.sent_ids):
    #             if int(sent_id.split('-')[-1]) < 4:
    #                 graphs[inst_idx] = Graph.empty_graph(vocabs)
    #     for graph in graphs:
    #         graph.clean(relation_directional=config.relation_directional,
    #                     symmetric_relations=config.symmetric_relations)
    #     dev_gold_graphs.extend(batch.graphs)
    #     dev_pred_graphs.extend(graphs)
    #     dev_sent_ids.extend(batch.sent_ids)
    #     dev_tokens.extend(batch.tokens)
    # progress.close()
    # dev_scores = score_graphs(dev_gold_graphs, dev_pred_graphs,
    #                           relation_directional=config.relation_directional)

    # dev set
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                         desc='Dev {}'.format(epoch))
    best_dev_role_model = False
    dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
    for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=dev_set.collate_fn):
        progress.update(1)
        graphs = model.predict(batch)
        if config.ignore_first_header:
            for inst_idx, sent_id in enumerate(batch.sent_ids):
                if int(sent_id.split('-')[-1]) < 4:
                    graphs[inst_idx] = Graph.empty_graph(vocabs)
        for graph in graphs:
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
        dev_gold_graphs.extend(batch.graphs)
        dev_pred_graphs.extend(graphs)
        dev_sent_ids.extend(batch.sent_ids)
        dev_tokens.extend(batch.tokens)
    progress.close()
    dev_scores = score_graphs(dev_gold_graphs, dev_pred_graphs,
                              relation_directional=config.relation_directional)
    
    # test set
    progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                         desc='Test {}'.format(epoch))
    test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []
    for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False,
                            collate_fn=test_set.collate_fn):
        progress.update(1)
        graphs = model.predict(batch)
        if config.ignore_first_header:
            for inst_idx, sent_id in enumerate(batch.sent_ids):
                if int(sent_id.split('-')[-1]) < 4:
                    graphs[inst_idx] = Graph.empty_graph(vocabs)
        for graph in graphs:
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
        test_gold_graphs.extend(batch.graphs)
        test_pred_graphs.extend(graphs)
        test_sent_ids.extend(batch.sent_ids)
        test_tokens.extend(batch.tokens)
    progress.close()
    test_scores = score_graphs(test_gold_graphs, test_pred_graphs,
                               relation_directional=config.relation_directional)

    if best_dev_role_model:
        save_result(test_result_file, test_gold_graphs, test_pred_graphs,
                    test_sent_ids, test_tokens)

    result = json.dumps(
        {'epoch': epoch, 'dev': dev_scores, 'test': test_scores})
    with open(log_file, 'a', encoding='utf-8') as w:
        w.write(result + '\n')
    print('Log file', log_file)

best_score_by_task(log_file, 'role')
