import os
import json
import time
import copy
from argparse import ArgumentParser
import random
import numpy as np

import tqdm
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from transformers import (RobertaTokenizer, RobertaConfig, AdamW,
                          get_linear_schedule_with_warmup)
from sequence_score_model import SeqScoreGraph
from amr_ie_model import AMRIE
from graph import Graph
from config import Config
from score_data import AuxDataset, IEGraphDataset
from scorer import score_graphs
from score_util import generate_vocabs, pred2score
from util import load_valid_patterns, save_result, best_score_by_task, pred2input
# from score_util import generate_vocabs


def load_model(model_path, device=0, gpu=True, beam_size=10, use_global_features=True):
    print('Loading the IE model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    # state = pickle.load(open(model_path, 'rb'))
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    config.bert_cache_dir = 'bert_cache'
    vocabs = state['vocabs']
    valid_patterns = state['valid']

    # recover the model
    model = AMRIE(config, vocabs, valid_patterns)
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
        raise NotImplementedError("language model should be roberta")

    return model, tokenizer, config, valid_patterns


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


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
    print('load seq score model')
    model = SeqScoreGraph(config, vocabs)
    model.load_bert(config.bert_model_name, cache_dir=config.bert_cache_dir)
    model.load_state_dict(state['model'])
    if gpu:
        model.cuda(device)
    return model, config, vocabs

# this function will perform one training step for both labeled and unlabeled dataset 
def train_batch(model, config, score_model=None, batch=None, aux_batch=None, mix_ratio=1.0, feedback=True):
    if aux_batch is None:
        model.beam_size = 10
        loss = model(batch)
        avg_role_scores = 0.0
        avg_trigger_scores = 0.0
        threshold_ratio = 0.0
    else:
        model.eval()
        model.beam_size = 1
        with torch.no_grad():
            pred_graphs = model.predict(aux_batch)
            for graph in pred_graphs:
                graph.clean(relation_directional=config.relation_directional,
                            symmetric_relations=config.symmetric_relations)
        ie_batch, role_masks, trigger_masks, count_role = pred2input(pred_graphs, aux_batch, model.vocabs, margin=config.score_threshold)
        model.train()
        if feedback:
            score_batch = pred2score(pred_graphs,aux_batch,model.vocabs,score_model.vocabs)
            with torch.no_grad():
                role_scores, count_role, trigger_scores, count_trigger, threshold_ratio  = score_model.feedback(score_batch,use_heuristic=config.use_heuristic,margin=config.score_threshold, threshold=config.score_threshold)
            avg_role_scores = sum(role_scores)/max(1,count_role)
            avg_trigger_scores = sum(trigger_scores)/max(1,count_trigger)
            if not config.use_trigger_score:
                trigger_scores = None
                avg_trigger_scores = 0.0
            loss = model(aux_batch=ie_batch, compat_scores=[role_scores, trigger_scores],trigger_weight=config.trigger_weight,count_role=count_role) * mix_ratio
        else:
            if not config.use_trigger_score:
                trigger_masks = None
                avg_trigger_scores = 0.0
            loss = model(aux_batch=ie_batch,compat_scores=[role_masks, trigger_masks],trigger_weight=config.trigger_weight,count_role=sum(role_masks)) * mix_ratio
            avg_role_scores = 0.0
            threshold_ratio = sum(role_masks)/max(1,count_role)
    return loss, avg_role_scores, avg_trigger_scores, threshold_ratio

# ----------------------------- configuration ------------------------------------#
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('-n', '--name', default='temp')
parser.add_argument('-g', '--gpu', type=int, default=0)
parser.add_argument('-m', '--mix_ratio', type=float, default=1.0)
parser.add_argument('-a', '--aux2train_ratio', type=int, default=10)
parser.add_argument('-f', '--feedback', default=False, action='store_true')
parser.add_argument('-ft', '--further_train', default=False, action='store_true')
parser.add_argument('-st', '--s_threshold', type=float, default=0.9)
args = parser.parse_args()

config = Config.from_json_file(args.config)
config.gpu_device = args.gpu
config.mix_ratio = args.mix_ratio
config.aux2train_ratio = args.aux2train_ratio
config.feedback = args.feedback
config.further_train = args.further_train
config.score_threshold = args.s_threshold
if config.further_train:
    assert not config.feedback
print("Run training on GPU " + str(config.gpu_device))

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

# datasets
model_name = config.bert_model_name
if config.bert_model_name.startswith("roberta"):
    tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=False)
else:
    raise NotImplementedError("language model should be roberta")

# set train false so won't ignore any instances
train_set = IEGraphDataset(config.train_file, config.train_amr, config, gpu=use_gpu,
                      relation_mask_self=config.relation_mask_self,
                      relation_directional=config.relation_directional,
                      symmetric_relations=config.symmetric_relations,
                      ignore_title=config.ignore_title,train=False)
dev_set = IEGraphDataset(config.dev_file, config.dev_amr, config, gpu=use_gpu,
                    relation_mask_self=config.relation_mask_self,
                    relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations,train=False)
test_set = IEGraphDataset(config.test_file, config.test_amr, config, gpu=use_gpu,
                     relation_mask_self=config.relation_mask_self,
                     relation_directional=config.relation_directional,
                     symmetric_relations=config.symmetric_relations,train=False)
print("loading aux train file")
aux_train_set = AuxDataset(config, gpu=use_gpu, require_len=len(train_set))
print("Finish loading aux train file")
print('Start to generate vocabs')

# TODO: load score model (Done)
print("Start loading score model")
score_model, score_config, score_vocabs = load_score_model(config.score_model_path, device=config.gpu_device, gpu=use_gpu)
score_model.eval()
print("Finish loading score model")

vocabs = generate_vocabs([train_set, dev_set, test_set])
train_set.numberize(tokenizer, vocabs)
dev_set.numberize(tokenizer, vocabs)
test_set.numberize(tokenizer, vocabs)
print("Numberizing the aux dataset")
aux_train_set.numberize(tokenizer, score_vocabs)
print("Finsh numberizing the aux dataset")
valid_patterns = load_valid_patterns(config.valid_pattern_path, vocabs)

batch_num = len(train_set) // config.batch_size
dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

# initialize the model
print('AMR-IE model config: ',config.to_dict())
model = AMRIE(config, vocabs, valid_patterns)
model.load_bert(model_name, cache_dir=config.bert_cache_dir)
if use_gpu:
    model.cuda(device=config.gpu_device)

# optimization for pretraining
if config.use_graph_encoder:
    graph_params = []
    for k in range(config.gnn_layers):
        for para in model.graph_encoder.gnn.gats[k].fc.parameters():
            graph_params.append(para)
        for para in model.graph_encoder.gnn.gats[k].attn_fc.parameters():
            graph_params.append(para) 
        for para in model.graph_encoder.gnn.gats[k].edge_fc.parameters():
            graph_params.append(para) 

    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
            'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                    and 'crf' not in n and 'global_feature' not in n],
            'lr': config.learning_rate, 'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                    and ('crf' in n or 'global_feature' in n)],
            'lr': config.learning_rate, 'weight_decay': 0
        },
        {
            'params': graph_params, 'lr': config.learning_rate, 'weight_decay': 0
        }
    ]

else:
    raise NotImplementedError("must use graph encoder")

optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * config.warmup_epoch,
                                           num_training_steps=batch_num * config.max_epoch)

# model state
state = dict(model=model.state_dict(),
             config=config.to_dict(),
             vocabs=vocabs,
             valid=valid_patterns)
tasks = ['entity', 'trigger', 'relation', 'role']
best_dev = {k: 0 for k in tasks}

# first pretrain steps
for epoch in range(config.max_epoch):
    print('Epoch: {}'.format(epoch))

    # training set
    progress = tqdm.tqdm(total=batch_num, ncols=75,
                         desc='Train {}'.format(epoch))
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(DataLoader(
            train_set, batch_size=config.batch_size // config.accumulate_step,
            shuffle=True, drop_last=True, collate_fn=train_set.collate_fn)):

        loss = model(batch)
        loss = loss * (1 / config.accumulate_step)
        loss.backward()

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    progress.close()

    # dev set
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                         desc='Dev {}'.format(epoch))
    best_dev_role_model = False
    dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
    for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=dev_set.collate_fn):
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
        dev_gold_graphs.extend(batch.graphs)
        dev_pred_graphs.extend(graphs)
        dev_sent_ids.extend(batch.sent_ids)
        dev_tokens.extend(batch.tokens)
    progress.close()
    dev_scores = score_graphs(dev_gold_graphs, dev_pred_graphs,
                              relation_directional=config.relation_directional)
    for task in tasks:
        if dev_scores[task]['f'] > best_dev[task]:
            best_dev[task] = dev_scores[task]['f']
            if task == 'role':
                print('Saving best role model')
                torch.save(state, best_role_model)
                best_dev_role_model = True
                save_result(dev_result_file,
                            dev_gold_graphs, dev_pred_graphs, dev_sent_ids,
                            dev_tokens)

    # test set
    progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                         desc='Test {}'.format(epoch))
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
print('Finish pretraining !')

# --------------------------------- start stf -------------------------- #
batch_num = len(train_set) // 10
# optimization for stf
if config.use_graph_encoder:
    stf_graph_params = []
    for k in range(config.gnn_layers):
        for para in model.graph_encoder.gnn.gats[k].fc.parameters():
            stf_graph_params.append(para)
        for para in model.graph_encoder.gnn.gats[k].attn_fc.parameters():
            stf_graph_params.append(para) 
        for para in model.graph_encoder.gnn.gats[k].edge_fc.parameters():
            stf_graph_params.append(para) 

    stf_param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
            'lr': 0.0001, 'weight_decay': 1.00E-05
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                    and 'crf' not in n and 'global_feature' not in n],
            'lr': 0.001, 'weight_decay': 5.00E-05
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                    and ('crf' in n or 'global_feature' in n)],
            'lr': 0.0, 'weight_decay': 0
        },
        {
            'params': stf_graph_params, 'lr': 0.001, 'weight_decay': 0
        }
    ]

else:
    raise NotImplementedError("must use graph encoder")

stf_optimizer = SGD(params=stf_param_groups)
stf_schedule = get_linear_schedule_with_warmup(stf_optimizer,
                                           num_warmup_steps=batch_num * 5,
                                           num_training_steps=batch_num * 10)
for epoch in range(10):
    print('Epoch: {}'.format(epoch))
    # training ie
    model.train()
    model.beam_size = 1
    progress = tqdm.tqdm(total=batch_num,
                         desc='Train {}'.format(epoch))
    stf_optimizer.zero_grad()

    train_loss = 0.0
    train_self_loss = 0.0
    batch_loss = 0.0
    batch_self_loss = 0.0
    train_role_score = 0.0
    train_trigger_score = 0.0
    batch_role_score = 0.0
    batch_trigger_score = 0.0
    train_threshold_ratio = 0.0
    batch_threshold_ratio = 0.0
    for batch_idx, (batch, aux_batch) in enumerate(zip(DataLoader(
            train_set, batch_size=10 // config.accumulate_step,
            shuffle=True, drop_last=True, collate_fn=train_set.collate_fn), DataLoader(
            aux_train_set, batch_size=10 // config.accumulate_step,
            shuffle=True, drop_last=True, collate_fn=aux_train_set.collate_fn))):

        loss, _, _, _ = train_batch(model, config, score_model=score_model, batch=batch)
        # if not config.further_train:
        #     print('supervised loss', loss.item())
        loss = loss * (1 / config.accumulate_step)
        batch_loss += loss.item()
        loss.backward()

        # print(config.further_train)
        if not config.further_train:
            self_loss, role_score, trigger_score, threshold_ratio = train_batch(model, config, score_model=score_model, aux_batch=aux_batch, mix_ratio=config.mix_ratio, feedback=config.feedback)
            # print('self-training loss', loss.item())
            self_loss = self_loss * (1 / config.accumulate_step)
            batch_self_loss += abs(self_loss.item())
            batch_role_score += role_score * (1 / config.accumulate_step)
            batch_trigger_score += trigger_score * (1 / config.accumulate_step)
            batch_threshold_ratio += threshold_ratio * (1 / config.accumulate_step)
            self_loss.backward()

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clipping)
            stf_optimizer.step()
            stf_schedule.step()
            stf_optimizer.zero_grad()
            train_loss += batch_loss
            train_self_loss += batch_self_loss
            train_role_score += batch_role_score
            train_trigger_score += batch_trigger_score
            train_threshold_ratio += batch_threshold_ratio
            batch_loss = 0.0
            batch_self_loss = 0.0
            batch_role_score = 0.0
            batch_trigger_score = 0.0
            batch_threshold_ratio = 0.0
            progress.set_description(f"loss {round(train_loss/(batch_idx+1),3)}, self-loss {round(train_self_loss/(batch_idx+1),3)}, r_score {round(train_role_score/(batch_idx+1),3)}, data_ratio {round(train_threshold_ratio/(batch_idx+1),3)}, t_score {round(train_trigger_score/(batch_idx+1),3)}")
    progress.close()

    # dev set
    model.beam_size = 10
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                         desc='Dev {}'.format(epoch+1))
    best_dev_role_model = False
    dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
    for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=dev_set.collate_fn):
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
        dev_gold_graphs.extend(batch.graphs)
        dev_pred_graphs.extend(graphs)
        dev_sent_ids.extend(batch.sent_ids)
        dev_tokens.extend(batch.tokens)
    progress.close()
    dev_scores = score_graphs(dev_gold_graphs, dev_pred_graphs,
                              relation_directional=config.relation_directional)
    for task in tasks:
        if dev_scores[task]['f'] > best_dev[task]:
            best_dev[task] = dev_scores[task]['f']
            if task == 'role':
                print('Saving best role model')
                torch.save(state, best_role_model)
                best_dev_role_model = True
                save_result(dev_result_file,
                            dev_gold_graphs, dev_pred_graphs, dev_sent_ids,
                            dev_tokens)

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
    test_scores = score_graphs(test_gold_graphs, test_pred_graphs,
                               relation_directional=config.relation_directional)

    if best_dev_role_model:
        save_result(test_result_file, test_gold_graphs, test_pred_graphs,
                    test_sent_ids, test_tokens)

    result = json.dumps(
        {'epoch': epoch+1, 'dev': dev_scores, 'test': test_scores})
    with open(log_file, 'a', encoding='utf-8') as w:
        w.write(result + '\n')
    print('Log file', log_file)

best_score_by_task(log_file, 'role')
