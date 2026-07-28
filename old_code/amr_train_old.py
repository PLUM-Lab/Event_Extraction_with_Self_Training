import os
import json
import time
from argparse import ArgumentParser
import pickle

import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (RobertaTokenizer, RobertaConfig, AdamW,
                          get_linear_schedule_with_warmup)
from transformers import (BertTokenizer, BertConfig, AdamW,
                          get_linear_schedule_with_warmup)
from amr_ie_model import AMRIE
from graph import Graph
from config import Config
from score_data import IEGraphDataset
from scorer import score_graphs
from score_util import generate_vocabs 
from util import load_valid_patterns, save_result, best_score_by_task
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

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('-n', '--name', default='experiment')
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()
config = Config.from_json_file(args.config)
config.gpu_device = args.gpu
print("Run training on GPU " + str(config.gpu_device))

# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# output
# timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
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
    tokenizer = BertTokenizer.from_pretrained(model_name,
                                              cache_dir=config.bert_cache_dir,
                                              do_lower_case=False)

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
vocabs = generate_vocabs([train_set, dev_set, test_set])

train_set.numberize(tokenizer, vocabs)
dev_set.numberize(tokenizer, vocabs)
test_set.numberize(tokenizer, vocabs)
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
        }
    ]

optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * config.warmup_epoch,
                                           num_training_steps=batch_num * config.max_epoch)

# model state
state = dict(model=model.state_dict(),
             config=config.to_dict(),
             vocabs=vocabs,
             valid=valid_patterns)

global_step = 0
global_feature_max_step = int(config.global_warmup * batch_num) + 1
print('global feature max step:', global_feature_max_step)

tasks = ['entity', 'trigger', 'relation', 'role']
best_dev = {k: 0 for k in tasks}
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
            global_step += 1
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
