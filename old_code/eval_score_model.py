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
from score_model import ScoreGraph
from graph import Graph
from config import Config
from data import IEDataset, AuxDatasetEval
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

# this function is to load score model
def load_score_model(model_path, device=0, gpu=True):
    print('Loading the score model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    # print('load score model config: ',config.to_dict())
    config.bert_cache_dir = 'bert_cache'
    vocabs = state['vocabs']

    # recover the model
    model = ScoreGraph(config, vocabs)
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
    config.optimizer = new_config.optimizer
    config.aux_train_file = new_config.aux_train_file
    config.train_amr = new_config.train_amr
    config.min_amr_length = new_config.min_amr_length

    return config

def sanity_check(ie_config, score_config):
    # the tokennizer must be the same
    assert ie_config.bert_model_name == score_config.bert_model_name


# this function will perform one training step for both labeled and unlabeled dataset 
def train_batch(model, score_model=None, batch=None, aux_batch=None, mix_ratio=1.0, feedback=True):
    if aux_batch is None:
        loss = model(batch)
        avg_scores = 0.0
    else:
        # use the ie mode lto make prediction
        model.eval()
        with torch.no_grad():
            pred_graphs = model.predict(aux_batch)
        
        # for i, graph in enumerate(pred_graphs):
        #     # print('pred graph triggers: ', graph.triggers)
        #     triggers = set([ (span[0],span[1]) for span in graph.triggers])
        #     # print('pred graph entities: ', graph.entities)
        #     entities = set([ (span[0],span[1]) for span in graph.entities])
        #     # print('amr graph: ', aux_batch.amr_infos[i]['all_nodes'])
        #     all_nodes = aux_batch.amr_infos[i]['all_nodes']
        #     if len(triggers)>0:
        #         print('trigger score ',len(triggers.intersection(all_nodes))/len(triggers) if len(triggers)>0 else 0)
        #         print('entity score ',len(entities.intersection(all_nodes))/len(entities) if len(entities)>0 else 0)


        # convert prediction to training data
        ie_batch = pred2input(pred_graphs, aux_batch, model.vocabs)
        if feedback:
            # # convert prediction to amr input
            amr_batch = pred2amr(pred_graphs, aux_batch, score_model.vocabs, model.vocabs)

            with torch.no_grad():
                scores = score_model.measure_compat(amr_batch)
                # print('Avg feedback: ',sum(scores)/len(scores))
            # # change model back to training mode
            avg_scores = sum(scores)/len(scores)
            model.train()
            # compute self training loss
            loss = model(aux_batch=ie_batch, compat_scores=scores) * mix_ratio
        else:
            model.train()
            # compute self training loss
            loss = model(aux_batch=ie_batch) * mix_ratio
            avg_scores = 0.0

        # loss = 0.0

        # convert ie graphs to input into amr model
        # use score model to compute score

        # aux_batch = pred2input(graphs, aux_batch) 
        # model.train()
        # loss = model(aux_batch=aux_batch)

        # first current ie model makes an prediction with no grad
        # covert the prediction into tensors for score model
        # use the score model to determine compatibility score for arg roles
        # input the prediction and score as label into ie model
        # this time only update the encoder and role classification layer
    return loss, avg_scores

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
train_set = IEDataset(config.train_file, if_train=True, gpu=use_gpu,
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

# TODO: load unlabled dataset (Done)
print("loading aux dev file")
aux_dev_set = AuxDatasetEval(config.aux_train_file, config.train_amr, gpu=use_gpu)
print("Finish loading aux dev file")

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

# now check if load IE model
if args.load:
    # use some of current config
    new_config = copy.deepcopy(config)
    model, tokenizer, config, valid_patterns = load_model(config.load_model_path, device=config.gpu_device, gpu=use_gpu, beam_size=config.beam_size, use_global_features=config.use_global_features)
    vocabs = model.vocabs
    config = copy2load_config(config, new_config)
    print('Load model config: ',config.to_dict())

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
print("Numberizing the aux dataset")
aux_dev_set.numberize(tokenizer, score_vocabs)
print("Finsh numberizing the aux dataset")

batch_num = len(train_set) // config.batch_size
dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

# optimizer
# param_groups = [
#     {
#         'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
#         'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
#     }, # bert
#     {
#         'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
#                    and 'crf' not in n and 'global_feature' not in n],
#         'lr': config.learning_rate, 'weight_decay': config.weight_decay
#     }, # classification
#     {
#         'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
#                    and ('crf' in n or 'global_feature' in n)],
#         'lr': config.learning_rate, 'weight_decay': 0
#     } # crf and global
# ]

# print("Use {} optimizer and {} lr rate.".format(config.optimizer, config.learning_rate))
# if config.optimizer == 'adamw':
#     optimizer = AdamW(params=param_groups)
# elif config.optimizer == 'sgd':
#     optimizer = SGD(model.parameters(), config.learning_rate)
# else:
#     raise NotImplementedError("optimizer should be in [adamw, sgd]")

# schedule = get_linear_schedule_with_warmup(optimizer,
#                                            num_warmup_steps=batch_num * config.warmup_epoch,
#                                            num_training_steps=batch_num * config.max_epoch)
# model state
state = dict(model=model.state_dict(),
             config=config.to_dict(),
             vocabs=vocabs,
             valid=valid_patterns)

# global_step = 0

tasks = ['entity', 'trigger', 'relation', 'role']
best_dev = {k: 0 for k in tasks}
for epoch in range(config.max_epoch):
    print('Epoch: {}'.format(epoch))

    # training set
    # model.beam_size = config.beam_size
    # progress = tqdm.tqdm(total=batch_num,
    #                      desc='Train {}'.format(epoch))
    # optimizer.zero_grad()

    # train_loss = 0.0
    # train_self_loss = 0.0
    # train_agree_score = 0.0
    # batch_loss = 0.0
    # batch_self_loss = 0.0
    # batch_agree_score = 0.0
    # for batch_idx, (batch, aux_batch) in enumerate(zip(DataLoader(
    #         train_set, batch_size=config.batch_size // config.accumulate_step,
    #         shuffle=True, drop_last=True, collate_fn=train_set.collate_fn), DataLoader(
    #         aux_train_set, batch_size=config.batch_size // config.accumulate_step,
    #         shuffle=True, drop_last=True, collate_fn=aux_train_set.collate_fn))):

    #     loss, _ = train_batch(model, score_model=score_model, batch=batch)
    #     # if not config.further_train:
    #     #     print('supervised loss', loss.item())
    #     loss = loss * (1 / config.accumulate_step)
    #     batch_loss += loss.item()
    #     loss.backward()

    #     # print(config.further_train)
    #     if not config.further_train:
    #         self_loss, compat_scores = train_batch(model, score_model=score_model, aux_batch=aux_batch, feedback=config.feedback)
    #         # print('self-training loss', loss.item())
    #         self_loss = self_loss * (1 / config.accumulate_step)
    #         batch_self_loss += self_loss.item()
    #         batch_agree_score += compat_scores * (1 / config.accumulate_step)
    #         self_loss.backward()

    #     if (batch_idx + 1) % config.accumulate_step == 0:
    #         progress.update(1)
    #         global_step += 1
    #         torch.nn.utils.clip_grad_norm_(
    #             model.parameters(), config.grad_clipping)
    #         optimizer.step()
    #         schedule.step()
    #         optimizer.zero_grad()
    #         train_loss += batch_loss
    #         train_self_loss += batch_self_loss
    #         train_agree_score += batch_agree_score
    #         batch_loss = 0.0
    #         batch_self_loss = 0.0
    #         batch_agree_score = 0.0
    #         progress.set_description(f"loss {round(train_loss/(batch_idx+1),4)}, self-loss {round(train_self_loss/(batch_idx+1),4)}, score {round(train_agree_score/(batch_idx+1),4)}")


    # progress.close()

    # dev set
    model.beam_size = config.test_beam_size
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                         desc='Dev {}'.format(epoch))
    best_dev_role_model = False
    dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
    for batch, aux_batch in zip(DataLoader(dev_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=dev_set.collate_fn), DataLoader(aux_dev_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=aux_dev_set.collate_fn)):
        # print('batch', batch.sent_ids)
        # print('aux batch', aux_batch.sent_ids)
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
        
        amr_batch = pred2amr(graphs, aux_batch, score_model.vocabs, model.vocabs)
        with torch.no_grad():
            graphs = score_model.predict_on_oneie(amr_batch, model.vocabs, batch.graphs)
            
        # need to modify prediction based on score model
        


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
