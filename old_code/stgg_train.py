import sys
import os
import json
import time
import copy
from argparse import ArgumentParser
import random
import numpy as np

import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (RobertaTokenizer, RobertaConfig, AdamW,
                          get_linear_schedule_with_warmup)

from sequence_score_model import SeqScoreGraph
from data import IEDataset

from graph import Graph
from config import Config
from score_data import AuxDataset, IEGraphDataset
from scorer import score_graphs
from score_util import pred2score
from score_util import generate_vocabs as score_generate_vocabs
from util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task, pred2input 


# --------------------------------- functions ------------------------------------ #

def eval_batch(model, ie_model, batch, epoch, config):
    model.eval()
    ie_model.eval()
    with torch.no_grad():
        pred_graphs = ie_model.predict(batch)
        for graph in pred_graphs:
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
        score_batch = pred2score(pred_graphs,batch,ie_model.vocabs,model.vocabs)
        pred_graphs = model.predict(score_batch)
    return pred_graphs

def compute_score_steps(pretrained_epoch, max_epoch, stgg_every_epoch, score_max_epoch, score_every_stgg):
    stgg_epoch = (max_epoch - pretrained_epoch) // stgg_every_epoch
    total_epoch = score_max_epoch + score_every_stgg * stgg_epoch
    # print(max_epoch, pretrained_epoch,stgg_every_epoch , score_max_epoch, score_every_stgg, stgg_epoch)
    print('total number of score epoch',total_epoch)
    return total_epoch

def train_score_model(config, model, ie_model, best_dev, train_set, dev_set, test_set, num_epoch, state, optimizer, schedule, best_role_model, dev_result_file, test_result_file, log_file):
    model.train()
    batch_num = len(train_set) // config.batch_size
    dev_batch_num = len(dev_set) // config.eval_batch_size + \
        (len(dev_set) % config.eval_batch_size != 0)
    test_batch_num = len(test_set) // config.eval_batch_size + \
        (len(test_set) % config.eval_batch_size != 0)
    vocabs = model.vocabs
    
    for epoch in range(num_epoch):
        print('Epoch: {}'.format(epoch))

        # training set
        progress = tqdm.tqdm(total=batch_num, ncols=75,
                            desc='Train {}'.format(epoch))
        optimizer.zero_grad()
        train_loss = 0.0
        for batch_idx, batch in enumerate(DataLoader(
                train_set, batch_size=config.batch_size // config.accumulate_step,
                shuffle=True, drop_last=True, collate_fn=train_set.collate_fn)):

            output = model(batch, epoch)
            loss = output['loss']
            loss = loss * (1 / config.accumulate_step)
            loss.backward()
            train_loss += loss.item()

            if (batch_idx + 1) % config.accumulate_step == 0:
                progress.update(1)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clipping)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
        progress.close()
        print("Training Loss: --- " + str(train_loss / (batch_idx + 1)))

        # dev set
        progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                            desc='Dev {}'.format(epoch))
        best_dev_role_model = False
        dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
        for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                                shuffle=False, collate_fn=dev_set.collate_fn):
            progress.update(1)
            graphs = eval_batch(model, ie_model, batch, epoch, config)
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
        
        if dev_scores['role']['f'] > best_dev:
            best_dev = dev_scores['role']['f']
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
            graphs = eval_batch(model, ie_model, batch, epoch, config)
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
            {'epoch': epoch, 'loss': train_loss / (batch_idx + 1), 'dev': dev_scores, 'test': test_scores})
        with open(log_file, 'a', encoding='utf-8') as w:
            w.write(result + '\n')
        print('Log file', log_file)

def train_ie_model(config, model, best_dev, train_set, dev_set, test_set, num_epoch, state, optimizer, schedule, best_role_model, dev_result_file, test_result_file, log_file):
    batch_num = len(train_set) // config.batch_size
    dev_batch_num = len(dev_set) // config.eval_batch_size + \
        (len(dev_set) % config.eval_batch_size != 0)
    test_batch_num = len(test_set) // config.eval_batch_size + \
        (len(test_set) % config.eval_batch_size != 0)
    vocabs = model.vocabs
    model.train()
    model.beam_size = config.test_beam_size

    for epoch in range(num_epoch):
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
        if dev_scores['role']['f'] > best_dev:
            best_dev = dev_scores['role']['f']
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

def train_stgg(config, model, score_model, best_dev, train_set, aux_train_set, dev_set, test_set, num_epoch, state, optimizer, schedule, best_role_model, dev_result_file, test_result_file, log_file):

    batch_num = len(train_set) // config.batch_size
    dev_batch_num = len(dev_set) // config.eval_batch_size + \
        (len(dev_set) % config.eval_batch_size != 0)
    test_batch_num = len(test_set) // config.eval_batch_size + \
        (len(test_set) % config.eval_batch_size != 0)
    vocabs = model.vocabs
    score_model.eval()
    for epoch in range(num_epoch):
        print('Epoch: {}'.format(epoch))

        model.train()
        model.beam_size = config.beam_size
        progress = tqdm.tqdm(total=batch_num,
                            desc='Train {}'.format(epoch))
        optimizer.zero_grad()

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
                train_set, batch_size=config.batch_size // config.accumulate_step,
                shuffle=True, drop_last=True, collate_fn=train_set.collate_fn), DataLoader(
                aux_train_set, batch_size=config.aux_batch_size // config.accumulate_step,
                shuffle=True, drop_last=True, collate_fn=aux_train_set.collate_fn))):

            loss, _, _, _ = train_stgg_batch(model, config, score_model=score_model, batch=batch)
            # if not config.further_train:
            #     print('supervised loss', loss.item())
            loss = loss * (1 / config.accumulate_step)
            batch_loss += loss.item()
            loss.backward()

            # print(config.further_train)
            if not config.further_train:
                self_loss, role_score, trigger_score, threshold_ratio = train_stgg_batch(model, config, score_model=score_model, aux_batch=aux_batch, mix_ratio=config.mix_ratio, feedback=config.feedback)
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
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
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
        model.beam_size = config.test_beam_size
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

        if dev_scores['role']['f'] > best_dev:
                best_dev = dev_scores['role']['f']
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

def train_stgg_batch(model, config, score_model=None, batch=None, aux_batch=None, mix_ratio=1.0, feedback=True):
    if aux_batch is None:
        model.beam_size = config.test_beam_size
        loss = model(batch)
        avg_role_scores = 0.0
        avg_trigger_scores = 0.0
        threshold_ratio = 0.0
    else:
        # use the ie mode lto make prediction
        model.eval()
        model.beam_size = config.beam_size
        with torch.no_grad():
            pred_graphs = model.predict(aux_batch)
            for graph in pred_graphs:
                graph.clean(relation_directional=config.relation_directional,
                            symmetric_relations=config.symmetric_relations)
                # ------------------- debug code ------------------ #
                # role_type_itos = {i: s for s, i in model.vocabs['role_type'].items()} 
                # for rol, rol_score in  zip(graph.roles, graph.role_scores):
                #     print(role_type_itos[rol[2]], rol_score)

        # convert prediction to training data
        ie_batch, role_masks, trigger_masks, count_role = pred2input(pred_graphs, aux_batch, model.vocabs, margin=config.score_threshold)
        model.train()
        if feedback:
            # # convert prediction to amr input
            # amr_batch = pred2amr(pred_graphs, aux_batch, score_model.vocabs, model.vocabs)
            score_batch = pred2score(pred_graphs,aux_batch,model.vocabs,score_model.vocabs)

            with torch.no_grad():
                role_scores, count_role, trigger_scores, count_trigger, threshold_ratio  = score_model.feedback(score_batch,use_heuristic=config.use_heuristic,margin=config.score_threshold, threshold=config.score_threshold)
                """ margin: prob > margin the pseudo will be used to train score model
                    threshold: used to filter out score model pseudo labels
                    count_role: num of pseudo labels satisfy score_threshold 
                    threshold_ratio: percentage of pseudo labels satisfy score_threshold 
                """
                # print('Avg feedback: ',sum(scores)/len(scores))
            # # change model back to training mode
            avg_role_scores = sum(role_scores)/max(1,count_role)
            avg_trigger_scores = sum(trigger_scores)/max(1,count_trigger)
            # compute self training loss
            if not config.use_trigger_score:
                trigger_scores = None
                avg_trigger_scores = 0.0
            loss, _ = model(aux_batch=ie_batch, compat_scores=[role_scores, trigger_scores],trigger_weight=config.trigger_weight,count_role=count_role, count_trigger=count_trigger)
            loss = loss* mix_ratio
            # print('role loss',loss.item())
        else:
            if not config.use_trigger_score:
                trigger_masks = None
                avg_trigger_scores = 0.0
            avg_role_scores = 0.0
            loss, avg_role_scores = model(aux_batch=ie_batch,compat_scores=[None, trigger_masks],trigger_weight=config.trigger_weight, threshold=config.score_threshold)
            loss = loss* mix_ratio
            threshold_ratio = sum(role_masks)/max(1,count_role)
    return loss, avg_role_scores, avg_trigger_scores, threshold_ratio

# --------------------------------- temp function ------------------------------ #
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
    model = IEmodel(config, vocabs, valid_patterns)
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
    print('load seq score model')
    model = SeqScoreGraph(config, vocabs)
    model.load_bert(config.bert_model_name, cache_dir=config.bert_cache_dir)
    model.load_state_dict(state['model'])
    if gpu:
        model.cuda(device)
    return model, config, vocabs

def main(args):
    config = Config.from_json_file(args.config)
    config.gpu_device = args.gpu
    config.learning_rate = args.learning_rate
    config.bert_learning_rate = args.bert_learning_rate
    config.mix_ratio = args.mix_ratio
    config.aux2train_ratio = args.aux2train_ratio
    config.feedback = args.feedback
    config.further_train = args.further_train
    config.score_threshold = args.s_threshold
    config.use_trigger_score = args.use_trigger_score
    config.trigger_weight = args.trigger_weight

    
    print('# ---------------- load ie model config --------------------- #')
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

    # load IE dataset
    if args.ie_model == 'oneie':
        from data import IEDataset
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
    elif args.ie_model == 'amrie':
        raise NotImplementedError
    vocabs = generate_vocabs([train_set, dev_set, test_set])
    valid_patterns = load_valid_patterns(config.valid_pattern_path, vocabs)
    

    # ---------------------- initialize ie model ------------------------ #
    model_name = config.bert_model_name
    if config.bert_model_name.startswith("roberta"):
        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=False)
    else:
        raise NotImplementedError
    model = IEmodel(config, vocabs, valid_patterns)
    model.load_bert(model_name, cache_dir=config.bert_cache_dir)
    if use_gpu:
        model.cuda(device=config.gpu_device)
    # ---------------------- load ie model ------------------------ #
    # model, tokenizer, _, valid_patterns = load_model(config.load_model_path, device=config.gpu_device, gpu=use_gpu, beam_size=config.beam_size, use_global_features=config.use_global_features)
    # vocabs = model.vocabs


    print('IE Model config: ',config.to_dict())
    print('IE model vocab')
    print('entity: ',len(vocabs['entity_type']),[ k for k in vocabs['entity_type']])
    print('trigger: ',len(vocabs['event_type']),[ k for k in vocabs['event_type']])
    print('role: ',len(vocabs['role_type']),[ k for k in vocabs['role_type']])
    print('relation: ',len(vocabs['relation_type']),[ k for k in vocabs['relation_type']])

    train_set.numberize(tokenizer, vocabs)
    dev_set.numberize(tokenizer, vocabs)
    test_set.numberize(tokenizer, vocabs)
    batch_num = len(train_set) // config.batch_size
    
    # optimizer
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

    best_dev = 0.0



    
    print('# ---------------- loading score model config --------------------- #')
    score_config = Config.from_json_file(args.score_config)
    score_config.gpu_device = args.gpu

    score_output_dir = os.path.join(score_config.log_path, args.name)
    if not os.path.exists(score_output_dir):
        os.mkdir(score_output_dir)
    score_log_file = os.path.join(score_output_dir, 'log.txt')
    with open(score_log_file, 'w', encoding='utf-8') as w:
        w.write(json.dumps(config.to_dict()) + '\n')
        print('Log file: {}'.format(score_log_file))

    score_best_role_model = os.path.join(output_dir, 'best.role.mdl')
    score_dev_result_file = os.path.join(output_dir, 'role.dev.json')
    score_test_result_file = os.path.join(output_dir, 'role.test.json')

    score_train_set = IEGraphDataset(score_config.train_file, score_config.train_amr, score_config, gpu=use_gpu,
                        relation_mask_self=score_config.relation_mask_self,
                        relation_directional=score_config.relation_directional,
                        symmetric_relations=score_config.symmetric_relations,
                        ignore_title=score_config.ignore_title)
    score_dev_set = IEGraphDataset(score_config.dev_file, score_config.dev_amr, score_config, gpu=use_gpu,
                        relation_mask_self=score_config.relation_mask_self,
                        relation_directional=score_config.relation_directional,
                        symmetric_relations=score_config.symmetric_relations,train=False)
    score_test_set = IEGraphDataset(score_config.test_file, score_config.test_amr, score_config, gpu=use_gpu,
                        relation_mask_self=score_config.relation_mask_self,
                        relation_directional=score_config.relation_directional,
                        symmetric_relations=score_config.symmetric_relations,train=False)
    score_vocabs = score_generate_vocabs([score_train_set, score_dev_set, score_test_set])
    

    # -------------------- init score model --------------------------- #
    score_tokenizer = RobertaTokenizer.from_pretrained(score_config.bert_model_name, do_lower_case=False)
    if score_config.model_type == 'seq_score':
        print("Using seq score model")
        score_model = SeqScoreGraph(score_config, score_vocabs)
    else:
        raise NotImplementedError
    if use_gpu:
        score_model.cuda(device=config.gpu_device)

    # ------------------- load score model -------------------------- #
    # score_model, _, score_vocabs = load_score_model(config.score_model_path, device=config.gpu_device, gpu=use_gpu)

    print('Score Model config: ',score_config.to_dict())

    score_train_set.numberize(score_tokenizer, score_vocabs)
    score_dev_set.numberize(score_tokenizer, score_vocabs)
    score_test_set.numberize(score_tokenizer, score_vocabs)
    score_batch_num = len(score_train_set) // score_config.batch_size

    score_param_groups = [
        {
            'params': [p for n, p in score_model.named_parameters() if n.startswith('bert')],
            'lr': score_config.bert_learning_rate, 'weight_decay': score_config.bert_weight_decay
        },
        {
            'params': [p for n, p in score_model.named_parameters() if not n.startswith('bert')],
            'lr': score_config.learning_rate, 'weight_decay': score_config.weight_decay
        }
    ]
    score_optimizer = AdamW(params=score_param_groups)
    total_score_epoch = compute_score_steps(config.pretrained_epoch, config.max_epoch, config.stgg_every_epoch, score_config.pretrained_epoch, score_config.score_every_stgg)
    score_schedule = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=score_batch_num * score_config.warmup_epoch,
                                            num_training_steps=score_batch_num * total_score_epoch)
    score_state = dict(model=score_model.state_dict(),
                config=score_config.to_dict(),
                vocabs=score_vocabs)
    score_best_dev = 0.0

    print("loading aux train file")
    aux_train_set = AuxDataset(config, gpu=use_gpu, require_len=len(train_set))
    print("Numberizing the aux dataset")
    aux_train_set.numberize(tokenizer, score_vocabs)



    print('# ---------------- pretrained ie model --------------------- #')
    train_ie_model(config, model, best_dev, train_set, dev_set, test_set, config.pretrained_epoch , state, optimizer, schedule, best_role_model, dev_result_file, test_result_file, log_file)

    print('# ---------------------- start iterative stgg --------------------- #')
    
    firt_score_epoch = True
    for epoch in range(config.pretrained_epoch, config.max_epoch, 1):
        print('Epoch: {}'.format(epoch))

        if epoch % config.stgg_every_epoch == 0:
            if firt_score_epoch:
                firt_score_epoch = False
                print('# ---------------- pretrained score model --------------------- #')
                train_score_model(score_config, score_model, model, score_best_dev, score_train_set, score_dev_set, score_test_set, score_config.pretrained_epoch, score_state, score_optimizer, score_schedule, score_best_role_model, score_dev_result_file, score_test_result_file, score_log_file)
            else:
                print('# ---------------- adapt score model to ie model --------------------- #')
                train_score_model(score_config, score_model, model, score_best_dev, score_train_set, score_dev_set, score_test_set, score_config.score_every_stgg, score_state, score_optimizer, score_schedule, score_best_role_model, score_dev_result_file, score_test_result_file, score_log_file)
        
        print('# ---------------- stgg train ie model --------------------- #')
        train_stgg(config, model, score_model, best_dev, train_set, aux_train_set, dev_set, test_set, 1, state, optimizer, schedule, best_role_model, dev_result_file, test_result_file, log_file)  
        # else:
        #     train_ie_model(config, model, best_dev, train_set, dev_set, test_set, 1, state, optimizer, schedule, best_role_model, dev_result_file, test_result_file, log_file)

    best_score_by_task(score_log_file, 'role')
    best_score_by_task(log_file, 'role')


if __name__ == "__main__":
    # ----------------------------- parse args ------------------------------------#
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default='config/example.json')
    parser.add_argument('-sc', '--score_config', default='config/example.json')
    parser.add_argument('-n', '--name', default='temp')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-s', '--seed', type=int, default=43)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-blr', '--bert_learning_rate', type=float, default=1e-5)
    parser.add_argument('-m', '--mix_ratio', type=float, default=1.0)
    parser.add_argument('-a', '--aux2train_ratio', type=int, default=10)
    parser.add_argument('-f', '--feedback', default=False, action='store_true')
    parser.add_argument('-tr', '--use_trigger_score', default=False, action='store_true')
    parser.add_argument('-trw', '--trigger_weight', type=float, default=1.0)
    parser.add_argument('-ft', '--further_train', default=False, action='store_true')
    # parser.add_argument('-i', '--exp_idx', default='2')
    parser.add_argument('-st', '--s_threshold', type=float, default=0.9)
    parser.add_argument('-ie', '--ie_model', default='oneie')
    args = parser.parse_args()

    # 
    print("Using seed {}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.ie_model == 'oneie':
        print('###### Using OneIE model as event extraction model ######')
        from baseline_model import OneIE as IEmodel
    elif args.ie_model == 'amrie':
        raise NotImplementedError
    sys.exit(main(args))
