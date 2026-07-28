"""Our scorer is adapted from: https://github.com/dwadden/dygiepp"""

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def convert_arguments(triggers, entities, roles):
    args = set()
    for trigger_idx, entity_idx, role in roles:
        arg_start, arg_end, _ = entities[entity_idx]
        trigger_label = triggers[trigger_idx][-1]
        args.add((arg_start, arg_end, trigger_label, role))
    return args


def score_graphs(gold_graphs, pred_graphs, vocabs,
                 relation_directional=False):
    for graph in pred_graphs:
        graph.clean_role()
    event_type_stoi = vocabs['event_type']
    role_type_stoi = vocabs['role_type']
    # event_type_itos = {event_idx:event for event, event_idx in event_type_stoi.items()}
    # role_type_stoi = {role_idx:role for role, role_idx in role_type_stoi.items()}
    per_event_num = {event_idx:[0,0,0,0,event_type] for event_type, event_idx in event_type_stoi.items()}
    per_role_num = {role_idx:[0,0,0,0,role_type] for role_type, role_idx in role_type_stoi.items()}

    gold_arg_num = pred_arg_num = arg_idn_num = arg_class_num = 0
    gold_trigger_num = pred_trigger_num = trigger_idn_num = trigger_class_num = 0
    gold_ent_num = pred_ent_num = ent_match_num = 0
    gold_rel_num = pred_rel_num = rel_match_num = 0
    gold_men_num = pred_men_num = men_match_num = 0

    for gold_graph, pred_graph in zip(gold_graphs, pred_graphs):
        # Entity
        gold_entities = gold_graph.entities
        pred_entities = pred_graph.entities
        gold_ent_num += len(gold_entities)
        pred_ent_num += len(pred_entities)
        ent_match_num += len([entity for entity in pred_entities
                              if entity in gold_entities])

        # Mention
        gold_mentions = gold_graph.mentions
        pred_mentions = pred_graph.mentions
        gold_men_num += len(gold_mentions)
        pred_men_num += len(pred_mentions)
        men_match_num += len([mention for mention in pred_mentions
                              if mention in gold_mentions])

        # Relation
        gold_relations = gold_graph.relations
        pred_relations = pred_graph.relations
        gold_rel_num += len(gold_relations)
        pred_rel_num += len(pred_relations)
        for arg1, arg2, rel_type in pred_relations:
            arg1_start, arg1_end, _ = pred_entities[arg1]
            arg2_start, arg2_end, _ = pred_entities[arg2]
            for arg1_gold, arg2_gold, rel_type_gold in gold_relations:
                arg1_start_gold, arg1_end_gold, _ = gold_entities[arg1_gold]
                arg2_start_gold, arg2_end_gold, _ = gold_entities[arg2_gold]
                if relation_directional:
                    if (arg1_start == arg1_start_gold and
                        arg1_end == arg1_end_gold and
                        arg2_start == arg2_start_gold and
                        arg2_end == arg2_end_gold
                    ) and rel_type == rel_type_gold:
                        rel_match_num += 1
                        break
                else:
                    if ((arg1_start == arg1_start_gold and
                            arg1_end == arg1_end_gold and
                            arg2_start == arg2_start_gold and
                            arg2_end == arg2_end_gold) or (
                        arg1_start == arg2_start_gold and
                        arg1_end == arg2_end_gold and
                        arg2_start == arg1_start_gold and
                        arg2_end == arg1_end_gold
                    )) and rel_type == rel_type_gold:
                        rel_match_num += 1
                        break

        # Trigger
        gold_triggers = gold_graph.triggers
        pred_triggers = pred_graph.triggers
        gold_trigger_num += len(gold_triggers)
        pred_trigger_num += len(pred_triggers)
        for trg_start, trg_end, event_type in gold_triggers:
            per_event_num[event_type][0]+=1
        for trg_start, trg_end, event_type in pred_triggers:
            per_event_num[event_type][1]+=1
            matched = [item for item in gold_triggers
                       if item[0] == trg_start and item[1] == trg_end]
            assert len(matched) <=1
            if matched:
                trigger_idn_num += 1
                for item in matched:
                    per_event_num[item[2]][2]+=1
                if matched[0][-1] == event_type:
                    per_event_num[event_type][3]+=1
                    trigger_class_num += 1

        # Argument
        gold_args = convert_arguments(gold_triggers, gold_entities,
                                      gold_graph.roles)
        pred_args = convert_arguments(pred_triggers, pred_entities,
                                      pred_graph.roles)
        gold_arg_num += len(gold_args)
        pred_arg_num += len(pred_args)
        for gold_arg in gold_args:
            arg_start, arg_end, event_type, role = gold_arg
            per_role_num[role][0]+=1
        for pred_arg in pred_args:
            arg_start, arg_end, event_type, role = pred_arg
            per_role_num[role][1]+=1
            gold_idn = {item for item in gold_args
                        if item[0] == arg_start and item[1] == arg_end
                        and item[2] == event_type}
            if not len(gold_idn) <=1:
                print(gold_idn)
            if gold_idn:
                arg_idn_num += 1
                for item in gold_idn:
                    per_role_num[item[3]][2]+=1
                gold_class = {item for item in gold_idn if item[-1] == role}
                assert len(gold_class) <= 1
                if gold_class:
                    per_role_num[role][3]+=1
                    arg_class_num += 1
    
    entity_prec, entity_rec, entity_f = compute_f1(
        pred_ent_num, gold_ent_num, ent_match_num)
    mention_prec, mention_rec, mention_f = compute_f1(
        pred_men_num, gold_men_num, men_match_num)
    trigger_id_prec, trigger_id_rec, trigger_id_f = compute_f1(
        pred_trigger_num, gold_trigger_num, trigger_idn_num)
    trigger_prec, trigger_rec, trigger_f = compute_f1(
        pred_trigger_num, gold_trigger_num, trigger_class_num)
    relation_prec, relation_rec, relation_f = compute_f1(
        pred_rel_num, gold_rel_num, rel_match_num)
    role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_class_num)

    print('Entity: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        entity_prec * 100.0, entity_rec * 100.0, entity_f * 100.0))
    # print('Mention: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
    #     mention_prec * 100.0, mention_rec * 100.0, mention_f * 100.0))
    print('Trigger identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        trigger_id_prec * 100.0, trigger_id_rec * 100.0, trigger_id_f * 100.0))
    print('Trigger: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        trigger_prec * 100.0, trigger_rec * 100.0, trigger_f * 100.0))
    print('Relation: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        relation_prec * 100.0, relation_rec * 100.0, relation_f * 100.0))
    print('Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    print('Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_prec * 100.0, role_rec * 100.0, role_f * 100.0))

    scores = {
        'entity': {'prec': entity_prec, 'rec': entity_rec, 'f': entity_f},
        'trigger': {'prec': trigger_prec, 'rec': trigger_rec, 'f': trigger_f},
        'trigger_id': {'prec': trigger_id_prec, 'rec': trigger_id_rec,
                       'f': trigger_id_f},
        'role': {'prec': role_prec, 'rec': role_rec, 'f': role_f},
        'role_id': {'prec': role_id_prec, 'rec': role_id_rec, 'f': role_id_f}
    }

    pre_event_score = {}
    count_gold = 0
    count_pred = 0
    for per_event_idx in range(len(per_event_num)):
        per_event = per_event_num[per_event_idx]
        #  for _, per_event in per_event_num.items():
        gold_num, pred_num, id_num, class_num, event_type = per_event
        if event_type == 'O':
            continue
        prec, rec, f1 = compute_f1(pred_num, gold_num, class_num)
        id_prec, id_rec, id_f1 = compute_f1(pred_num, gold_num, id_num)
        pre_event_score[event_type] = {'prec': round(prec* 100.0,2), 'rec': round(rec* 100.0,2), 'f': round(f1* 100.0,2)}
        pre_event_score[event_type+'_id'] = {'prec': round(id_prec* 100.0,2), 'rec': round(id_rec* 100.0,2), 'f': round(id_f1* 100.0,2)}
        count_gold+=gold_num
        count_pred+=pred_num
        print('{}: P: {}, R: {}, F: {}, Gold: {}, Pred: {}, Match: {}'.format(event_type, round(prec* 100.0,2), round(rec* 100.0,2), round(f1* 100.0,2), gold_num,pred_num,class_num))
    assert gold_trigger_num == count_gold
    assert pred_trigger_num == count_pred
    pre_event_score['gold_trigger_num'] = gold_trigger_num
    pre_event_score['pred_trigger_num'] = pred_trigger_num
    print('gold event num: ',gold_trigger_num)
    print('pred event num: ',pred_trigger_num)

    pre_role_score = {}
    count_gold = 0
    count_pred = 0
    for per_role_idx in range(len(per_role_num)):
        per_role = per_role_num[per_role_idx]
        # for _, per_role in per_role_num.items():
        gold_num, pred_num, id_num, class_num, role_type = per_role
        if per_role == 'O':
            continue
        prec, rec, f1 = compute_f1(pred_num, gold_num, class_num)
        id_prec, id_rec, id_f1 = compute_f1(pred_num, gold_num, id_num)
        pre_role_score[role_type] = {'prec': round(prec* 100.0,2), 'rec': round(rec* 100.0,2), 'f': round(f1* 100.0,2)}
        pre_role_score[role_type+'_id'] = {'prec': round(id_prec* 100.0,2), 'rec': round(id_rec* 100.0,2), 'f': round(id_f1* 100.0,2)}
        count_gold+=gold_num
        count_pred+=pred_num
        print('{}: P: {}, R: {}, F: {}, Gold: {}, Pred: {}, Match: {}'.format(role_type, round(prec* 100.0,2), round(rec* 100.0,2), round(f1* 100.0,2), gold_num,pred_num,class_num))
    assert gold_arg_num == count_gold
    assert pred_arg_num == count_pred
    pre_event_score['gold_role_num'] = gold_arg_num
    pre_event_score['pred_role_num'] = pred_arg_num
    print('gold role num: ',gold_arg_num)
    print('pred role num: ',pred_arg_num)




    return scores, pre_event_score, pre_role_score

def score_coref(gold_graphs, pred_graphs):
    pass