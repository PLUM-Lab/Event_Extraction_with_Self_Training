import copy
import itertools
import json
import torch
import dgl
import random
import math

from torch.utils.data import Dataset
from collections import Counter, namedtuple, defaultdict

from graph import Graph
from score_util import read_ltf, read_txt, read_json, read_json_single, categorize_amr_role, get_graph, get_ie2amr_role
from tqdm import tqdm

instance_fields = [
    'sent_id', 'tokens', 'pieces', 'piece_idxs', 'token_lens', 'attention_mask',
    'entity_label_idxs', 'trigger_label_idxs',
    'entity_type_idxs', 'event_type_idxs',
    'relation_type_idxs', 'role_type_idxs',
    'mention_type_idxs',
    'graph', 'entity_num', 'trigger_num',
    'ie2amr_edge_map', 'amr_bidir_graph', 'amr_edge2type',
    'amr', 'align', 'exist'
]
instance_ldc_eval_fields = [
    'sent_id', 'tokens', 'token_ids', 'pieces', 'piece_idxs',
    'token_lens', 'attention_mask'
]
batch_fields = [
    'sent_ids', 'tokens', 'pieces', 'piece_idxs', 'token_lens', 'attention_masks',
    'entity_label_idxs', 'trigger_label_idxs',
    'entity_type_idxs', 'event_type_idxs', 'mention_type_idxs',
    'relation_type_idxs', 'role_type_idxs',
    'graphs', 'token_nums',
    'ie2amr_edge_map','amr_bidir_graph', 'amr_edge2type',
    'amr', 'align', 'exist'
]
batch_ldc_eval_fields = [
    'sent_ids', 'token_ids', 'tokens', 'piece_idxs', 'token_lens', 'attention_masks', 'token_nums'
]
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))
InstanceLdcEval = namedtuple('InstanceLdcEval',
                             field_names=instance_ldc_eval_fields,
                             defaults=[None] * len(instance_ldc_eval_fields))
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))
BatchLdcEval = namedtuple('BatchLdcEval',
                          field_names=batch_ldc_eval_fields,
                          defaults=[None] * len(batch_ldc_eval_fields))
BatchEval = namedtuple('BatchEval', field_names=['sent_ids', 'piece_idxs',
                                                 'tokens', 'attention_masks',
                                                 'token_lens', 'token_nums'])


def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    for entity in entities:
        start, end = entity['start'], entity['end']
        for i in range(start, end):
            if tokens[i]:
                id_map[entity['id']] = tokens[i]
                continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity['id']
    return entities_, id_map


def get_entity_labels(entities, token_num):
    """Convert entity mentions in a sentence to an entity label sequence with
    the length of token_num
    CHECKED
    :param entities (list): a list of entity mentions.
    :param token_num (int): the number of tokens.
    :return:a sequence of BIO format labels.
    """
    labels = ['O'] * token_num
    for entity in entities:
        start, end = entity['start'], entity['end']
        entity_type = entity['entity_type']
        if any([labels[i] != 'O' for i in range(start, end)]):
            continue
        labels[start] = 'B-{}'.format(entity_type)
        for i in range(start + 1, end):
            labels[i] = 'I-{}'.format(entity_type)
    return labels


def get_trigger_labels(events, token_num):
    """Convert event mentions in a sentence to a trigger label sequence with the
    length of token_num.
    :param events (list): a list of event mentions.
    :param token_num (int): the number of tokens.
    :return: a sequence of BIO format labels.
    """
    labels = ['O'] * token_num
    for event in events:
        trigger = event['trigger']
        start, end = trigger['start'], trigger['end']
        event_type = event['event_type']
        labels[start] = 'B-{}'.format(event_type)
        for i in range(start + 1, end):
            labels[i] = 'I-{}'.format(event_type)
    return labels


def get_relation_types(entities, relations, id_map, directional=False,
                       symmetric=None):
    """Get relation type labels among all entities in a sentence.
    :param entities (list): a list of entity mentions.
    :param relations (list): a list of relation mentions.
    :param id_map (dict): a dict of entity ID mapping.
    :param symmetric (set): a set of symmetric relation types.
    :return: a matrix of relation type labels.
    
    "relation_mentions": [{"id": "APW_ENG_20030520.0757-R7-1", "relation_type": "PART-WHOLE", "relation_subtype": "PART-WHOLE:Geographical", "arguments": [{"entity_id": "APW_ENG_20030520.0757-E44-59", "text": "capital", "role": "Arg-1"}, {"entity_id": "APW_ENG_20030520.0757-E14-60", "text": "Saudi", "role": "Arg-2"}]}]
    """

    entity_num = len(entities)
    labels = [['O'] * entity_num for _ in range(entity_num)]
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    for relation in relations:
        entity_1 = entity_2 = -1
        for arg in relation['arguments']:
            entity_id = arg['entity_id']
            entity_id = id_map.get(entity_id, entity_id)
            if arg['role'] == 'Arg-1':
                entity_1 = entity_idxs[entity_id]
            elif arg['role'] == 'Arg-2':
                entity_2 = entity_idxs[entity_id]
        if entity_1 == -1 or entity_2 == -1:
            continue
        labels[entity_1][entity_2] = relation['relation_type']
        if not directional:
            labels[entity_2][entity_1] = relation['relation_type']
        if symmetric and relation['relation_type'] in symmetric:
            labels[entity_2][entity_1] = relation['relation_type']
    return labels


def get_relation_list(entities, relations, id_map, vocab, directional=False,
                      symmetric=None):
    """Get the relation list (used for Graph objects)
    :param entities (list): a list of entity mentions.
    :param relations (list): a list of relation mentions.
    :param id_map (dict): a dict of entity ID mapping.
    :param vocab (dict): a dict of label to label index mapping.
    """
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(entities))]
    relation_list = []
    for relation in relations:
        arg_1 = arg_2 = None
        for arg in relation['arguments']:
            if arg['role'] == 'Arg-1':
                arg_1 = entity_idxs[id_map.get(
                    arg['entity_id'], arg['entity_id'])]
            elif arg['role'] == 'Arg-2':
                arg_2 = entity_idxs[id_map.get(
                    arg['entity_id'], arg['entity_id'])]
        if arg_1 is None or arg_2 is None:
            continue
        relation_type = relation['relation_type']
        if (not directional and arg_1 > arg_2) or \
                (directional and symmetric and relation_type in symmetric and arg_1 > arg_2):
            arg_1, arg_2 = arg_2, arg_1
        if visited[arg_1][arg_2] == 0:
            relation_list.append((arg_1, arg_2, vocab[relation_type]))
            visited[arg_1][arg_2] = 1

    relation_list.sort(key=lambda x: (x[0], x[1]))
    return relation_list


def get_role_types(entities, events, id_map):
    labels = [['O'] * len(entities) for _ in range(len(events))]
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    for event_idx, event in enumerate(events):
        for arg in event['arguments']:
            entity_id = arg['entity_id']
            entity_id = id_map.get(entity_id, entity_id)
            entity_idx = entity_idxs[entity_id]
            # if labels[event_idx][entity_idx] != 'O':
            #     print('Conflict argument role {} {} {}'.format(event['trigger']['text'], arg['text'], arg['role']))
            labels[event_idx][entity_idx] = arg['role']
    return labels


def get_role_list(entities, events, id_map, vocab):
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(events))]
    role_list = []
    for i, event in enumerate(events):
        for arg in event['arguments']:
            entity_idx = entity_idxs[id_map.get(
                arg['entity_id'], arg['entity_id'])]
            if visited[i][entity_idx] == 0:
                role_list.append((i, entity_idx, vocab[arg['role']]))
                visited[i][entity_idx] = 1
    role_list.sort(key=lambda x: (x[0], x[1]))
    return role_list


def get_coref_types(entities):
    entity_num = len(entities)
    labels = [['O'] * entity_num for _ in range(entity_num)]
    clusters = defaultdict(list)
    for i, entity in enumerate(entities):
        entity_id = entity['entity_id']
        cluster_id = entity_id[:entity_id.rfind('-')]
        clusters[cluster_id].append(i)
    for _, entities in clusters.items():
        for i, j in itertools.combinations(entities, 2):
            labels[i][j] = 'COREF'
            labels[j][i] = 'COREF'
    return labels


def get_coref_list(entities, vocab):
    clusters = defaultdict(list)
    coref_list = []
    for i, entity in enumerate(entities):
        entity_id = entity['entity_id']
        cluster_id = entity_id[:entity_id.rfind('-')]
        clusters[cluster_id].append(i)
    for _, entities in clusters.items():
        for i, j in itertools.combinations(entities, 2):
            if i < j:
                coref_list.append((i, j, vocab['COREF']))
            else:
                coref_list.append((j, i, vocab['COREF']))
    coref_list.sort(key=lambda x: (x[0], x[1]))
    return coref_list


def merge_coref_relation_lists(coref_list, relation_list, entity_num):
    visited = [[0] * entity_num for _ in range(entity_num)]
    merge_list = []
    for i, j, l in coref_list:
        visited[i][j] = 1
        visited[j][i] = 1
        merge_list.append((i, j, l))
    for i, j, l in relation_list:
        assert visited[i][j] == 0 and visited[j][i] == 0
        merge_list.append((i, j, l))
    merge_list.sort(key=lambda x: (x[0], x[1]))


def merge_coref_relation_types(coref_types, relation_types):
    entity_num = len(coref_types)
    labels = copy.deepcopy(coref_types)
    for i in range(entity_num):
        for j in range(entity_num):
            label = relation_types[i][j]
            if label != 0:
                assert labels[i][j] == 0
                labels[i][j] = label
    return labels


class IEGraphDataset(Dataset):
    def __init__(self, path, amr_path, config, max_length=128, gpu=False, ignore_title=False,
                 relation_mask_self=True, relation_directional=False,
                 coref=False, symmetric_relations=None, train=True):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        :param ignore_title (bool): Ignore sentences that are titles (default=False).
        """
        self.path = path
        self.train = train
        self.config = config
        self.data = []
        self.gpu = gpu
        self.max_length = max_length
        self.ignore_title = ignore_title
        self.relation_mask_self = relation_mask_self
        self.relation_directional = relation_directional
        self.coref = coref
        self.amr_path = amr_path
        if symmetric_relations is None:
            self.symmetric_relations = set()
        else:
            self.symmetric_relations = symmetric_relations

        self.amr_edge_type_set = set()
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def entity_type_set(self):
        type_set = set()
        for inst in self.data:
            for entity in inst['entity_mentions']:
                type_set.add(entity['entity_type'])
        return type_set

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def relation_type_set(self):
        type_set = set()
        for inst in self.data:
            for relation in inst['relation_mentions']:
                type_set.add(relation['relation_type'])
        return type_set

    @property
    def role_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['role'])
        return type_set

    def process_amr(self, raw_amr):
        amr_split_list = raw_amr.split('\n')
        node_to_idx, node_to_offset, node_to_offset_whole = {}, {}, {}
        node_num = 0
        # first to fill in the node list
        for line in amr_split_list:
            if line.startswith('# ::node'):
                node_split = line.split('\t') # ['# ::node','2','multiple','1-2']
                if len(node_split) != 4:
                    # check if the alignment text spans exist
                    continue
                else:
                    align_span = node_split[3].split('-') #['1','2']
                    if not align_span[0].isdigit():
                        continue
                    head_word_idx = int(align_span[1]) - 1 # 1
                    try:
                        start = int(align_span[0]) # 1
                    except:
                        # print(amr_list[i])
                        raise ValueError
                    end = int(align_span[1]) #2
                    if (start, end) not in list(node_to_offset_whole.values()):
                        node_to_offset.update({node_split[1]: head_word_idx}) # {'node_id 2': end_idx-1}
                        node_to_offset_whole.update({node_split[1]: (start, end)}) # {'node_id 2': (start_idx, end_idx)}
                        node_to_idx.update({node_split[1]: node_num}) # {'node_id 2': node_index}
                        node_num += 1
            else:
                continue

        # node_idx_list.append(node_to_idx)
        # change node_id2offset to idx2offset
        node_idx_to_offset = {}
        for key in node_to_idx.keys():
            node_idx_to_offset.update({node_to_idx[key]: node_to_offset[key]}) # {node_index: end_idx-1}

        node_idx_to_offset_whole = {}
        for key in node_to_idx.keys():
            node_idx_to_offset_whole.update({node_to_idx[key]: node_to_offset_whole[key]}) # {node_index: (start_idx, end_idx)}

        edge_type_dict = {}

        # add amr prior order
        for line in amr_split_list:
            if line.startswith('# ::root'):
                root_split = line.split('\t')
                root = root_split[1]

        prior_dict = {root:[]}

        start_list = []
        end_list = []

        for line in amr_split_list:
            if line.startswith('# ::edge'):
                edge_split = line.split('\t') # ['# ::edge','multiple','op1','1000000','2','3']
                amr_edge_type = edge_split[2] #'op1'
                edge_start = edge_split[4] # '2' this is node id
                edge_end = edge_split[5] # '3' this is node id
                # check if the start and end nodes exist
                if (edge_start in node_to_idx) and (edge_end in node_to_idx):
                    # check if the edge type is "ARGx-of", if so, reverse the direction of the edge
                    if amr_edge_type.endswith("-of"):
                        edge_start, edge_end = edge_end, edge_start
                        amr_edge_type = amr_edge_type[0:-3]
                    amr_edge_type = categorize_amr_role(amr_edge_type)
                    self.amr_edge_type_set.add(amr_edge_type)
                    start_idx = node_to_idx[edge_start] # node_index 
                    end_idx = node_to_idx[edge_end]
                    edge_type_dict.update({(start_idx, end_idx): amr_edge_type}) #{(start_node_idx,end_node_idx): edge type}
                else:
                    continue
                if edge_end != root and (not ((edge_start in end_list) and (edge_end in start_list))):
                    start_list.append(edge_start)
                    end_list.append(edge_end)
                if edge_start not in prior_dict:
                    prior_dict.update({edge_start:[edge_end]})
                else:
                    prior_dict[edge_start].append(edge_end)
            else:
                continue
        final_order_list = []
        # output orders
        candidate_nodes = node_to_idx.copy()
        while len(candidate_nodes) != 0:
            current_level_nodes = []
            for key in candidate_nodes:
                if key not in end_list:
                    final_order_list.append(candidate_nodes[key])
                    current_level_nodes.append(key)
            # Remove current level nodes from the dictionary
            for node in current_level_nodes:
                candidate_nodes.pop(node)

            # deleting from start lists the current level nodes
            for node in current_level_nodes:
                indices_list = [i for i, x in enumerate(start_list) if x == node]
                start_list = [x for x in start_list if x != node]
                new_end_list = []
                for i in range(len(end_list)):
                    if i not in indices_list:
                        new_end_list.append(end_list[i])
                end_list = new_end_list
        return {'node_to_idx':node_to_idx, 'node_idx_to_offset':node_idx_to_offset,'node_idx_to_offset_whole':node_idx_to_offset_whole, 'edge_type_dict':edge_type_dict,'decode_order':final_order_list}

    def load_data(self):
        """Load data from file."""
        overlength_num = title_num = problem_num = 0
        amr_list = torch.load(self.amr_path)
        with open(self.path, 'r', encoding='utf-8') as r:
            for i, line in enumerate(r):
                inst = json.loads(line)
                inst_len = len(inst['pieces'])
                is_title = inst['sent_id'].endswith('-3') \
                    and inst['tokens'][-1] != '.' \
                    and len(inst['entity_mentions']) == 0
                if self.ignore_title and is_title:
                    title_num += 1
                    continue
                if self.max_length != -1 and inst_len > self.max_length - 2:
                    overlength_num += 1
                    continue
                if not sum(inst['token_lens']) == inst_len:
                    problem_num += 1
                    continue
                if not '# ::root' in amr_list[i]:
                    problem_num += 1
                    continue
                amr_info = self.process_amr(amr_list[i])
                inst.update({"amr_info": amr_info})
                self.data.append(inst)
        if overlength_num:
            print('Discarded {} overlength instances'.format(overlength_num))
        if title_num:
            print('Discarded {} titles'.format(title_num))
        if problem_num:
            print('Discarded {} problematic instances'.format(problem_num))
        print('Loaded {} instances from {}'.format(len(self), self.path))
    
    def create_amr_graph(self, ie_graph, amr_info, vocabs, tokens):
        amr_graph, align_list, exist_list, amr_bidir_graph, amr_edge2type = get_graph(amr_info, vocabs, tokens)
        ie2amr_node_map, ie2amr_edge_map = get_ie2amr_role(amr_graph, align_list, exist_list, amr_bidir_graph, amr_edge2type, ie_graph, vocabs, tokens)
        amr_graph.ndata['node_idx'] = amr_graph.ndata["token_span"].new_tensor(torch.arange(amr_graph.num_nodes()),dtype=torch.long).unsqueeze(-1)
        return amr_graph, align_list, exist_list, ie2amr_node_map, ie2amr_edge_map, amr_bidir_graph, amr_edge2type


    def numberize(self, tokenizer, vocabs):
        entity_type_stoi = vocabs['entity_type']
        event_type_stoi = vocabs['event_type']
        relation_type_stoi = vocabs['relation_type']
        role_type_stoi = vocabs['role_type']
        mention_type_stoi = vocabs['mention_type']
        entity_label_stoi = vocabs['entity_label']
        trigger_label_stoi = vocabs['trigger_label']

        data = []
        ignored = 0
        total_role = 0
        total_event = 0
        total_entity = 0
        for inst in self.data:
            tokens = inst['tokens']
            pieces = inst['pieces']
            sent_id = inst['sent_id']
            entities = inst['entity_mentions']
            entities, entity_id_map = remove_overlap_entities(entities)
            entities.sort(key=lambda x: x['start'])
            events = inst['event_mentions']
            events.sort(key=lambda x: x['trigger']['start'])
            relations = inst['relation_mentions']
            token_num = len(tokens)
            token_lens = inst['token_lens']
            
            # Pad word pieces with special tokens
            piece_idxs = tokenizer.encode(pieces,
                                          add_special_tokens=True,
                                          max_length=self.max_length,
                                          truncation=True)
            pad_num = self.max_length - len(piece_idxs)
            attn_mask = [1] * len(piece_idxs) + [0] * pad_num
            piece_idxs = piece_idxs + [0] * pad_num

            # Entity
            entity_labels = get_entity_labels(entities, token_num)
            entity_label_idxs = [entity_label_stoi[l] for l in entity_labels]
            entity_types = [e['entity_type'] for e in entities]
            entity_type_idxs = [entity_type_stoi[l] for l in entity_types]
            entity_list = [(e['start'], e['end'], entity_type_stoi[e['entity_type']])
                           for e in entities]
            entity_num = len(entity_list)

            # don't use mention
            mention_types = [e['mention_type'] for e in entities]
            mention_type_idxs = [mention_type_stoi[l] for l in mention_types]
            mention_list = [(i, j, l) for (i, j, k), l
                            in zip(entity_list, mention_type_idxs)]

            # Trigger
            trigger_labels = get_trigger_labels(events, token_num)
            trigger_label_idxs = [trigger_label_stoi[l]
                                  for l in trigger_labels]
            event_types = [e['event_type'] for e in events]
            event_type_idxs = [event_type_stoi[l] for l in event_types]
            trigger_list = [(e['trigger']['start'], e['trigger']['end'],
                             event_type_stoi[e['event_type']])
                            for e in events]
            trigger_num = len(trigger_list)


            # Relation
            relation_types = get_relation_types(entities, relations,
                                                entity_id_map,
                                                directional=self.relation_directional,
                                                symmetric=self.symmetric_relations)
            relation_type_idxs = [[relation_type_stoi[l] for l in ls]
                                  for ls in relation_types]
            if self.relation_mask_self:
                for i in range(len(relation_type_idxs)):
                    relation_type_idxs[i][i] = -100
            relation_list = get_relation_list(entities, relations,
                                              entity_id_map, relation_type_stoi,
                                              directional=self.relation_directional,
                                              symmetric=self.symmetric_relations)
            relation_num = len(relation_list)

            # Argument role
            role_types = get_role_types(entities, events, entity_id_map)
            role_type_idxs = [[role_type_stoi[l] for l in ls]
                              for ls in role_types]
            role_list = get_role_list(entities, events,
                                      entity_id_map, role_type_stoi)
            role_num = len(role_list)

            # ignore instances that don't contain any node
            if self.train:
                if trigger_num==0 or entity_num==0:
                    ignored+=1
                    continue
            
            total_event+=trigger_num
            total_role+=role_num
            total_entity+=entity_num

            # Graph
            graph = Graph(
                entities=entity_list,
                triggers=trigger_list,
                relations=relation_list,
                roles=role_list,
                mentions=mention_list,
                vocabs=vocabs,
            )

            # create graphs
            amr_graph, align_list, exist_list, ie2amr_node_map, ie2amr_edge_map, amr_bidir_graph, amr_edge2type = self.create_amr_graph(graph, inst['amr_info'], vocabs, tokens)
            instance = Instance(
                sent_id=sent_id,
                tokens=tokens,
                pieces=pieces,
                piece_idxs=piece_idxs,
                token_lens=token_lens,
                attention_mask=attn_mask,
                entity_label_idxs=entity_label_idxs,
                trigger_label_idxs=trigger_label_idxs,
                entity_type_idxs=entity_type_idxs,
                event_type_idxs=event_type_idxs,
                relation_type_idxs=relation_type_idxs,
                mention_type_idxs=mention_type_idxs,
                role_type_idxs=role_type_idxs,
                graph=graph,
                entity_num=len(entities),
                trigger_num=len(events),
                # ie2amr_map=ie2amr_node_map,
                ie2amr_edge_map=ie2amr_edge_map, # this now map ie edge : amr node path, amr node rel, ie edge type 
                amr=amr_graph,
                align=align_list,
                exist=exist_list,
                amr_bidir_graph=amr_bidir_graph,
                amr_edge2type=amr_edge2type
            )
            data.append(instance)
        self.data = data
        print('number of ignored instances: ',ignored)
        print('number of event: ',total_event)
        print('number of entity: ',total_entity)
        print('number of event arguments: ',total_role)

        
    def collate_fn(self, batch):
        # print(batch)
        batch_piece_idxs = []
        batch_tokens = []
        batch_entity_labels, batch_trigger_labels = [], []
        batch_entity_types, batch_event_types = [], []
        batch_relation_types, batch_role_types = [], []
        batch_mention_types = []
        batch_graphs = []
        batch_token_lens = []
        batch_attention_masks = []

        sent_ids = [inst.sent_id for inst in batch]
        token_nums = [len(inst.tokens) for inst in batch]
        max_token_num = max(token_nums)

        max_entity_num = max([inst.entity_num for inst in batch] + [1])
        max_trigger_num = max([inst.trigger_num for inst in batch] + [1])

        amrs = []
        aligns = []
        exists = []
        amr_bidir_graphs = []
        amr_edge2types = []
        batch_ie2amr_map = []
        for inst in batch:
            token_num = len(inst.tokens)
            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_mask)
            batch_token_lens.append(inst.token_lens)
            batch_graphs.append(inst.graph)
            batch_tokens.append(inst.tokens)
            # for identification
            batch_entity_labels.append(inst.entity_label_idxs +
                                       [0] * (max_token_num - token_num))
            batch_trigger_labels.append(inst.trigger_label_idxs +
                                        [0] * (max_token_num - token_num))
            # for classification
            batch_entity_types.extend(inst.entity_type_idxs +
                                      [-100] * (max_entity_num - inst.entity_num))
            batch_event_types.extend(inst.event_type_idxs +
                                     [-100] * (max_trigger_num - inst.trigger_num))
            batch_mention_types.extend(inst.mention_type_idxs +
                                       [-100] * (max_entity_num - inst.entity_num))
            for l in inst.relation_type_idxs:
                batch_relation_types.extend(
                    l + [-100] * (max_entity_num - inst.entity_num))
            batch_relation_types.extend(
                [-100] * max_entity_num * (max_entity_num - inst.entity_num))
            for l in inst.role_type_idxs:
                batch_role_types.extend(
                    l + [-100] * (max_entity_num - inst.entity_num))
            batch_role_types.extend(
                [-100] * max_entity_num * (max_trigger_num - inst.trigger_num))
            # amr
            amrs.append(inst.amr.to(self.config.gpu_device))
            aligns.append(inst.align)
            exists.append(inst.exist)
            amr_bidir_graphs.append(inst.amr_bidir_graph)
            amr_edge2types.append(inst.amr_edge2type)
            batch_ie2amr_map.append(inst.ie2amr_edge_map)
            
        if self.gpu:
            batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
            batch_attention_masks = torch.cuda.FloatTensor(
                batch_attention_masks)
            batch_entity_labels = torch.cuda.LongTensor(batch_entity_labels)
            batch_trigger_labels = torch.cuda.LongTensor(batch_trigger_labels)
            batch_entity_types = torch.cuda.LongTensor(batch_entity_types)
            batch_mention_types = torch.cuda.LongTensor(batch_mention_types)
            batch_event_types = torch.cuda.LongTensor(batch_event_types)
            batch_relation_types = torch.cuda.LongTensor(batch_relation_types)
            batch_role_types = torch.cuda.LongTensor(batch_role_types)
            token_nums = torch.cuda.LongTensor(token_nums)

        else:
            batch_piece_idxs = torch.LongTensor(batch_piece_idxs)
            batch_attention_masks = torch.FloatTensor(batch_attention_masks)
            batch_entity_labels = torch.LongTensor(batch_entity_labels)
            batch_trigger_labels = torch.LongTensor(batch_trigger_labels)
            batch_entity_types = torch.LongTensor(batch_entity_types)
            batch_mention_types = torch.LongTensor(batch_mention_types)
            batch_event_types = torch.LongTensor(batch_event_types)
            batch_relation_types = torch.LongTensor(batch_relation_types)
            batch_role_types = torch.LongTensor(batch_role_types)
            token_nums = torch.LongTensor(token_nums)

        return Batch(
            sent_ids=sent_ids,
            tokens=[inst.tokens for inst in batch],
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            entity_label_idxs=batch_entity_labels,
            trigger_label_idxs=batch_trigger_labels,
            entity_type_idxs=batch_entity_types,
            mention_type_idxs=batch_mention_types,
            event_type_idxs=batch_event_types,
            relation_type_idxs=batch_relation_types,
            role_type_idxs=batch_role_types,
            graphs=batch_graphs,
            token_nums=token_nums,
            ie2amr_edge_map=batch_ie2amr_map,
            amr=amrs,
            align=aligns,
            exist=exists,
            amr_bidir_graph=amr_bidir_graphs,
            amr_edge2type=amr_edge2types
        )

class AuxDataset(Dataset):
    def __init__(self, config, gpu=False, require_len=19190, train = True):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        :param ignore_title (bool): Ignore sentences that are titles (default=False).
        """
        self.config = config
        self.train = train
        self.aux2train_ratio = config.aux2train_ratio
        self.path = config.aux_train_file
        self.data = []
        self.gpu = gpu
        self.max_length = 128
        self.min_length = config.min_amr_length
        self.order_file = config.order_file
        self.load_data()
        self.require_len = require_len
        self.ordered_aux = config.ordered_aux

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def process_amr(self, inst):
        nodes, edges, final_order_list = inst['nodes'], inst['edges'], inst['decode_order']
        node_to_idx, node_to_offset, node_to_offset_whole = {}, {}, {}
        node_num = 0
        # first to fill in the node list
        for node in nodes:
            node_id = node['node_id']
            align_span = [node['start'],node['end']] #['1','2']
            head_word_idx = int(align_span[1]) - 1 # 1
            start = int(align_span[0])
            end = int(align_span[1]) #2
            if (start, end) not in list(node_to_offset_whole.values()):
                node_to_offset.update({node_id: head_word_idx}) # {'node_id 2': end_idx-1}
                node_to_offset_whole.update({node_id: (start, end)}) # {'node_id 2': (start_idx, end_idx)}
                node_to_idx.update({node_id: node_num}) # {'node_id 2': node_index}
                node_num += 1

        node_idx_to_offset = {}
        for key in node_to_idx.keys():
            node_idx_to_offset.update({node_to_idx[key]: node_to_offset[key]}) # {node_index: end_idx-1}

        node_idx_to_offset_whole = {}
        for key in node_to_idx.keys():
            node_idx_to_offset_whole.update({node_to_idx[key]: node_to_offset_whole[key]}) # {node_index: (start_idx, end_idx)}

        edge_type_dict = {}
        for edge, edge_type in edges.items():
            edge = edge.split('\t')
            amr_edge_type = edge_type #'op1'
            edge_start = edge[0] # '2' this is node id
            edge_end = edge[1] # '3' this is node id
            # check if the start and end nodes exist
            if (edge_start in node_to_idx) and (edge_end in node_to_idx):
                # check if the edge type is "ARGx-of", if so, reverse the direction of the edge
                if amr_edge_type.endswith("-of"):
                    edge_start, edge_end = edge_end, edge_start
                    amr_edge_type = amr_edge_type[0:-3]
                amr_edge_type = categorize_amr_role(amr_edge_type)
                # self.amr_edge_type_set.add(amr_edge_type)
                start_idx = node_to_idx[edge_start] # node_index 
                end_idx = node_to_idx[edge_end]
                edge_type_dict.update({(start_idx, end_idx): amr_edge_type})
            else:
                continue
        return {'node_to_idx':node_to_idx, 'node_idx_to_offset':node_idx_to_offset,'node_idx_to_offset_whole':node_idx_to_offset_whole, 'edge_type_dict':edge_type_dict,'decode_order':final_order_list}
    
    def load_data(self):
        """Load data from file."""
        overlength_num = 0
        underlength_num = 0
        self.problematic_num = 0
        if self.order_file == '':
            order_data = []
            with open(self.path,'r',encoding='utf-8') as fin:
                for i, line in enumerate(fin):
                    inst = json.loads(line)
                    order_data.append(inst)
        else:
            order_list = json.load(open(self.order_file,'r'))
            split_point = math.ceil(len(order_list)*0.75)
            if self.train:
                order_list = order_list[:split_point]
            else:
                order_list = order_list[split_point:]
            print('Total {} instances in ordered list of aux dataset'.format(len(order_list)))
            raw_data = {}
            # raw_data_list = []
            with open(self.path,'r',encoding='utf-8') as fin:
                for i, line in enumerate(fin):
                    inst = json.loads(line)
                    raw_data[inst['sent_id']] = inst
                    # raw_data_list.append(inst)
            order_data = [raw_data[id_[0]] for id_ in order_list]
            # if self.aux2train_ratio == -1:
            #     order_data = raw_data_list

        for inst in order_data:
            inst_len = len(inst['pieces'])
            if self.max_length != -1 and inst_len > self.max_length - 2:
                overlength_num += 1
                continue
            elif self.min_length != -1 and inst_len < self.min_length - 2:
                underlength_num += 1
                continue
            bad_inst = False
            for node in inst['nodes']:
                if node['end'] - node['start']<1:
                    self.problematic_num += 1
                    bad_inst = True
                    break
            if bad_inst:
                continue
            amr_info = self.process_amr(inst)
            inst.update({"amr_info": amr_info})
            self.data.append(inst)
        if overlength_num:
            print('Discarded {} overlength instances'.format(overlength_num))
        if underlength_num:
            print('Discarded {} underlength instances'.format(underlength_num))
        if self.problematic_num:
            print('Discarded {} problematic instances'.format(self.problematic_num))
        # print('Loaded {} instances from {}'.format(len(self), self.path))

    def numberize(self, tokenizer, vocabs):
        """Numberize word pieces, labels, etcs.
        :param tokenizer: Bert tokenizer.
        :param vocabs (dict): a dict of vocabularies.
        """
        data = []
        # random.shuffle(self.data)
        if self.aux2train_ratio == -1:
            print('Loaded {} instances from {}'.format(len(self), self.path))
            for inst in tqdm(self.data):
                tokens = inst['tokens']
                pieces = inst['pieces']
                sent_id = inst['sent_id']
                token_num = len(tokens)
                token_lens = inst['token_lens']
                amr_info = inst['amr_info']
                piece_idxs = tokenizer.encode(pieces,
                                            add_special_tokens=True,
                                            max_length=self.max_length,
                                            truncation=True)
                pad_num = self.max_length - len(piece_idxs)
                attn_mask = [1] * len(piece_idxs) + [0] * pad_num
                piece_idxs = piece_idxs + [0] * pad_num

                amr_graph, align_list, exist_list, amr_bidir_graph, amr_edge2type = get_graph(amr_info, vocabs, tokens)

                instance = Instance(
                    sent_id=sent_id,
                    tokens=tokens,
                    pieces=pieces,
                    piece_idxs=piece_idxs,
                    token_lens=token_lens,
                    attention_mask=attn_mask,
                    amr=amr_graph,
                    align=align_list,
                    exist=exist_list,
                    ie2amr_edge_map=None,
                    amr_bidir_graph=amr_bidir_graph,
                    amr_edge2type=amr_edge2type
                )
                data.append(instance)
        else:
            if self.aux2train_ratio > len(self.data):
                print('load all unlabeled instance')
            else:
                if self.ordered_aux:
                    self.data = self.data[:self.aux2train_ratio*1000]
                    random.shuffle(self.data)
                    print('use ordered aux data')
                else:
                    random.shuffle(self.data)
                    self.data = self.data[:self.aux2train_ratio*1000]
                    print('use unordered aux data')
            print('Loaded {} instances from {}'.format(len(self), self.path))
            # if len(self.data) < self.require_len:
            #     self.data = self.data * math.ceil(self.require_len/float(len(self.data)))
            for inst in tqdm(self.data):
                tokens = inst['tokens']
                pieces = inst['pieces']
                sent_id = inst['sent_id']
                token_num = len(tokens)
                token_lens = inst['token_lens']
                amr_info = inst['amr_info']
                piece_idxs = tokenizer.encode(pieces,
                                            add_special_tokens=True,
                                            max_length=self.max_length,
                                            truncation=True)
                pad_num = self.max_length - len(piece_idxs)
                attn_mask = [1] * len(piece_idxs) + [0] * pad_num
                piece_idxs = piece_idxs + [0] * pad_num

                amr_graph, align_list, exist_list, amr_bidir_graph, amr_edge2type = get_graph(amr_info, vocabs, tokens)

                instance = Instance(
                    sent_id=sent_id,
                    tokens=tokens,
                    pieces=pieces,
                    piece_idxs=piece_idxs,
                    token_lens=token_lens,
                    attention_mask=attn_mask,
                    amr=amr_graph,
                    align=align_list,
                    exist=exist_list,
                    ie2amr_edge_map=None,
                    amr_bidir_graph=amr_bidir_graph,
                    amr_edge2type=amr_edge2type
                )
                data.append(instance)
            # else:
                # for i in tqdm(range(self.require_len)):
                #     if i < len(self.data):
                #         inst = self.data[i]
                #     else:
                #         inst = self.data[random.randint(0,len(self.data)-1)]
                #     tokens = inst['tokens']
                #     pieces = inst['pieces']
                #     sent_id = inst['sent_id']
                #     token_num = len(tokens)
                #     token_lens = inst['token_lens']
                #     amr_info = inst['amr_info']
                #     piece_idxs = tokenizer.encode(pieces,
                #                                 add_special_tokens=True,
                #                                 max_length=self.max_length,
                #                                 truncation=True)
                #     pad_num = self.max_length - len(piece_idxs)
                #     attn_mask = [1] * len(piece_idxs) + [0] * pad_num
                #     piece_idxs = piece_idxs + [0] * pad_num

                #     amr_graph, align_list, exist_list, amr_bidir_graph, amr_edge2type = get_graph(amr_info, vocabs, tokens)

                #     instance = Instance(
                #         sent_id=sent_id,
                #         tokens=tokens,
                #         pieces=pieces,
                #         piece_idxs=piece_idxs,
                #         token_lens=token_lens,
                #         attention_mask=attn_mask,
                #         amr=amr_graph,
                #         align=align_list,
                #         exist=exist_list,
                #         ie2amr_edge_map=None,
                #         amr_bidir_graph=amr_bidir_graph,
                #         amr_edge2type=amr_edge2type
                #     )
                #     data.append(instance)


        self.data = data
    
    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_tokens = []
        batch_token_lens = []
        batch_attention_masks = []
        sent_ids = [inst.sent_id for inst in batch]
        token_nums = [len(inst.tokens) for inst in batch]
        max_token_num = max(token_nums)

        amrs = []
        aligns = []
        exists = []
        amr_bidir_graphs = []
        amr_edge2types = []
        batch_ie2amr_map = []
        for inst in batch:
            token_num = len(inst.tokens)
            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_mask)
            batch_token_lens.append(inst.token_lens)
            batch_tokens.append(inst.tokens)
            amrs.append(inst.amr.to(self.config.gpu_device))
            aligns.append(inst.align)
            exists.append(inst.exist)
            amr_bidir_graphs.append(inst.amr_bidir_graph)
            amr_edge2types.append(inst.amr_edge2type)
            batch_ie2amr_map.append(inst.ie2amr_edge_map)
            
        if self.gpu:
            batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
            batch_attention_masks = torch.cuda.FloatTensor(
                batch_attention_masks)
            token_nums = torch.cuda.LongTensor(token_nums)
        else:
            batch_piece_idxs = torch.LongTensor(batch_piece_idxs)
            batch_attention_masks = torch.FloatTensor(batch_attention_masks)
            token_nums = torch.LongTensor(token_nums)

        return Batch(
            sent_ids=sent_ids,
            tokens=[inst.tokens for inst in batch],
            pieces=[inst.pieces for inst in batch],
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            token_nums=token_nums,
            amr=amrs,
            align=aligns,
            exist=exists,
            ie2amr_edge_map=batch_ie2amr_map,
            amr_bidir_graph=amr_bidir_graphs,
            amr_edge2type=amr_edge2types
        )

class UnlabeledDataset(Dataset):
    def __init__(self, config, gpu=False, require_len=19190):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        :param ignore_title (bool): Ignore sentences that are titles (default=False).
        """
        self.config = config
        self.path = config.aux_train_file
        self.data = []
        self.gpu = gpu
        self.max_length = 128
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        """Load data from file."""
        overlength_num = 0
        underlength_num = 0
        self.problematic_num = 0
        raw_data_list = []
        with open(self.path,'r',encoding='utf-8') as fin:
            for i, line in enumerate(fin):
                inst = json.loads(line)
                raw_data_list.append(inst)
        order_data = raw_data_list

        for inst in order_data:
            inst_len = len(inst['pieces'])
            if self.max_length != -1 and inst_len > self.max_length - 2:
                overlength_num += 1
                continue
            self.data.append(inst)
        if overlength_num:
            print('Discarded {} overlength instances'.format(overlength_num))
        if underlength_num:
            print('Discarded {} underlength instances'.format(underlength_num))
        if self.problematic_num:
            print('Discarded {} problematic instances'.format(self.problematic_num))
        # print('Loaded {} instances from {}'.format(len(self), self.path))

    def numberize(self, tokenizer, vocabs):
        """Numberize word pieces, labels, etcs.
        :param tokenizer: Bert tokenizer.
        :param vocabs (dict): a dict of vocabularies.
        """
        data = []
        
        print('Loaded {} instances from {}'.format(len(self), self.path))
        for inst in tqdm(self.data):
            tokens = inst['tokens']
            pieces = inst['pieces']
            sent_id = inst['sent_id']
            token_num = len(tokens)
            token_lens = inst['token_lens']
            
            piece_idxs = tokenizer.encode(pieces,
                                        add_special_tokens=True,
                                        max_length=self.max_length,
                                        truncation=True)
            pad_num = self.max_length - len(piece_idxs)
            attn_mask = [1] * len(piece_idxs) + [0] * pad_num
            piece_idxs = piece_idxs + [0] * pad_num

            instance = Instance(
                sent_id=sent_id,
                tokens=tokens,
                pieces=pieces,
                piece_idxs=piece_idxs,
                token_lens=token_lens,
                attention_mask=attn_mask
            )
            data.append(instance)
            
        self.data = data
    
    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_tokens = []
        batch_token_lens = []
        batch_attention_masks = []
        sent_ids = [inst.sent_id for inst in batch]
        token_nums = [len(inst.tokens) for inst in batch]
        max_token_num = max(token_nums)

        
        for inst in batch:
            token_num = len(inst.tokens)
            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_mask)
            batch_token_lens.append(inst.token_lens)
            batch_tokens.append(inst.tokens)
            
        if self.gpu:
            batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
            batch_attention_masks = torch.cuda.FloatTensor(
                batch_attention_masks)
            token_nums = torch.cuda.LongTensor(token_nums)
        else:
            batch_piece_idxs = torch.LongTensor(batch_piece_idxs)
            batch_attention_masks = torch.FloatTensor(batch_attention_masks)
            token_nums = torch.LongTensor(token_nums)

        return Batch(
            sent_ids=sent_ids,
            tokens=[inst.tokens for inst in batch],
            pieces=[inst.pieces for inst in batch],
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            token_nums=token_nums
        )