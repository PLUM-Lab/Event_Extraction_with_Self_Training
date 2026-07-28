import json

ie_roles = ['Org', 'Instrument', 'Artifact', 'Agent', 'Victim', 'Person', 'Buyer', 'Beneficiary', 'Destination', 'Place', 'Seller', 'Origin', 'Target', 'Recipient', 'Prosecutor', 'Attacker', 'Plaintiff', 'Entity', 'Giver', 'Vehicle', 'Defendant', 'Adjudicator', 'O']
ie_rels = ['GEN-AFF', 'PHYS', 'PER-SOC', 'ORG-AFF', 'PART-WHOLE', 'ART', 'O']
ie_triggers = ['Personnel:End-Position', 'Personnel:Nominate', 'Transaction:Transfer-Ownership', 'Justice:Convict', 'Justice:Appeal', 'Justice:Acquit', 'Conflict:Attack', 'Life:Marry', 'Justice:Sentence', 'Life:Injure', 'Justice:Charge-Indict', 'Justice:Sue', 'Transaction:Transfer-Money', 'Contact:Meet', 'Justice:Arrest-Jail', 'Justice:Fine', 'Conflict:Demonstrate', 'Business:End-Org', 'Personnel:Elect', 'Business:Declare-Bankruptcy', 'Justice:Pardon', 'Life:Be-Born', 'Justice:Execute', 'Personnel:Start-Position', 'Justice:Release-Parole', 'Movement:Transport', 'Life:Divorce', 'Justice:Trial-Hearing', 'Business:Merge-Org', 'Justice:Extradite', 'Life:Die', 'Business:Start-Org', 'Contact:Phone-Write', 'O']
ie_entities = ['FAC', 'WEA', 'VEH', 'ORG', 'GPE', 'LOC', 'PER', 'O']


input_path = "dataset/aligned_amr3.0/aux_amr3.0.jsonl"

rel_dict = {}

with open(input_path,'r') as fin:
    for line in fin:
        line = json.loads(line)
        edges = line['edges']
        for edge, rel in edges.items():
            if rel.endswith('-of'):
                rel = rel[:-3]
            if rel in rel_dict:
                rel_dict[rel]+=1
            else:
                rel_dict[rel] = 1
removed_rel = set([])
count = 0


arg_role = []
for k, v in rel_dict.items():
    if k in ['ARG0','ARG1','ARG2','ARG3','ARG4']:
        arg_role.append(k)
        removed_rel.add(k)
        # print(k,v)
        count+=1
print('arg_role: ',arg_role)

prep_role = []
for k, v in rel_dict.items():
    if k.startswith('prep'):
        prep_role.append(k)
        removed_rel.add(k)
        # print(k,v)
        count+=1
print('prep_role: ',prep_role)

op_role = []
for k, v in rel_dict.items():
    if k.startswith('op'):
        op_role.append(k)
        removed_rel.add(k)
        # print(k,v)
        count+=1
print('op_role: ',op_role)

# snt_role = []
# for k, v in rel_dict.items():
#     if k.startswith('snt'):
#         snt_role.append(k)
#         removed_rel.add(k)
#         # print(k,v)
#         count+=1
# print('snt_role: ',snt_role)

time_role = ['calendar', 'century', 'day', 'dayperiod', 'decade', 'era', 'month', 'quarter', 'season', 'timezone', 'weekday', 'year', 'year2', 'time', 'duration']
for k, v in rel_dict.items():
    if k in time_role:
        removed_rel.add(k)
        # print(k,v)
        count+=1
print('time_role',time_role)


place_role = ['location', 'path', 'direction']
for k, v in rel_dict.items():
    if k in place_role:
        removed_rel.add(k)
        # print(k,v)
        count+=1
print('place_role',place_role)

media_role = ['manner', 'poss', 'medium', 'topic']
for k, v in rel_dict.items():
    if k in media_role:
        removed_rel.add(k)
        # print(k,v)
        count+=1
print('media_role',media_role)

mod_role = ['wiki','name','domain','mod','example']
for k, v in rel_dict.items():
    if k in mod_role:
        removed_rel.add(k)
        # print(k,v)
        count+=1
print('mod_role',mod_role)

part_role = ['part','consist','subevent','subset']
for k, v in rel_dict.items():
    if k in part_role:
        removed_rel.add(k)
        # print(k,v)
        count+=1
print('part_role',part_role)

# remove not related role
not_related = []
fin = open('amr_processing/amrRelAnnotation.txt').read().strip().split('\n')
for line in fin:
    rel, freq, _, related = line.strip().split('\t')
    rel = rel[1:]
    if related == '0':
        not_related.append(rel)
not_related.append('conj-as-if')
not_related.append('range')
not_related.append('cause')
not_related.extend(['snt1', 'snt2', 'snt3', 'snt4', 'snt10', 'snt5', 'snt6', 'snt7', 'snt8', 'snt9', 'snt11'])
not_related.extend(['ARG5', 'ARG6', 'ARG7', 'ARG8', 'ARG9'])
not_related_role = []
for k, v in rel_dict.items():
    if k in not_related and not k in removed_rel:
        removed_rel.add(k)
        # print(k,v)
        not_related_role.append(k)
        count+=1
print('not_related_role',not_related_role)


all_rel = set([])
for k, v in rel_dict.items():
    all_rel.add(k)
print(count)
print(all_rel-removed_rel)



# {'wiki', 'name', 'part', 'source', 'domain', 'consist', 'mod', 'beneficiary', 'example'}

#['Org', 'Instrument', 'Artifact', 'Agent', 'Victim', 'Person', 'Buyer', 'Beneficiary', 'Destination', 'Place', 'Seller', 'Origin', 'Target', 'Recipient', 'Prosecutor', 'Attacker', 'Plaintiff', 'Entity', 'Giver', 'Vehicle', 'Defendant', 'Adjudicator', 'O'] 22

#  ie                          amr
#  Instrument                  instrument
#  Beneficiary                 beneficiary
#  Agent,Victim,Person,Buyer,Seller,Recipient,Prosecutor,Attacker,Plaintiff,Giver,Defendant,Adjudicator     ARG0,ARG1,ARG2,ARG3,ARG4
#  Destination                 destination
#  Place                       place_role
#  Origin                      source
#  Org
#  Artifact
#  Target
#  Entity
#  Vehicle


