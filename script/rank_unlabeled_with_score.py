# this code ranks the unlabled instance with score from score model

import json


# PERCENT = 0.1

data_list = []

for i in range(5):
    data = []
    with open('experiment/label_data/ACE+/labeled_dataset_'+str(i+1)+'.txt','r') as fin:
        for line in fin:
            line = json.loads(line)
            data.append(line)
    data_list.append(data)

# sorted_data = sorted(data, key=lambda inst: inst['avg_score'], reverse=True)

trigger_stat = {}
role_stat = {}

# length = round(1 * len(sorted_data))
# print('top: ',length)

# total_trigger = 0
# total_entity =0
total_role = 0
num_valid = 0

sent_id_list = []
for data in data_list:
    send_ids = set([])
    for inst in data:
        if inst['avg_score'] == 0:
            continue
        send_ids.add(inst['sent_id'])
    print(len(send_ids))
    sent_id_list.append(send_ids)

# collect all ids that have argument role
confident_ids = None
for send_ids in sent_id_list:
    if confident_ids is None:
        confident_ids = send_ids
    else:
        confident_ids = confident_ids.intersection(send_ids)

print(len(confident_ids))

order_list = {}
for data in data_list:
    for inst in data:
        if inst['avg_score'] == 0:
            continue
        if inst['sent_id'] in confident_ids:
            avg_score = 0.0
            for pred in inst['pred']:
                avg_score += (pred['score'] + 1) / 2
            avg_score /= len(inst['pred'])
            if avg_score > 1.0 or avg_score < 0.0:
                print(avg_score)
            if inst['sent_id'] in order_list:
                order_list[inst['sent_id']].append(avg_score)
            else:
                order_list[inst['sent_id']] = [avg_score]

output_list = []
for k, v in order_list.items():
    if not len(v) == 5:
        print(v)
    v = sum(v)/len(v)
    output_list.append(tuple([k,v]))

output_list = sorted(output_list, key=lambda inst: inst[1], reverse=True)
with open('dataset/aligned_amr3.0/order_list_ace+.score.json','w') as fout:
    json.dump(output_list,fout)



# order_list = []
# for inst in sorted_data[:length]:
#     if inst['avg_score'] == 0:
#         continue
#     order_list.append(tuple([inst['sent_id'],inst['avg_score']]))
#     num_valid += 1
#     for pred in inst['pred']:
#         trigger_type = pred['trigger_type']
#         role_type = pred['pred_role']
#         role_score = pred['ie_score']
#         trigger_score = pred['trigger_score']
#         score = pred['score']
#         # if score < 0:
#         #     print(inst)
#         # if not trigger_type in trigger_stat:
#         #     trigger_stat[trigger_type] = {'num':1,'ie_score':trigger_score}
#         # else:
#         #     trigger_stat[trigger_type]['num'] += 1
#         #     trigger_stat[trigger_type]['ie_score'] += trigger_score
#         total_role+=1
#         if not role_type in role_stat:
#             role_stat[role_type] = {'num':1,'ie_score':role_score,'score':score}
#         else:
#             role_stat[role_type]['num'] += 1
#             role_stat[role_type]['ie_score'] += role_score
#             role_stat[role_type]['score'] += score
    

# # for key, item in trigger_stat.items():
# #     print(key)
# #     print(item['num'],item['ie_score']/item['num'])

# print('total valid: ',num_valid)
# print('total role: ',total_role)
# for key, item in role_stat.items():
#     print(key)
#     print(item['num'],item['ie_score']/item['num'], item['score']/item['num'])

# with open('dataset/aligned_amr3.0/order_list_ace+.json','w') as fout:
#     json.dump(order_list,fout)

