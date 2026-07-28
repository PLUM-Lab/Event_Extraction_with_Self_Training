import json

input_path = "dataset/ACE05-E/roberta_large/train.oneie.json"

role_type_count = {}

with open(input_path, 'r', encoding='utf-8') as r:
    for line in r:
        inst = json.loads(line)
        events = inst['event_mentions']
        for event in events:
            for arg in event['arguments']:
                role = arg['role']
                if not role in role_type_count:
                    role_type_count[role] = 1
                else:
                    role_type_count[role] += 1
print(role_type_count)