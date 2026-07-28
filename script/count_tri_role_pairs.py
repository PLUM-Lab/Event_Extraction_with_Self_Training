import json



event_arg_pairs = {}
event_types = set([])
role_types = set([])
with open('dataset/ACE05-E/roberta_large/train.oneie.json','r') as fin:
    for line in fin:
        line = json.loads(line)
        events = line['event_mentions']
        for event in events:
            event_type = event['event_type']
            args = event['arguments']
            for arg in args:
                role = arg['role']
                pair = (event_type, role)
                if not pair in event_arg_pairs:
                    event_arg_pairs[pair] = 1
                else:
                    event_arg_pairs[pair] += 1
                event_types.add(event_type)
                role_types.add(role)


print(event_types)
print(role_types)
print(event_arg_pairs)

