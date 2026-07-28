import json
import os
"""Splits the input file into train/dev/test sets.

Args:
    input_file (str): path to the input file.
    output_dir (str): path to the output directory.
    split_path (str): path to the split directory that contains three files,
        train.doc.txt, dev.doc.txt, and test.doc.txt . Each line in these
        files is a document ID.
"""
print('Splitting the dataset into train/dev/test sets')

split_path = 'resource/splits/ERE-EN'
input_path = 'datasets/ere'
output_dir = 'datasets/ere'
all_data = []

with open(os.path.join(input_path, 'english.normal.oneie.json'), 'r', encoding='utf-8') as r:
    for line in r:
        line = json.loads(line)
        all_data.append(line)

with open(os.path.join(input_path, 'english.r2v2.oneie.json'), 'r', encoding='utf-8') as r:
    for line in r:
        line = json.loads(line)
        all_data.append(line)

with open(os.path.join(input_path, 'english.parallel.oneie.json'), 'r', encoding='utf-8') as r:
    for line in r:
        line = json.loads(line)
        all_data.append(line)

print('Total number of data',len(all_data))

train_docs, dev_docs, test_docs = set(), set(), set()
# Load doc ids
with open(os.path.join(split_path, 'train.doc.txt')) as r:
    train_docs.update(r.read().strip('\n').split('\n'))
with open(os.path.join(split_path, 'dev.doc.txt')) as r:
    dev_docs.update(r.read().strip('\n').split('\n'))
with open(os.path.join(split_path, 'test.doc.txt')) as r:
    test_docs.update(r.read().strip('\n').split('\n'))

# Split the dataset
used_doc = set([])
num_train = 0
num_dev = 0
num_test = 0
with open(os.path.join(output_dir, 'train.oneie.json'), 'w') as w_train, \
    open(os.path.join(output_dir, 'dev.oneie.json'), 'w') as w_dev, \
    open(os.path.join(output_dir, 'test.oneie.json'), 'w') as w_test:
    for line in all_data:
        inst = line
        doc_id = inst['doc_id']
        if doc_id in train_docs:
            w_train.write(json.dumps(line)+'\n')
            used_doc.add(doc_id)
        elif doc_id in dev_docs:
            w_dev.write(json.dumps(line)+'\n')
            used_doc.add(doc_id)
        elif doc_id in test_docs:
            w_test.write(json.dumps(line)+'\n')
            used_doc.add(doc_id)
all_docs = list(train_docs) + list(dev_docs) + list(test_docs)
for doc_id in all_docs:
    if not doc_id in used_doc:
        print(doc_id)