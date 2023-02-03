# Script to check which answers from instrospect are not in the answer vocabulary of LXMERT
# To be run after curate_introspect.py has been executed and the ans2label and label2ans have been copied to the introspect folder.

import json
from os.path import join as jp

subsets = ['train', 'minival', 'nominival']
path_data = '/home/sergio814/Documents/PhD/code/data/lxmert/data/vqa'

# read answer vocab
with open(jp(path_data,'trainval_ans2label.json'), 'r') as f:
    ans2label = json.load(f)

for s in subsets:
    missing = [] # list to store answers from instrospect that are not in the answer vocab
    # read introspect data
    with open(jp(path_data, '{}.json'.format(s)), 'r') as f:
        introspect = json.load(f)
    for entry in introspect:
        lab = entry['label']
        for k,_ in lab.items():
            if k not in ans2label:
                missing.append(k)
    print('Missing for', s, ' ({}):'.format(len(set(missing))))
        