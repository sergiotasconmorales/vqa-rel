# Check how often pairs in train appear in val

import json
from tqdm import tqdm
from os.path import join as jp
import os
import torch

path_introspect = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_noeq'
path_exp = '/home/sergio814/Documents/PhD/code/logs/lxmert/snap/vqa/config_006_hpc'

if not os.path.exists(jp(path_exp, 'id2inc.pt')):
    raise FileNotFoundError

# read train
with open(jp(path_introspect, 'train.json'), 'r') as f:
    t = json.load(f)

# read val
with open(jp(path_introspect, 'val.json'), 'r') as f:
    v = json.load(f)

# build pairs for train
all_sub_train = [e for e in t if e['role'] == 'sub']
id2question_train = {e['question_id']: e['sent'] for e in t}
pairs_train = [(id2question_train[sub['parent']].lower(), sub['sent'].lower()) for sub in all_sub_train]
main_train = [e[0] for e in pairs_train]
sub_train = [e[1] for e in pairs_train]

# build pairs for val
all_sub_val = [e for e in v if e['role'] == 'sub']
id2question_val = {e['question_id']: e['sent'] for e in v}
pairs_val = [(id2question_val[sub['parent']].lower(), sub['sent'].lower()) for sub in all_sub_val]
rels_val = [sub['rel'] for sub in all_sub_val]
ids_val = [sub['question_id'] for sub in all_sub_val]

cnt = 0
ids_overlap = []
for p_val, rel, id_  in tqdm(zip(pairs_val, rels_val, ids_val), total=len(rels_val)):
    if p_val[0] in main_train:
        if p_val[1] in sub_train:
            if rel in ['<--', '-->']:
                cnt += 1
                ids_overlap.append(id_)

print('total pairs from val which are also in train:', cnt, 'out of', len(pairs_val), '({:.2f} %)'.format(100*cnt/len(pairs_val)))

# now check how many of these ids are consistent for a given experiment
id2inc = torch.load(jp(path_exp, 'id2inc.pt'))
accu = 0 # number of inconsistencies among the val pairs that overlap with train for valid rels (<-- or -->)
for id_ in ids_overlap:
    accu += id2inc[id_]

print('Out of', cnt, 'pairs,', accu, 'are inconsistent for current exp. That is', '{:.2f}%'.format(100*accu/cnt))