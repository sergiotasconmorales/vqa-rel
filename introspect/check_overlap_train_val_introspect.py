# script to check the level of overlap in introspect pairs

import json
from os.path import join as jp
import os
import numpy as np


path_introspect_nodup = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_nodup'

def get_ans(dict_labels):
    # finds best answer: the one with highest score
    ans_list = list(dict_labels.keys())
    ans_scores = list(dict_labels.values())
    index_max = np.argmax(ans_scores)
    return ans_list[index_max]

# read train.json and val.json
with open(jp(path_introspect_nodup, 'train.json'), 'r') as f:
    t = json.load(f)

id2entry_train = {e['question_id']: e for e in t}

with open(jp(path_introspect_nodup, 'val.json'), 'r') as f:
    v = json.load(f)

id2entry_val = {e['question_id']: e for e in v}

# list all sub-questions in train and in val (binary only)
all_sub_train = [e for e in t if e['role'] == 'sub' and (get_ans(e['label']) == 'yes' or get_ans(e['label']) == 'no') and (get_ans(id2entry_train[e['parent']]['label']).lower() == 'yes' or get_ans(id2entry_train[e['parent']]['label']).lower() == 'no')]
all_sub_val = [e for e in v if e['role'] == 'sub' and (get_ans(e['label']) == 'yes' or get_ans(e['label']) == 'no') and (get_ans(id2entry_val[e['parent']]['label']).lower() == 'yes' or get_ans(id2entry_val[e['parent']]['label']).lower() == 'no')]

# now build pairs
all_pairs_train = set([(id2entry_train[sub['parent']]['sent'].lower(), get_ans(id2entry_train[sub['parent']]['label']).lower(), sub['sent'].lower(), get_ans(sub['label']).lower()) for sub in all_sub_train])
all_pairs_val = set([(id2entry_val[sub['parent']]['sent'].lower(), get_ans(id2entry_val[sub['parent']]['label']).lower(), sub['sent'].lower(), get_ans(sub['label']).lower()) for sub in all_sub_val])

a = 42