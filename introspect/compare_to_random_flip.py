# Script to compare baseline to case in which, for each inconsistent case, I randomly flip one of the two answers. Then check the performance. 

import json
import os
import numpy as np
import torch
from tqdm import tqdm
import random
from os.path import join as jp
from collections import defaultdict
from copy import deepcopy
# set seed for random
random.seed(1234)

strategy = 'flip_sub'
exp_base = '181'
path_exp = '/home/sergio814/Documents/PhD/code/logs/lxmert/snap/vqa/config_{}_hpc'.format(exp_base)
path_data = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_nodup'

# read val_predict.json
with open(jp(path_exp, 'val_predict.json'), 'r') as f:
    pred = json.load(f)
    pred_dict = {e['question_id']: e['answer'] for e in pred}

# read id2inc.pt
id2inc = torch.load(jp(path_exp, 'id2inc.pt'))

# read val.json
with open(jp(path_data, 'val.json'), 'r') as f:
    qa = json.load(f)
    id2datum = {e['question_id']: e for e in qa}

for k,v in id2datum.items():
    v['dset'] = 'introspect'

# build sub_id2main_id
sub2main = {e['question_id']: e['parent'] for e in qa if e['role']=='sub'}

def get_accu(uid2ans, pprint=False):
    score = 0.
    cnt = 0
    dset2score = defaultdict(lambda: 0.)
    dset2cnt = defaultdict(lambda: 0)
    for uid, ans in uid2ans.items():
        if uid not in id2datum:   # Not a labeled data
            continue
        datum = id2datum[uid]
        label = datum['label']
        dset = datum['dset']
        if ans in label:
            score += label[ans]
            dset2score[dset] += label[ans]
        cnt += 1
        dset2cnt[dset] += 1
    accu = score / cnt
    dset2accu = {}
    for dset in dset2cnt:
        dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

    if pprint:
        accu_str = "Overall Accu %0.4f, " % (accu)
        sorted_keys = sorted(dset2accu.keys())
        for key in sorted_keys:
            accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
        print(accu_str)

    return accu, dset2accu


def flip_answer(ans):
    if ans=='yes':
        return 'no'
    elif ans=='no':
        return 'yes'
    else:
        return ans

acc, dset2accu = get_accu(deepcopy(pred_dict))
print('Acc before flipping: {:.2f}'.format(acc*100))

for id_sub, inc in tqdm(id2inc.items()):
    id_main = sub2main[id_sub]
    if inc==1: # inconsistent
        if strategy == 'random_flip':
            # flip either the main question or the sub-question's answer
            if random.random() < 0.5:
                # flip answer of main question
                pred_dict[id_main] = flip_answer(pred_dict[id_main])
            else:
                # flip answer of sub-question
                pred_dict[id_sub] = flip_answer(pred_dict[id_sub])
        elif strategy == 'flip_main':
            pred_dict[id_main] = flip_answer(pred_dict[id_main])
        elif strategy == 'flip_sub':
            pred_dict[id_sub] = flip_answer(pred_dict[id_sub])
        else:
            raise ValueError('Unknown strategy {}'.format(strategy))

acc, dset2accu = get_accu(deepcopy(pred_dict))
print('Acc after flipping: {:.2f}'.format(acc*100))

# write new val_predict.json
val_predict_new = [{'question_id': k, 'answer': v} for k, v in pred_dict.items()]
with open(jp(path_exp, 'val_predict_new.json'), 'w') as f:
    json.dump(val_predict_new, f)

