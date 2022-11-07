# Script to compare baseline to case in which, for each inconsistent case, I randomly flip one of the two answers. Then check the performance. 

import json
import os
import numpy as np
import torch
from os.path import join as jp

exp_base = '177'
path_exp = '/home/sergio814/Documents/PhD/code/logs/lxmert/snap/vqa/config_{}_hpc'.format(exp_base)
path_data = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_nodup'

# read val_predict.json
with open(jp(path_exp, 'val_predict.json'), 'r') as f:
    pred = json.load(f)

# read id2inc.pt
id2inc = torch.load(jp(path_exp, 'id2inc.pt'))

# read val.json
with open(jp(path_data, 'val.json'), 'r') as f:
    qa = json.load(f)

a = 42