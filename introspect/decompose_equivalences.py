# Script to remove <-> relations. Pair is duplicated and relations --> and <-- are assigned.
# IMPORTANT: Files are modified on the fly, meaning you have to put the files with <-> in a folder to be changed

import json
import copy
from tqdm import tqdm
from os.path import join as jp

path_data = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_noeq_nodup'
subsets = ['train', 'val']
new_len_new_q_id = 16 # ids in VQA2 have max length 9 and in introspect they have length 15, so it should be fine to use 16 to ensure unicity

for s in subsets:
    print('Now doing subset:', s)
    new = []
    all_new_ids = []
    with open(jp(path_data, s + '.json'), 'r') as f:
        data = json.load(f)
    for e in tqdm(data): # for each sample
        if e['role'] == 'sub':
            # check relation
            if e['rel'] == '<->':
                temp1 = e.copy()
                temp2 = e.copy()
                temp1['rel'] = '-->'
                temp2['rel'] = '<--'
                # now, you have to create a new question_id for one of the new samples
                # For simplicity, let's take the original id and make it have 14 digits. This should guarantee unicity.
                curr_id_len = len(str(temp2['question_id']))
                missing = new_len_new_q_id - curr_id_len
                found = False
                new_id =  int(str(temp2['question_id']).ljust(missing+curr_id_len, '0'))
                while not found: # attempt to guarantee unicity of question ids
                    if new_id in all_new_ids:
                        new_id += 1
                    else:
                        found = True
                        all_new_ids.append(new_id)
                temp2['question_id'] = new_id
                new.append(temp1)
                new.append(temp2)
            else:
                new.append(e)
        else:
            new.append(e)
    # now save the new version of the entries
    with open(jp(path_data, s + '.json'), 'w') as f:
        json.dump(new, f)