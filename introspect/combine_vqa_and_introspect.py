# Script to combine VQA2 and Introspect

from os.path import join as jp
import json
import os
from copy import deepcopy
from tqdm import tqdm

path_data = '/home/sergio814/Documents/PhD/code/data/lxmert/data'
path_output = jp(path_data, 'vqa2introspect')
os.makedirs(path_output, exist_ok=True)
vqa_folder_name = 'vqa'
introspect_folder_name = 'introspect_nodup'

subsets = ['train', 'val']

for s in subsets:

    if os.path.exists(jp(path_output, '{}.json'.format(s))):
        print('File already exists')
        continue

    # open vqa2 data
    if s == 'train':
        file_name = s
    else:
        file_name = 'nominival'
    with open(jp(path_data, vqa_folder_name, '{}.json'.format(file_name)), 'r') as f:
        vqa_data = json.load(f)


    # open introspect data
    with open(jp(path_data, introspect_folder_name, '{}.json'.format(s)), 'r') as f:
        int_data = json.load(f)

    combined = deepcopy(int_data) # new list to store combined set of QA pairs

    # list all question ids in vqa so that you don't repeat them when adding introspect
    ids_introspect = [e['question_id'] for e in int_data]

    cnt=0 # count common (sanity check)
    for item in tqdm(vqa_data):
        item_id = item['question_id']
        if item_id not in ids_introspect:
            item['role'] = 'ind'
            combined.append(item)
        else:
            cnt += 1
            ids_introspect.remove(item_id)

    with open(jp(path_output, '{}.json'.format(s)), 'w') as f:
        json.dump(combined, f)

# now process ans2label files (only once)
# vqa
with open(jp(path_data, vqa_folder_name, 'trainval_label2ans.json'), 'r') as f:
    lab2ans_vqa = json.load(f)

# introspect
with open(jp(path_data, introspect_folder_name, 'trainval_label2ans.json'), 'r') as f:
    lab2ans_int = json.load(f)

missing = set(lab2ans_int) - set(lab2ans_vqa)

lab2ans_vqa += list(missing)

ans2lab = {e:i for e,i in zip(lab2ans_vqa, range(len(lab2ans_vqa)))}

with open(jp(path_output, 'trainval_ans2label.json'), 'w') as f:
    json.dump(ans2lab, f)

with open(jp(path_output, 'trainval_label2ans.json'), 'w') as f:
    json.dump(lab2ans_vqa, f)
