# Script to convert the DME VQA dataset to the format required by the LXMERT model.
# fields I need: answer_type, img_id, label, question_id, question_type, sent, role and parent (if sub) and rel (if sub). 

import os
import json
from tqdm import tqdm
from os.path import join as jp

path_dme = '/home/sergio814/Documents/PhD/code/data/dme_dataset_8_balanced_rels_noeq'
path_output = '/home/sergio814/Documents/PhD/code/data/lxmert/data/dme_dataset_8_balanced_rels_noeq'

# create output folder if it does not exist
os.makedirs(path_output, exist_ok=True)

# for each subset, create a json file with the questions and answers
for subset in ['train', 'val', 'test']:
    print('Processing {} subset...'.format(subset))
    new_data = []
    # open json file
    with open(jp(path_dme, 'qa', '{}qa.json'.format(subset))) as f:
        data = json.load(f)
    for e in tqdm(data):
        # convert entry to desired format
        if isinstance(e['answer'], int):
            ans_type = 'grade'
        else:
            ans_type = 'yes/no'
        new_data.append({
            'answer_type': ans_type,
            'img_id': e['image_name'].split('.')[0],
            'label': {e['answer']: 1},
            'question_id': e['question_id'],
            'question_type': ' '.join(e['question'].split()[:2] + e['question'].split()[-1:]),
            'sent': e['question'],
            'role': e['role'],
            'mask_id': e['mask_name'].split('.')[0],
            'center': e['center']
        })
        # if sub, add fields for parent and rel
        if e['role'] == 'sub':
            new_data[-1]['parent'] = [k['question_id'] for k in data if k['image_name'] == e['image_name']][0]
            new_data[-1]['rel'] = e['rel']
    # save json file
    with open(jp(path_output, '{}.json'.format(subset)), 'w') as f:
        json.dump(new_data, f)