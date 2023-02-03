# Script to show inconsistent answers for a given experiment.

import os
import torch
import json
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
from os.path import join as jp

path_exp = '/home/sergio814/Documents/PhD/code/logs/lxmert/snap/vqa/config_009_hpc'
path_vqa_data = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_noeq'
path_images = '/home/sergio814/Documents/PhD/code/data/coco/images/val'
num_pairs = 100

path_output = jp(path_exp, 'inconsistent_examples')
os.makedirs(path_output, exist_ok=True)

# open json file
with open(jp(path_exp, 'val_predict.json')) as f:
    val_predict = json.load(f)
q_id_ans_pred = {e['question_id']: e['answer'] for e in val_predict}

# open val file
with open(jp(path_vqa_data, 'val.json')) as f:
    val = json.load(f)

# load label2ans dict
with open(jp(path_vqa_data, 'trainval_label2ans.json')) as f:
    label2ans = json.load(f)

path_inconsistencies_file = jp(path_exp, 'id2inc.pt')
if not os.path.exists(path_inconsistencies_file):
    raise ValueError('No inconsistencies file found at {}'.format(path_inconsistencies_file))

q_id_question = {e['question_id']: e['sent'] for e in val}
q_id_image = {e['question_id']: e['img_id']+'.jpg' for e in val}
q_id_ans_gt = {e['question_id']: e['label'] for e in val}

# open pt file using torch
id2inc = torch.load(path_inconsistencies_file)
all_inconsistent = {k:v for k,v in id2inc.items() if v==1}

subid2mainid = {e['question_id']:e['parent'] for e in val if e['role'] == 'sub'}
subid2rel = {e['question_id']:e['rel'] for e in val if e['role'] == 'sub'}

# find some random pairs and plot them
for i in tqdm(range(num_pairs)):
    sub_id = random.choice(list(all_inconsistent.keys()))
    main_id = subid2mainid[sub_id]
    main_question = q_id_question[main_id]
    main_ans_pred = q_id_ans_pred[main_id]
    main_ans_gt = q_id_ans_gt[main_id]
    rel = subid2rel[sub_id]
    sub_question = q_id_question[sub_id]
    sub_nas_pred = q_id_ans_pred[sub_id]
    sub_ans_gt = q_id_ans_gt[sub_id]

    assert q_id_image[main_id] == q_id_image[sub_id]
    image_name = q_id_image[sub_id]
    path_image = jp(path_images, image_name)

    plt.figure()
    img = plt.imread(path_image)
    plt.imshow(img)
    plt.axis('off')
    plt.title('{} {}, {}\n{}\n{} {}, {}'.format(main_question, main_ans_gt, main_ans_pred, rel, sub_question, sub_ans_gt, sub_nas_pred))
    plt.savefig(jp(path_output, str(i).zfill(5) + '.png'), bbox_inches='tight')