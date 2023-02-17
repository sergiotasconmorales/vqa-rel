import os
from os.path import join as jp
from matplotlib import pyplot as plt
import numpy as np
import json
import torch
import random


path_data = '/home/sergio814/Documents/PhD/code/data/lxmert/data/gqa/'
path_images = '/home/sergio814/Documents/PhD/code/data/GQA/images/'
path_output = '/home/sergio814/Documents/PhD/code/data/lxmert/data/examples_gqa'
os.makedirs(path_output, exist_ok=True)

subset = 'val'
os.makedirs(jp(path_output, subset), exist_ok=True)
num_samples = 50

# read qa
with open(jp(path_data, subset + '.json'), 'r') as f:
    qa = json.load(f)
    id2entry = {e['question_id']: e for e in qa}

qa_sub = [e for e in qa if 'rel' in e] # i.e. all sub-questions

for i in range(num_samples):
    # choose a random sub-question and plot it along with it's parent
    sub = random.choice(qa_sub)
    main = id2entry[sub['parent']]
    sub_id = sub['question_id']
    main_id = main['question_id']
    sub_question = sub['sent']
    main_question = main['sent']
    sub_ans_gt = list(sub['label'].keys())[0]
    main_ans_gt = list(main['label'].keys())[0]
    sub_img_path = jp(path_images, subset, sub['img_id'] + '.jpg')
    main_img_path = jp(path_images, subset, main['img_id'] + '.jpg')
    if not os.path.exists(sub_img_path) or not os.path.exists(main_img_path):
        continue
    sub_img = plt.imread(sub_img_path)
    main_img = plt.imread(main_img_path)
    # plot images with questions and answers as titles
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax[0].imshow(main_img)
    ax[0].set_title(main_question + ' ' + main_ans_gt)
    ax[1].imshow(sub_img)
    ax[1].set_title(sub_question + ' ' + sub_ans_gt)
    plt.savefig(jp(path_output, subset, str(i) + '.png'))
