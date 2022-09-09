# Project:
#   VQA
# Description:
#   Script to investigate how the samples are changing when the consistency is increased using the loss term. 
# Author: 
#   Sergio Tascon-Morales

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from os.path import join as jp
from PIL import Image
from matplotlib import pyplot as plt

exp_before = '009' # baseline
exp_after = '019' # with loss term

path_data = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_noeq_faulty'
path_logs = '/home/sergio814/Documents/PhD/code/logs/lxmert/snap/vqa'
base_name = 'config_<>_hpc'
path_images = '/home/sergio814/Documents/PhD/code/data/coco/images/val'
path_save = jp(path_data, 'investigate_samples_' + exp_before + '_' + exp_after)

# define paths
path_before = jp(path_logs, base_name.replace('<>', exp_before))
path_after = jp(path_logs, base_name.replace('<>', exp_after))

def get_ans(dict_labels):
    # finds best answer: the one with highest score
    ans_list = list(dict_labels.keys())
    ans_scores = list(dict_labels.values())
    index_max = np.argmax(ans_scores)
    return ans_list[index_max]

# read validation json file
with open(jp(path_data, 'val.json'), 'r') as f:
    val = json.load(f)
    id2gtans = {e['question_id']: get_ans(e['label']) for e in val}
    id2entry = {e['question_id']: e for e in val}

# read answers from path_before and path_after
with open(jp(path_before, 'val_predict.json'), 'r') as f:
    val_before = json.load(f)
    id2ans_before = {e['question_id']: e['answer'] for e in val_before}

with open(jp(path_after, 'val_predict.json'), 'r') as f:
    val_after = json.load(f)
    id2ans_after = {e['question_id']: e['answer'] for e in val_after}

counts_acc = {'rr': [], 'wr': [], 'rw': [], 'ww': []}

for q_id, gt_ans in tqdm(id2gtans.items()):
    ans_before = id2ans_before[q_id]
    ans_after = id2ans_after[q_id]
    if gt_ans == ans_before: # answer was correct before
        if gt_ans == ans_after: # answer is still correct after
            counts_acc['rr'].append(q_id)
        else: # answer is wrong after
            counts_acc['rw'].append(q_id)
    else: # answer was wrong before
        if gt_ans == ans_after: # answer is now correct after
            counts_acc['wr'].append(q_id)
        else: # answer is still wrong after
            counts_acc['ww'].append(q_id)

print('Accuracy changes for all questions')
print('rr: {} ({:.2f}%)'.format(len(counts_acc['rr']), 100*len(counts_acc['rr'])/len(id2gtans)))
print('wr: {} ({:.2f}%)'.format(len(counts_acc['wr']), 100*len(counts_acc['wr'])/len(id2gtans)))
print('rw: {} ({:.2f}%)'.format(len(counts_acc['rw']), 100*len(counts_acc['rw'])/len(id2gtans)))
print('ww: {} ({:.2f}%)'.format(len(counts_acc['ww']), 100*len(counts_acc['ww'])/len(id2gtans)))

# read file id2inc.pt from path_before and path_after
id2inc_before = torch.load(jp(path_before, 'id2inc.pt'))
id2inc_after = torch.load(jp(path_after, 'id2inc.pt'))

# read file id2valid.pt from path_before and path_after
id2valid_before = torch.load(jp(path_before, 'id2valid.pt'))
id2valid_after = torch.load(jp(path_after, 'id2valid.pt'))

counts_cons = {'cc': [], 'ci': [], 'ic': [], 'ii': []}

for q_id in id2inc_before.keys():
    if id2inc_before[q_id] and id2inc_after[q_id]: # both are inconsistent (ii)
        counts_cons['ii'].append(q_id)
    elif id2inc_before[q_id] and not id2inc_after[q_id]: # before was inconsistent, now is consistent (ic)
        counts_cons['ic'].append(q_id)
    elif not id2inc_before[q_id] and id2inc_after[q_id]: # before was consistent, now is inconsistent (ci)
        counts_cons['ci'].append(q_id)
    else: # both are consistent (cc)
        counts_cons['cc'].append(q_id)

print('Consistency changes for valid pairs')
print('cc: {} ({:.2f})%'.format(len(counts_cons['cc']), 100*len(counts_cons['cc'])/len(id2inc_before)))
print('ic: {} ({:.2f})%'.format(len(counts_cons['ic']), 100*len(counts_cons['ic'])/len(id2inc_before)))
print('ci: {} ({:.2f})%'.format(len(counts_cons['ci']), 100*len(counts_cons['ci'])/len(id2inc_before)))
print('ii: {} ({:.2f})%'.format(len(counts_cons['ii']), 100*len(counts_cons['ii'])/len(id2inc_before)))

# now I need to analyze for example in the ic pairs, which of the two questions was corrected more often
# to do this it is convenient to analyze both the accuracy and the consistency at the same time

# first I need a dictionary to go from id of the sub-question to id of the corresponding main question
idsub2idmain = {e['question_id']: e['parent'] for e in val if e['role'] == 'sub'}

# now iterate through every sub id
acc_increase = []
acc_decrease = []
acc_same = []
cnt_ = 0
for q_id in id2inc_before.keys():
    if id2inc_before[q_id] and not id2inc_after[q_id]: # ic case
        main_better = False
        sub_better = False
        # at this point I know it is a ic pair, so now I have to check the frequency with which the main question was corrected and the frequency with which the sub-question was
        id_main = idsub2idmain[q_id] # get parent id
        gt_ans_main = id2gtans[id_main]
        ans_main_before = id2ans_before[id_main]
        ans_main_after = id2ans_after[id_main]

        gt_ans_sub = id2gtans[q_id]
        ans_sub_before = id2ans_before[q_id]
        ans_sub_after = id2ans_after[q_id]

        # let's check it in terms of accuracies (increase, same, decrease) of the pair
        initial_acc = int(gt_ans_main == ans_main_before) + int(gt_ans_sub == ans_sub_before)
        final_acc = int(gt_ans_main == ans_main_after) + int(gt_ans_sub == ans_sub_after)
        if final_acc > initial_acc:
            acc_increase.append((id_main, q_id))
        elif final_acc < initial_acc:
            acc_decrease.append((id_main, q_id))
        else:
            acc_same.append((id_main, q_id))

        cnt_ += 1

    elif not id2inc_before[q_id] and id2inc_after[q_id]: # ci case
        pass

# now plot the pairs for the decrease case and save them.
for pair in acc_decrease:
    id_main, id_sub = pair
    main_entry = id2entry[id_main]
    sub_entry = id2entry[id_sub]

    # read main image and sub image
    main_img = Image.open(jp(path_images, main_entry['img_id'] + '.jpg'))
    sub_img = Image.open(jp(path_images, sub_entry['img_id'] + '.jpg'))

    main_id = main_entry['question_id']
    sub_id = sub_entry['question_id']

    main_question = main_entry['sent']
    sub_question = sub_entry['sent']

    main_gt_ans = id2gtans[id_main]
    sub_gt_ans = id2gtans[id_sub]

    main_ans_before = id2ans_before[id_main]
    sub_ans_before = id2ans_before[id_sub]

    main_ans_after = id2ans_after[id_main]
    sub_ans_after = id2ans_after[id_sub]

    # now plot the images

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(main_img)
    ax[0].set_title('ID:{}\nMain question: {}\n (gt: {}, before: {}, after: {})'.format(main_id, main_question, main_gt_ans, main_ans_before, main_ans_after))
    ax[0].axis('off')
    ax[1].imshow(sub_img)
    ax[1].set_title('ID:{}\nSub question: {}\n (gt: {}, before: {}, after: {})'.format(sub_id, sub_question, sub_gt_ans, sub_ans_before, sub_ans_after))
    ax[1].axis('off')
    plt.suptitle('Relation: ' + sub_entry['rel'])
    plt.tight_layout()
    plt.show()
    #plt.savefig(jp(path_save, 'ic_accdecrease_{}_{}.png'.format(id_main, id_sub)))
    

    a = 42