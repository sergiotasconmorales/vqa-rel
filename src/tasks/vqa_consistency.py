# script to compute the consistency given the predictions and the QA pairs

from os.path import join as jp
import json
import torch
from torch.nn import ReLU
import numpy as np

path_pred = '/home/sergio814/Documents/PhD/code/logs/lxmert/snap/vqa/config_030_hpc'
path_qa = '/home/sergio814/Documents/PhD/code/data/lxmert/data/introspect_noeq'

pred_name =  'val_predict.json'
qa_name = 'val.json'
map_name = 'trainval_ans2label.json'


def compute_consistency_rels(correct_main, correct_sub, rels, return_indiv=False):
    """Function to compute consistency taking into account the relationships between main and sub-question

    Parameters
    ----------
    correct_main : torch.Tensor
        Binary vector with as many elements as there are sub-questions with relationships. i-th entry is 1 if i-th main question was answered correcty by model.
    correct_sub : torch.Tensor
        Binary vector with as many elements as there are sub-questions with relationships.i-th entry is 1 if i-th sub question was answered correcty by model.
    rels : torch.Tensor of size [N, 4]
        One hot encoding of the relationships for each pair. Columns correspond to -->, <--, <->, ---
    """
    assert len(correct_main) == len(correct_sub)
    assert rels.shape[0] == len(correct_main)

    relu = ReLU()

    # First process <-- relationships (i.e. necessary)
    diff1 = correct_sub - correct_main
    th1 = relu(diff1.squeeze(-1))
    masked1 = th1*rels[:,1]
    necessary_term = torch.sum(masked1)

    # Now --> relations (i.e. sufficient)
    diff2 = correct_main - correct_sub
    th2 = relu(diff2.squeeze(-1))
    masked2 = th2*rels[:,0]
    sufficient_term = torch.sum(masked2)

    # finally <-> relationships (i.e. equivalent pairs)
    th3 = torch.logical_xor(correct_main, correct_sub).to(int)
    masked3 = th3.squeeze(-1)*rels[:,2]
    equivalent_term = torch.sum(masked3)

    # Consistency is defined in terms of the relationships present in the data as follows:
    total_inconsistencies = necessary_term + sufficient_term + equivalent_term
    c = 1 - total_inconsistencies/torch.sum(rels[:,:3])

    if return_indiv:
        return 100*c.item(), {'<--': float(100*necessary_term/total_inconsistencies), '-->': float(100*sufficient_term/total_inconsistencies), '<->': float(100*equivalent_term/total_inconsistencies)}
    else:
        return 100*c.item()

rels_dict = {'-->': 0, '<--': 1, '<->':2, '---':3, 'unk': 3}

# read qa
with open(jp(path_qa, qa_name), 'r') as f:
    qa = json.load(f)
qaid2label = {e['question_id']: e['label'] for e in qa}

# read preds
with open(jp(path_pred, pred_name), 'r') as f:
    pred = json.load(f)

# load map
with open(jp(path_qa, map_name), 'r') as f:
    ans2label = json.load(f)

predid2ans = {e['question_id']: e['answer'] for e in pred}

# add predicted answers to qa
qa_with_rel = [e for e in qa if 'rel' in e]

correct_main = torch.LongTensor(len(qa_with_rel), 1)
correct_sub = torch.LongTensor(len(qa_with_rel), 1)
rels_int = torch.LongTensor(len(qa_with_rel), 1).zero_()
rels_onehot = torch.LongTensor(len(qa_with_rel), 4)
rels_onehot.zero_()

def get_ans(dict_labels):
    # finds best answer: the one with highest score
    ans_list = list(dict_labels.keys())
    ans_scores = list(dict_labels.values())
    index_max = np.argmax(ans_scores)
    return ans2label[ans_list[index_max]]

for i in range(correct_main.shape[0]):
    rels_int[i] = rels_dict[qa_with_rel[i]['rel']]
    main_id = qa_with_rel[i]['parent']
    sub_id = qa_with_rel[i]['question_id']

    main_ans_gt = get_ans(qa_with_rel[i]['label'])
    sub_ans_gt = get_ans(qaid2label[main_id])

    main_ans_pred = ans2label[predid2ans[main_id]]
    sub_ans_pred = ans2label[predid2ans[sub_id]]

    correct_main[i, 0] = torch.eq(torch.tensor(main_ans_pred), torch.tensor(main_ans_gt))
    correct_sub[i, 0] = torch.eq(torch.tensor(sub_ans_pred), torch.tensor(sub_ans_gt))

rels_onehot.scatter_(1, rels_int, 1)
c = compute_consistency_rels(correct_main, correct_sub, rels_onehot, return_indiv=True)
print('Consistency:', c)