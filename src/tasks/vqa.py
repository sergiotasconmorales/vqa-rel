# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import default_collate
from tqdm import tqdm
from aux.io import read_config, update_args

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQADatasetPairs, VQATorchDataset, VQATorchDatasetPairs, VQAEvaluator, collater_pairs

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
EPSILON = 1e-10 

def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    if 'pairs' in args:
        if args.pairs and args.test is None: 
            dset = VQADatasetPairs(splits)
            tset = VQATorchDatasetPairs(dset)
            collater_fn = collater_pairs
        else:
            dset = VQADataset(splits)
            tset = VQATorchDataset(dset)
            collater_fn = default_collate
    else:
        dset = VQADataset(splits)
        tset = VQATorchDataset(dset)
        collater_fn = default_collate
    
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True, collate_fn=collater_fn
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)



def consistency_loss(prob, target, rel, cnst_fcn='fcn1'):
    assert prob.shape[0] == target.shape[0] == rel.shape[0]
    if torch.sum(rel) == 2*rel.shape[0]: # no useful pairs
        return torch.tensor(0)
    # get main and sub parts of everything
    prob_main = prob[::2, :]
    prob_sub = prob[1::2, :]
    target_main = target[::2, :]
    target_sub = target[1::2, :]
    rel_red = rel[::2]

    p = torch.zeros_like(rel_red, dtype=torch.float32).cuda()
    q = torch.zeros_like(rel_red, dtype=torch.float32).cuda()

    # For <-- relations (0 index in rel)
    p[rel_red==0] = 1 - (prob_main[rel_red==0]*(target_main[rel_red==0]>0).to(torch.float32)).sum(1)
    q[rel_red==0] = (prob_sub[rel_red==0]*(target_sub[rel_red==0]>0).to(torch.float32)).sum(1)

    # For --> relations (1 index in rel)
    p[rel_red==1] = (prob_main[rel_red==1]*(target_main[rel_red==1]>0).to(torch.float32)).sum(1)
    q[rel_red==1] = 1 - (prob_sub[rel_red==1]*(target_sub[rel_red==1]>0).to(torch.float32)).sum(1)

    flag_valid = (rel_red<2).to(torch.int64)

    if cnst_fcn == 'fcn1':
        return torch.mean((torch.log(1-p + EPSILON)*torch.log(1-q + EPSILON))[torch.where(flag_valid>0)])
    else:
        sigma = 0.4
        return torch.mean(torch.exp(-((p - 1)**2 + (q - 1)**2)/(2*sigma**2))[torch.where(flag_valid>0)])

class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)


    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        softmax = nn.Softmax(dim=1)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target, rel, flag) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target, rel = feats.cuda(), boxes.cuda(), target.cuda(), rel.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                if 'cnst_fcn' in args: 
                    gain = getattr(args, 'gain')
                    cons_term = consistency_loss(softmax(logit), target, rel, args.cnst_fcn)
                    loss = loss + gain*cons_term
                else:
                    loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        #self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        print(type(eval_tuple).__name__) # DEBUG
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target, _, _) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        if len(quesid2ans) == 0:
            return 0
        else:
            return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":

    # Read config
    config, exp_name = read_config(args.path_config, return_config_name=True)
    # Update args with info from config
    update_args(args, config, exp_name)

    print("Experiment for config file:", exp_name)

    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        # load weights of this config file
        vqa.load(os.path.join(args.output, 'BEST')) 
        args.fast = args.tiny = False       # Always loading all data in test 
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # make sure no pairs are built 
            if hasattr(args, 'pairs'):
                delattr(args, 'pairs')
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            if 'minival' in vqa.train_tuple.dataset.splits:
                subset = 'minival'
            else:
                subset = 'val'
            result = vqa.evaluate(
                get_data_tuple(subset, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, subset + '_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)


