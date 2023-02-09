

from os.path import join as jp
import os
import pandas as pd
import json
from tqdm import tqdm
import gc
import ijson
import random

def build_entry(q_id, e, role='main', parent=None, rel=None):
    if 'image_id' in e:
        img_id = e['image_id']
    else:
        img_id = e['imageId']
    dicti = {   'answer_type': 'yes/no',
                'img_id': img_id,
                'label': {e['answer']: 1},
                'question_id': int('1' + q_id), # add 1 to keep uniqueness of ids
                'question_type': 'gqa',
                'sent': e['question'],
                'role': role
                }
    if role == 'sub':
        dicti['parent'] = parent
        dicti['rel'] = rel
    return dicti


class GQA(object):
    """Class to represent raw GQA dataset, i.e. questions and answers"""
    def __init__(self, path_gqa, binary_only=False, add_equivalences=False, max_qa_per_file=100000, train=True):
        self.path_gqa = path_gqa # path to GQA QA pairs
        self.path_output = jp(self.path_gqa, 'gqa') # path to output folder
        os.makedirs(self.path_output, exist_ok=True)
        self.add_equivalences = add_equivalences # whether to add equivalent questions
        self.binary_only = binary_only 
        # problem with this dataset is the size of each json file. For traininig there are 9 files, each containing 
        # thousands of questions and requiring a lot of RAM
        if train:
            self.path_train_questions = jp(self.path_gqa, 'train_all_questions') # path to 10 json files with trainin questions
            self.prefix = 'train'
        else:
            self.path_train_questions = jp(self.path_gqa, 'val_all_questions')
            self.prefix = 'val'
        self.max_qa_per_file = max_qa_per_file # max number of questions per file

    def extract_binary_related_pairs(self):

        # Step 1. Build dictionary to map question id to filename.
        print('Step 1')
        qa = {}
        self.train_files = os.listdir(self.path_train_questions)
        self.train_files.sort()
        entry_idx = 0
        for trainf in tqdm(self.train_files, desc='Processing qa files', colour='blue'): # 
            print('Current file:', trainf)
            with open(jp(self.path_train_questions, trainf), 'rb') as f:
                elems = ijson.kvitems(f, '')
                #q_ids = [k for k,v in elems]
                tempi = {k:{'question': v['question'], 'answer': v['answer'], 'image_id': v['imageId'], 'sfile': trainf} for k,v in elems}
            qa = qa | tempi

        # Step 2. Go through every file and build df without filling info about entailed, but write the file using qa
        print('Step 2')
        all_pairs = set() # to keep track of already added pairs
        all_sub = set() # to keep track of already added sub questions, so that they cannot be main
        all_main = set() # to keep track of already added main questions, so that they cannot be sub
        for i_file, trainf in enumerate(tqdm(self.train_files)): 
            print('Current file:', trainf)
            entries = [] # to save all pairs in LXMERT format
            with open(jp(self.path_train_questions, trainf), 'rb') as f:
                elems = ijson.kvitems(f, '')
                for k,v in elems:
                    for i_ent, e in enumerate(v['entailed']): # check entailed
                        if int('1'+e) in all_sub: continue
                        if self.binary_only and not ((v['answer'] in ['yes', 'no']) and (qa[e]['answer'] in ['yes', 'no'])): continue
                        # save main and sub
                        if (int('1'+k), int('1'+e)) in all_pairs or (int('1'+e), int('1'+k)) in all_pairs: continue # to avoid duplicates
                        if int('1'+k) in all_sub or int('1'+e) in all_main: continue # to avoid having a sub question as main
                        if int('1'+k) not in all_main:
                            entries.append(build_entry(k, v)) # add main question only once
                            all_main.add(int('1'+k))
                        entries.append(build_entry(e, qa[e], role='sub', parent=int('1'+k), rel='-->'))
                        all_pairs.add((int('1'+k), int('1'+e)))
                        all_pairs.add((int('1'+e), int('1'+k)))
                        all_sub.add(int('1'+e))
                    if self.add_equivalences: # add equivalences if required
                        for i_eq, e in enumerate(v['equivalent']): # check equivalent
                            if int('1'+e) in all_sub: continue
                            if self.binary_only and not ((v['answer'] in ['yes', 'no']) and (qa[e]['answer'] in ['yes', 'no'])): continue
                            if int('1'+k) == int('1'+e): continue # q1 is annotated as equivalent to q1, so here I avoid having the same question as main and sub
                            if (int('1'+k), int('1'+e)) in all_pairs or (int('1'+e), int('1'+k)) in all_pairs: continue # to avoid duplicates
                            if int('1'+k) in all_sub or int('1'+e) in all_main: continue # to avoid having a sub question as main
                            if int('1'+k) not in all_main:
                                entries.append(build_entry(k, v)) # add main question only once
                                all_main.add(int('1'+k))
                            entries.append(build_entry(e, qa[e], role='sub', parent=int('1'+k), rel='<->'))
                            all_pairs.add((int('1'+k), int('1'+e)))
                            all_pairs.add((int('1'+e), int('1'+k)))
                            all_sub.add(int('1'+e))
                            
                    # if max number of qa pairs is reached for current file, save it and start a new one
                    if len(entries) >= self.max_qa_per_file: 
                        with open(jp(self.path_output, self.prefix + 'gqa_related_pairs_' + str(i_file) + '.json'), 'w') as f:
                            json.dump(entries, f)
                        break

    
    def merge_files(self):
        print('Step 3')
        # merge all files into one
        entries = []
        for i_file in tqdm(range(len(self.train_files)), desc='Merging files', colour='blue'):
            with open(jp(self.path_output, self.prefix + 'gqa_related_pairs_' + str(i_file) + '.json'), 'r') as f:
                entries += json.load(f)
        with open(jp(self.path_output, self.prefix + '_unprocessed.json'), 'w') as f:
            json.dump(entries, f)


    def build(self):
        # dataset builder
        self.extract_binary_related_pairs() # step 1 and 2. This generates a temp file with all the questions and answers
        # step 3. Merge all files into one
        self.merge_files()

