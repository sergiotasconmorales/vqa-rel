# Script to create GQA dataset with entailed and equivalent pairs. Only binary questions are considered.

from gqa import GQA

# train
gqa_train = GQA(path_gqa='/home/sergio814/Documents/PhD/code/data/GQA/questions1.2', binary_only=True, add_equivalences=True, max_qa_per_file=100000, train=True)
gqa_train.build()
#val
#gqa_val = GQA(path_gqa='/home/sergio814/Documents/PhD/code/data/GQA/questions1.2', binary_only=True, add_equivalences=True, max_qa_per_file=int(200000), train=False)
#gqa_val.build()