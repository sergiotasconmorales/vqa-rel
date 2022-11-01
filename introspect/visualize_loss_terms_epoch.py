# Script to visualize the behavior of the loss terms (BCE and consistency term) for each epoch, for each iteration.

import json
from os.path import join as jp
from matplotlib import pyplot as plt

exp_name = '138'
path_logs = '/home/sergio814/Documents/PhD/code/logs/lxmert/snap/vqa'

# open consistency_log.json
with open(jp(path_logs, 'config_{}_hpc'.format(exp_name),'consistency_log.json')) as f:
    consistency_log = json.load(f)

# do the same for bce_log.json
with open(jp(path_logs, 'config_{}_hpc'.format(exp_name), 'bce_log.json')) as f:
    bce_log = json.load(f)

# get the number of epochs
n_epochs = len(consistency_log)
assert len(consistency_log) == len(bce_log)
# create subplots, one for each epoch
fig, ax = plt.subplots(1, n_epochs, figsize=(10, 10))
for i in range(n_epochs):
    assert len(consistency_log[str(i)]) == len(bce_log[str(i)])
    # plot both things on the same subplot
    ax[i].plot(consistency_log[str(i)], label='consistency')
    ax[i].plot(bce_log[str(i)], label='bce')
    ax[i].set_title('Epoch ' + str(i))
    ax[i].legend()
plt.show()
a = 42