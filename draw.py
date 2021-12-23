import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import matplotlib.pyplot as plt

# plot loss curves from checkpoint
ckpt = torch.load('./logs_noaug_stl10unlabeled/train_epoch_299.pth')
training_loss_list = ckpt['train_loss_list']

plt.plot(training_loss_list)
plt.title('Raw Training Loss on STL-10-unlabeled')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('logs_noaug_stl10unlabeled.png')
