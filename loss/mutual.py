#! /usr/bin/env python

import pdb
import torch
from  torch import nn
import torch.nn.functional as F


# Deep Mutual Learning
class DMLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, soft_targets):
        # F.kl_div(x_log, y) == D_KL(y||x)
        loss_kd = F.kl_div(F.log_softmax(outputs, dim=1),
                           F.softmax(soft_targets, dim=1),
                           reduction='batchmean')
        return loss_kd

