#! /usr/bin/env python
""" Knowledge Distillation Loss """

import pdb
import torch
from  torch import nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    def __init__(self, alpha=0, T=4):
        super().__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, outputs, targets, soft_targets):
        alpha = self.alpha
        T = self.T
        loss_ce = F.cross_entropy(outputs, targets)
        loss_kd = F.kl_div(F.log_softmax(outputs / T, dim=1),
                           F.softmax(soft_targets / T, dim=1),
                        reduction='batchmean')
        loss = (1 - alpha) * loss_ce + alpha * (T**2) * loss_kd
        return loss

