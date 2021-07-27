#! /usr/bin/env python
import torch
import torch.nn as nn

class RecLoss(nn.Module):
    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def forward(self, x, y):
        diff = x - y
        diff = diff.norm(p=self.norm).pow(self.norm) / diff.numel()
        return diff.pow(1/self.norm)
