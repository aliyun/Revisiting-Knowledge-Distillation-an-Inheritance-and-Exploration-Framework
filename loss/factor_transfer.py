#! /usr/bin/env python
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

def _factor_transfer(x, y, p=1):
    assert x.dim() == y.dim() == 4
    x = F.normalize(x.view(x.size(0), -1), p=2, dim=1)
    y = F.normalize(y.view(y.size(0), -1), p=2, dim=1)
    diff = x - y
    diff = diff.norm(p=p).pow(p) / diff.numel()
    return diff.pow(1/p)

class NormLoss(nn.Module):
    def __init__(self, p=1, negative=False):
        super().__init__()
        self.p = p
        self.negative = negative

    def forward(self, x, y):
        if self.negative:
            return  -_factor_transfer(x, y, p=self.p)
        else:
            return _factor_transfer(x, y, p=self.p)

class CosLoss(nn.Module):
    def __init__(self, negative=False):
        super().__init__()
        self.negative = negative

    def forward(self, x, y):
        assert x.dim() == y.dim() == 4
        x = F.normalize(x.view(x.size(0), -1), p=2, dim=1)
        y = F.normalize(y.view(y.size(0), -1), p=2, dim=1)
        inner_product = torch.sum(x * y, dim=1)

        if self.negative:
            return inner_product.abs().sum() / inner_product.numel()
        else:
            return (1 - inner_product).sum() / inner_product.numel()

class FTLoss(nn.Module):
    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def forward(self, x, y):
        return _factor_transfer(x, y, p=self.norm)

class DDTLoss(nn.Module):
    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def forward(self, inh, exp, stu):
        l_inh = _factor_transfer(inh, stu, p=self.norm)
        l_exp = _factor_transfer(exp, stu, p=self.norm)
        return l_inh - l_exp

