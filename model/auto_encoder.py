#! /usr/bin/env python
import pdb
import math
import torch
import torch.nn as nn
import torch.nn.init as init

def conv(cin, cout):
    return nn.Conv2d(cin, cout, kernel_size=3, padding=1, stride=1, bias=False)

def bn(chn):
    return nn.BatchNorm2d(chn)

def relu():
    return nn.LeakyReLU(0.1)


class Coder(nn.Module):
    """ Three Conv layer Coder """
    def __init__(self, cin, cout, use_bn=True):
        super().__init__()

        if use_bn:
            layers = [relu(), conv(cin, cin), bn(cin), relu(),
                      conv(cin, cout), bn(cout), relu(),
                      conv(cout, cout), bn(cout), relu()]
        else:
            layers = [conv(cin, cin), relu(),
                      conv(cin, cout), relu(),
                      conv(cout, cout), relu()]
        self.layers = nn.Sequential(*layers)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)


class auto_encoder(nn.Module):
    """ Six Conv Layer Auto-Encoder """
    def __init__(self, cin, cout, use_bn=True):
        super().__init__()

        self.encoder = Coder(cin, cout, use_bn=use_bn)
        self.decoder = Coder(cout, cin, use_bn=use_bn)


    def forward(self, x):
        factor = self.encoder(x)
        y = self.decoder(factor)
        return factor, y

