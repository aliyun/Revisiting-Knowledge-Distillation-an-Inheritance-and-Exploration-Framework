'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 0.01)
        init.zeros_(m.bias)

def bn_init(bn, scale=1):
    init.constant_(bn.weight, scale)
    init.constant_(bn.bias, 0)

def conv_init(conv):
    init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        init.constant_(conv.bias, 0)

def dw_conv_init(conv):
    tensor = conv.weight
    num_output_fmaps = tensor.size(0)
    receptive_field_size = tensor[0][0].numel()
    # fan_out = num_output_fmaps * receptive_field_size
    fan_out = receptive_field_size
    std = math.sqrt(2.0 / fan_out)
    init.normal_(conv.weight, 0, std)
    if conv.bias is not None:
        init.constant_(conv.bias, 0)

def linear_init(fc):
    init.normal_(fc.weight, 0, 0.01)
    init.constant_(fc.bias, 0)

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.reset_parameters()

    def reset_parameters(self):
        dw_conv_init(self.conv1)
        bn_init(self.bn1)
        conv_init(self.conv2)
        bn_init(self.bn2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class mobilenet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100, width_multiplier=1.0, dropout=0):
        super().__init__()
        self.width_multiplier = width_multiplier

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(int(1024 * width_multiplier), num_classes)

        # self.apply(_weights_init)
        self.reset_parameters()

    def reset_parameters(self):
        conv_init(self.conv1)
        bn_init(self.bn1)
        linear_init(self.linear)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            out_planes = int(out_planes * self.width_multiplier)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        features = out
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out, features


def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
