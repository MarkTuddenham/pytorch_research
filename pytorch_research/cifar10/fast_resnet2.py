import torch
from torch import nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, c, **kw):
        super().__init__()
        self.res1 = make_conv_bn(c, c, pool=False, **kw)
        self.res2 = make_conv_bn(c, c, pool=False, **kw)

    def forward(self, x):
        return self.res2(self.res1(x)) + x


class FastResNet2(nn.Module):
    def __init__(self, weight=1/16):
        super().__init__()
        self.weight = weight
        channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}

        prep = make_conv_bn(3, channels['prep'], pool=False)
        layer1 = make_conv_bn(channels['prep'], channels['layer1'])
        layer2 = make_conv_bn(channels['layer1'], channels['layer2'])
        layer3 = make_conv_bn(channels['layer2'], channels['layer3'])
        pool = nn.MaxPool2d(4)

        layer1 = nn.Sequential(layer1, Residual(channels['layer1']))
        layer3 = nn.Sequential(layer3, Residual(channels['layer3']))

        self.net = nn.Sequential(
            prep,
            layer1,
            layer2,
            layer3,
            pool
        )

        self.linear = nn.Linear(channels['layer3'], 10, bias=False)

    def forward(self, inputs):
        x = self.net(inputs)
        x = x.view(x.size(0), x.size(1))
        x = self.linear(x)
        return self.weight * x


def make_conv_bn(c_in, c_out, pool=True):
    bn = nn.BatchNorm2d(c_out, eps=1e-05, momentum=0.1)
    bn.weight.data.fill_(1.0)
    bn.bias.data.fill_(0.0)

    layers = [
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        bn,
        nn.ReLU(True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)
