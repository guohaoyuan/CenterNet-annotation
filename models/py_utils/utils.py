import torch
import torch.nn as nn

class convolution(nn.Module):                                           # 草丛三基友：conv + bn +rule
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class fully_connected(nn.Module):                                       # FC layer + bn + relu
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear            # 判断是否添加BN层
        relu   = self.relu(bn)
        return relu

class residual(nn.Module):          # hourglass-104 is built from residual blocks which consists of two 3x3 conv and a skip connection
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):                                                # k在本函数中未得到使用，只是为了保持与make_layer函数参数一致方便
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)       # 正常情况下，有BN层就不需要用bias
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):                                      # residual block，先改变通道数再级联.
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):                                                                                 # modules表示的是layer的层数，比如说3层残差模块
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)                                                                               # 将网络结构以元组形式，输入到seq序列中

def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):                                 # residual block，先级联再改变通道数
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))                                                         # 除了把第一个块放到最后位置，看不出来与上面函数的区别，
    return nn.Sequential(*layers)
