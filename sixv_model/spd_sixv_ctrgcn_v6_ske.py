import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.autograd import Variable
# from model.spd.nn import BiMap, LogEig, SPDCov2d, DiagonalizingLayer, ReEig
from utils.modules import BiMap, LogEig, ReEig


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, residual=True, alpha=None):
        super(unit_gcn, self).__init__()
        # (4, 3, 64, 25)
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_heads = 8 if in_channels > 8 else 1
        self.fc1 = nn.Parameter(
            torch.stack([torch.stack([torch.eye(A.shape[-1]) for _ in range(self.num_heads)], dim=0) for _ in range(3)],
                        dim=0), requires_grad=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1, groups=self.num_heads) for _ in range(3)])

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.A = A

        self.rpe = nn.Parameter(torch.zeros((3, self.num_heads, int(self.A.max() + 1))))

    def L2_norm(self, weight):
        weight_norm = torch.norm(weight, 2, dim=-2, keepdim=True) + 1e-4
        return weight_norm

    def forward(self, x):
        # (4, 3, 64, 25)
        N, C, T, V = x.size()
        y = None

        pos_emb = self.rpe[:, :, self.A]

        for i in range(3):
            weight_norm = self.L2_norm(self.fc1[i])
            w1 = self.fc1[i]
            w1 = w1 / weight_norm

            w1 = w1 + pos_emb[i] / self.L2_norm(pos_emb[i])
            x_in = x.view(N, self.num_heads, C // self.num_heads, T, V)

            z = torch.einsum("nhctv, hvw -> nhctw", (x_in, w1)).contiguous().view(N, -1, T, V)
            z = self.fc2[i](z)

            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, kernel_size=5,
                 dilations=[1, 2], alpha=None):
        super(TCN_GCN_unit, self).__init__()
        # (4, 3, 64, 25) , out=64
        self.gcn1 = unit_gcn(in_channels, out_channels, A, alpha=alpha)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # (4, 3, 64, 25)
        y = self.gcn1(x)
        y = self.relu(self.tcn1(y) + self.residual(x))
        return y


def batch_cov(points):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


class TopoTrans(nn.Module):
    def __init__(self, out_dim, in_dim=64):
        super(TopoTrans, self).__init__()
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(in_dim, out_dim)
        # self.tanh = nn.Tanh
        # self.pa = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        # x = x.repeat(2,1)
        x = self.mlp(x)
        # BN
        x = self.bn(x)
        x = self.relu(x)

        return x.unsqueeze(2).unsqueeze(3)


class Topo(nn.Module):
    def __init__(self):
        super(Topo, self).__init__()
        self.model = nn.Sequential(
            BiMap(1, 25, 20),
            ReEig(),
            BiMap(1, 20, 15),
            ReEig(),
            BiMap(1, 15, 10),
            ReEig(),
            LogEig()
        )
        self.fc = nn.Linear(100, 64)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.mean(-2)  # (N, C, V)
        x = batch_cov(x)
        x = x + 1e-8 * torch.eye(V, device=x.get_device())  # 64, 25, 25
        x = x.unsqueeze(1)  # 64, 1, 25, 25
        x = self.model(x).view(N, -1)
        x = self.fc(x)

        return x


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, alpha=None):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * 128 * num_point)

        self.to_joint_embedding = nn.Linear(in_channels, 128)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, 128))

        # (4, 3, 64, 25)
        self.l1 = TCN_GCN_unit(128, 128, A, alpha=alpha)
        self.l2 = TCN_GCN_unit(128, 128, A, alpha=alpha)
        self.l3 = TCN_GCN_unit(128, 128, A, alpha=alpha)
        self.l4 = TCN_GCN_unit(128, 128, A, alpha=alpha)
        self.l5 = TCN_GCN_unit(128, 256, A, stride=2, alpha=alpha)

        self.l6 = TCN_GCN_unit(256, 256, A, alpha=alpha)
        self.l7 = TCN_GCN_unit(256, 256, A, alpha=alpha)
        self.l8 = TCN_GCN_unit(256, 256, A, stride=2, alpha=alpha)
        self.l9 = TCN_GCN_unit(256, 256, A, alpha=alpha)
        self.l10 = TCN_GCN_unit(256, 256, A, alpha=alpha)

        self.t0 = TopoTrans(out_dim=128)
        self.t1 = TopoTrans(out_dim=128)
        self.t2 = TopoTrans(out_dim=128)
        self.t3 = TopoTrans(out_dim=128)
        self.t4 = TopoTrans(out_dim=128)

        self.t5 = TopoTrans(out_dim=256)
        self.t6 = TopoTrans(out_dim=256)
        self.t7 = TopoTrans(out_dim=256)
        self.t8 = TopoTrans(out_dim=256)
        self.t9 = TopoTrans(out_dim=256)
        self.topo = Topo()

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        self.aalpha = alpha

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, spd_A=None):
        log = {'image': {}, 'pram': {}}
        """
            N 视频个数(batch_size)
            C = 3 (X,Y,S)代表一个点的信息(位置+预测的可能性)
            T = 64 一个视频的帧数paper规定是64帧，不足的重头循环，多的clip
            V 25 数据集中25个结点
            M = 2 人数，paper中将人数限定在最大2个人
        """
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

        # N C T V M -> N M V C T -> N MVC T
        # print(x.shape)
        x = self.data_bn(x)  # batch_normalize
        # print(x.shape)
        x = x.view(N, M, V, 128, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, 128, T, V)
        # N MVC T -> N M V C T -> N M C T V -> NM C T V

        a = self.topo(x)

        x = self.l1(x + self.t0(a))  # (N*M, 60 , 64, 25)
        x = self.l2(x + self.t1(a))  # (N*M, 60, 64, 25)
        x = self.l3(x + self.t2(a))  # (N*M, 60, 64, 25)
        x = self.l4(x + self.t3(a))  # (N*M, 60, 64, 25)

        x = self.l5(x + self.t4(a))  # (N*M, 120, 32, 25)
        x = self.l6(x + self.t5(a))  # (N*M, 120, 32, 25)
        x = self.l7(x + self.t6(a))  # (N*M, 120, 32, 25)

        x = self.l8(x + self.t7(a))  # (N*M, 240, 16, 25)
        x = self.l9(x + self.t8(a))  # (N*M, 240, 16, 25)
        x = self.l10(x + self.t9(a))  # (N*M, 240, 16, 25)

        # x2 = self.first_tram(x2)
        # x3 = self.second_tram(x3)
        # x = x + x2 + x3

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)  # N, M, C, T*V
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        log['image'] = {
            'rpe': self.l10.gcn1.rpe[0][:, self.l10.gcn1.A].detach(),
            'A': self.l10.gcn1.fc1[0].detach()
        }

        return self.fc(x), log
