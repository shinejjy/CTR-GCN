import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.spd.nn import BiMap, LogEig, ReEig


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


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        # (4, 3, 64, 25)
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        # (4, 8, 25) (4, 8, 25) (4, 64, 64, 25)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # (4, 64, 25, 25)
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        # (4, 64, 25, 25) (1, 1, 25, 25)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        # (4, 64, 64, 25)
        return x1


class unit_spd(nn.Module):
    def __init__(self):
        super(unit_spd, self).__init__()
        self.spd_net = nn.Sequential(
            BiMap(1, 1, 17 * 17, 10 * 10),
            ReEig(),
            BiMap(1, 1, 10 * 10, 25),
            LogEig()
        )

    def forward(self, x):
        x = self.spd_net(x)

        return x


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
    def __init__(self, in_channels, out_channels, A3, coff_embedding=4, residual=True):
        super(unit_gcn, self).__init__()
        # (4, 3, 64, 25)
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A3.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

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

        self.A3 = nn.Parameter(torch.from_numpy(A3.astype(np.float32)))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        # (4, 3, 64, 25)
        y = None
        A3 = self.A3.cuda(x.get_device())

        for i in range(self.num_subset):
            z = self.convs[i](x, A3[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class simple_gcn(nn.Module):
    def __init__(self):
        super(simple_gcn, self).__init__()

    def forward(self, x, A):
        out = torch.einsum('uv,nctv->nctu', A, x)
        return out


class split_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A6):
        super(split_gcn, self).__init__()
        self.A6 = nn.Parameter(torch.from_numpy(A6.astype(np.float32)))
        num_split = self.A6.shape[0]
        assert out_channels % num_split == 0
        branch_channels = out_channels // num_split
        self.branches1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_split)
        ])
        self.branches2 = nn.ModuleList([
            simple_gcn() for _ in range(num_split)
        ])

        self.branches3 = nn.ModuleList([
            nn.BatchNorm2d(branch_channels) for _ in range(num_split)
        ])

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        # input dim: (N, C, T, V)
        A6 = self.A6.cuda(x.get_device())
        res = self.residual(x)
        branch_outs = []
        for conv in self.branches1:
            out = conv(x)
            branch_outs.append(out)
        for i, gcn in enumerate(self.branches2):
            branch_outs[i] = gcn(branch_outs[i], A6[i])
        for i, bn in enumerate(self.branches3):
            branch_outs[i] = bn(branch_outs[i])

        out = torch.cat(branch_outs, dim=1)
        out = out + res
        return out


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A3, A6, stride=1, residual=True, kernel_size=5,
                 dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()
        # (4, 3, 64, 25) , out=64
        self.gcn1 = unit_gcn(in_channels, out_channels, A3)
        self.gcn2 = split_gcn(in_channels, out_channels, A6)
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
        spatial_out = self.gcn1(x) + self.gcn2(x)
        y = self.relu(self.tcn1(spatial_out) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        # self.stage1 = Stage1(self.graph, c_dim=17)
        self.stage2 = Stage2(num_class, num_point, num_person, self.graph, in_channels, drop_out)

    def forward(self, x):
        # spd_A = self.stage1().cuda(x.get_device())
        x = self.stage2(x)
        return x


class Stage1(nn.Module):
    def __init__(self, graph, c_dim=21):
        super(Stage1, self).__init__()
        self.spd_A = nn.Parameter(torch.from_numpy(graph.spd_A.astype(np.float32)))
        # self.pre_spd_conv = nn.Conv2d(6, 6, kernel_size=3, stride=2, padding=3)
        self.pre_spd_conv = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=5),
            nn.ReLU(inplace=True),
        )
        self.spdn = unit_spd()
        self.layer3 = nn.Linear(625, 36)
        # self.beta = nn.Parameter(torch.zeros(1))
        self.c_dim = c_dim
        self.conv = nn.Conv2d(1, 6, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self):
        device = self.spd_A.device
        self.spdn = self.spdn.to('cpu')
        spd_A = self.spd_A.view(1, 6, 25, 25)
        spd_A = self.pre_spd_conv(spd_A).view(6, self.c_dim ** 2).detach().cpu().numpy()
        spd_A = np.cov(spd_A, rowvar=False)
        spd_A = torch.from_numpy(spd_A).to(torch.float32).view(1, 1, self.c_dim ** 2, self.c_dim ** 2)
        spd_A = self.spdn(spd_A).view(1, 25, 25)
        spd_A = spd_A.to(device)
        self.spdn = self.spdn.to(device)
        spd_A = self.relu(self.conv(spd_A))

        return spd_A


class Stage2(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0):
        super(Stage2, self).__init__()

        A3 = graph.A3  # 3,25,25
        A6 = graph.A6  # 6,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 60
        # (4, 3, 64, 25)
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A3, A6, residual=False)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A3, A6)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A3, A6)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A3, A6)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A3, A6, stride=2)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A3, A6)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A3, A6)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A3, A6, stride=2)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A3, A6)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A3, A6)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        # # Retrospect Model
        # self.first_tram = nn.Sequential(
        #     nn.AvgPool2d((4, 1)),
        #     nn.Conv2d(60, 240, 1),
        #     nn.BatchNorm2d(240),
        #     nn.ReLU()
        # )
        # self.second_tram = nn.Sequential(
        #     nn.AvgPool2d((2, 1)),
        #     nn.Conv2d(120, 240, 1),
        #     nn.BatchNorm2d(240),
        #     nn.ReLU()
        # )

    def forward(self, x):

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

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # N C T V M -> N M V C T -> N MVC T
        # print(x.shape)
        x = self.data_bn(x)  # batch_normalize
        # print(x.shape)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        # N MVC T -> N M V C T -> N M C T V -> NM C T V

        x = self.l1(x)  # (N*M, 60 , 64, 25)
        x = self.l2(x)  # (N*M, 60, 64, 25)
        x = self.l3(x)  # (N*M, 60, 64, 25)
        x = self.l4(x)  # (N*M, 60, 64, 25)
        x = self.l5(x)  # (N*M, 120, 32, 25)
        x = self.l6(x)  # (N*M, 120, 32, 25)
        x = self.l7(x)  # (N*M, 120, 32, 25)
        x = self.l8(x)  # (N*M, 240, 16, 25)
        x = self.l9(x)  # (N*M, 240, 16, 25)
        x = self.l10(x)  # (N*M, 240, 16, 25)

        # x2 = self.first_tram(x2)
        # x3 = self.second_tram(x3)
        # x = x + x2 + x3

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)  # N, M, C, T*V
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        log['image'] = {
            'A3': self.l10.gcn1.A3.detach(),
            'A6': self.l10.gcn2.A6.detach(),
        }

        log['pram'] = {
            'a_alpha': self.l10.gcn1.alpha.detach(),
        }

        return self.fc(x), log