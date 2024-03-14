import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import graph
from graph import ntu_rgb_d


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
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
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
                 dilations=[1,2,3,4],
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
            kernel_size = [kernel_size]*len(dilations)
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
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
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
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        # (4, 3, 64, 25)
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
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
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
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
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, len_temp, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        # (4, 3, 64, 25) , out=64
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = Temporal_GCN(out_channels, out_channels, len_temp=len_temp, kernel_size=kernel_size, stride=stride, dilations=dilations,
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
        spatial_out = self.gcn1(x)
        temporal_out, logs = self.tcn1(spatial_out)

        y = self.relu(temporal_out + self.residual(x))
        return y, logs


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        if num_point == 25:
            base_temp_length = 64
        else:
            base_temp_length = 52
        # (4, 3, 64, 25)
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, base_temp_length, residual=False, adaptive=adaptive)

        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, base_temp_length,  adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, base_temp_length, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, base_temp_length,  adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, base_temp_length, stride=2, adaptive=adaptive)

        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, base_temp_length // 2, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, base_temp_length // 2, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, base_temp_length // 2, stride=2, adaptive=adaptive)

        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, base_temp_length // 4, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, base_temp_length // 4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
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
        x, _ = self.l1(x)

        x, _ = self.l2(x)
        x, _ = self.l3(x)
        x, (temporal_A4, A4) = self.l4(x)

        x, _ = self.l5(x)
        x, _ = self.l6(x)
        x, (temporal_A7, A7) = self.l7(x)

        x, _ = self.l8(x)
        x, _ = self.l9(x)
        x, (temporal_A10, A10) = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        log = {}

        log['image'] = {
            'temporal_A4': temporal_A4,
            'temporal_A7': temporal_A7,
            'temporal_A10': temporal_A10,
            'A4': A4,
            'A7': A7,
            'A10': A10,
        }
        log['pram'] = {}

        return self.fc(x), log


class Temporal_GCN_block(nn.Module):
    def __init__(self, in_channel, rel_channel):
        super(Temporal_GCN_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel // rel_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channel, in_channel // rel_channel, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)  # (b, c', t, v)
        x2 = self.conv2(x)  # (b, c', t, v)
        b, c, t, v = x1.shape
        x1 = x1.permute(0, 2, 1, 3).reshape(b, t, -1)
        x2 = x2.permute(0, 1, 3, 2).reshape(b, -1, t)
        At = torch.matmul(x1, x2)
        At = F.softmax(At, dim=1)

        return At


class Temporal_GCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 len_temp,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2],
                 rel_channel=4,
                 n_head=4,
                 residual=False,
                 residual_kernel_size=1,
                 learnable=False):
        super(Temporal_GCN, self).__init__()
        assert in_channels % rel_channel == 0 and out_channels % n_head == 0 and n_head == len(dilations) + 2
        branch_channels = out_channels // n_head
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        self.n_head = n_head

        self.temporal_gcn_blocks = nn.ModuleList([Temporal_GCN_block(in_channels, rel_channel) for _ in range(n_head)])

        self.branches_convs = nn.ModuleList([
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
        self.branches_convs.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches_convs.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))


        if learnable:
            self.alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.alpha = 1.0
        self.A = Variable(self.temporal_real_adj(len_temp), requires_grad=False)
        self.WA = nn.Parameter(torch.ones((len_temp, len_temp), dtype=torch.float32))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    @classmethod
    def temporal_real_adj(cls, len_temp):
        adj = torch.zeros((len_temp, len_temp), dtype=torch.float)
        for i in range(len_temp):
            adj[i, 0:min(len_temp, i + 2)] = 1

        return adj

    def forward(self, x):
        # x.shape (b, c, t, v)
        res = self.residual(x)

        temporal_A = None
        A = self.A.cuda(x.get_device())
        WA = self.WA.cuda(x.get_device())
        A = A * WA
        branch_out = []
        for i in range(self.n_head):
            multi_At = self.temporal_gcn_blocks[i](x)
            temporal_A = self.alpha * multi_At + A
            z = torch.einsum('btg,bctv->bcgv', temporal_A, x)
            z = self.branches_convs[i](z)
            branch_out.append(z)

        out = torch.cat(branch_out, dim=1)
        out = out + res

        return out, (temporal_A.detach(), A.detach())


if __name__ == '__main__':
    model = Model(graph='graph.ntu_rgb_d.Graph')
    X = torch.randn(2, 3, 64, 25, 2)
    y = model(X)
    print(y.shape)