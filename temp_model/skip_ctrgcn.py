import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F


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
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()
        # (4, 3, 64, 25) , out=64
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
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
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A3  # 3,25,25
        A6 = self.graph.A6_ones

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        if num_point == 25:
            len_temp = 64
        else:
            len_temp = 52
        # (4, 3, 64, 25)
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)

        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel, A, stride=2, adaptive=adaptive)
        self.sst_gcn_bottom = SST_GCN(base_channel, base_channel * 2, num_point, len_temp, A6)

        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, stride=2, adaptive=adaptive)
        self.sst_gcn_mid = SST_GCN(base_channel * 2, base_channel * 4, num_point, len_temp // 2, A6)

        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
        # self.sst_gcn_top = SST_GCN(base_channel * 4, base_channel * 4, num_point, len_temp // 4, A6)

        self.fc = nn.Linear(base_channel * 4, num_class)
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
        x = self.l1(x)
        g1 = x

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        g2, (As_at2, At_at2, As_act2, At_act2) = self.sst_gcn_bottom(g1, x)

        x = self.l6(g2)
        x = self.l7(x)
        x = self.l8(x)

        g3, (As_at3, At_at3, As_act3, At_act3) = self.sst_gcn_mid(g2, x)

        x = self.l9(g3)
        x = self.l10(x)

        # g4, (_) = self.sst_gcn_top(g3)
        # x = g4 + x

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        logs = {}
        logs['image'] = {
            'As_at2': As_at2,
            'At_at2': At_at2,
            'As_act2': As_act2,
            'At_act2': At_act2,
            'As_at3': As_at3,
            'At_at3': At_at3,
            'As_act3': As_act3,
            'At_act3': At_act3
        }
        logs['pram'] = {}

        return self.fc(x), logs


class SST_GCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, len_temp, rel_reduction=8):
        super(SST_GCN_block, self).__init__()
        assert in_channels % rel_reduction == 0
        self.spatial_conv1 = nn.Conv2d(in_channels, in_channels // rel_reduction, kernel_size=1)
        self.spatial_conv2 = nn.Conv2d(in_channels, in_channels // rel_reduction, kernel_size=1)

        self.temporal_conv1 = nn.Conv2d(in_channels, in_channels // rel_reduction, kernel_size=1)
        self.temporal_conv2 = nn.Conv2d(in_channels, in_channels // rel_reduction, kernel_size=1)

        self.stride_pro = unit_tcn(in_channels, in_channels, kernel_size=1, stride=2)

        self.W1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.W2 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

        self.act_s = nn.Parameter(torch.ones((num_point, num_point), dtype=torch.float32), requires_grad=True)
        self.act_t = nn.Parameter(torch.ones((len_temp, len_temp), dtype=torch.float32), requires_grad=True)

        self.sigmoid = nn.Sigmoid()

        self.tanh_s = nn.Tanh()
        self.tanh_t = nn.Tanh()

        self.relu_s = nn.ReLU()
        self.relu_t = nn.ReLU()

        self.bn = nn.BatchNorm2d(out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def get_As(self, x):
        b, c, t, v = x.shape
        spatial_feature = torch.mean(x, dim=-2, keepdim=True)
        spatial_feature1 = self.spatial_conv1(spatial_feature)  # (b, 1, 1, v)
        spatial_feature2 = self.spatial_conv2(spatial_feature)  # (b, 1, 1, v)
        spatial_feature1 = spatial_feature1.permute(0, 3, 1, 2).reshape(b, v, -1)
        spatial_feature2 = spatial_feature2.permute(0, 1, 2, 3).reshape(b, -1, v)

        As = torch.matmul(spatial_feature1, spatial_feature2)
        As = self.relu_s(self.tanh_s(As))
        # As = F.softmax(As, dim=2)  # (b, v, v)

        return As

    def get_At(self, x):
        b, c, t, v = x.shape
        temporal_feature = torch.mean(x, dim=-1, keepdim=True)
        temporal_feature1 = self.temporal_conv1(temporal_feature)  # (b, 1, t, 1)
        temporal_feature2 = self.temporal_conv2(temporal_feature)  # (b, 1, t, 1)
        temporal_feature1 = temporal_feature1.permute(0, 2, 1, 3).reshape(b, t, -1)
        temporal_feature2 = temporal_feature2.permute(0, 1, 3, 2).reshape(b, -1, t)

        At = torch.matmul(temporal_feature1, temporal_feature2)
        At = self.relu_t(self.tanh_t(At))
        # At = F.softmax(At, dim=1)  # (b, t, t)

        return At

    def forward(self, g1, h2, As_r, At_r):
        As_at = self.get_As(g1)
        At_at = self.get_At(g1)
        As_act = As_r * self.act_s
        At_act = At_r * self.act_t
        As = As_at + As_act
        At = At_at + At_act
        nu1 = torch.einsum('buv,bctv->bctu', As, g1)
        nu1 = torch.einsum('bctu,bty->bcyu', nu1, At)
        nu1 = self.stride_pro(nu1)
        if h2 is None:
            h2 = self.stride_pro(g1)
        h_star2 = torch.cat([nu1, h2], dim=1)
        g2 = self.W1(h_star2) * self.sigmoid(self.W2(h_star2))
        g2 = self.bn(g2)

        return g2, (As_at.detach(), At_at.detach(), As_act.detach(), At_act.detach())


class SST_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, len_temp, As, rel_reduction=8, adaptive=False):
        super(SST_GCN, self).__init__()

        self.adaptive = adaptive

        self.sst_gcn_blocks = nn.ModuleList([
            SST_GCN_block(in_channels, out_channels, num_point, len_temp, rel_reduction=rel_reduction)
            for _ in range(6)
        ])
        if not adaptive:
            self.As = Variable(torch.from_numpy(As), requires_grad=False)
            self.At = Variable(torch.from_numpy(self.temporal_real_adj(len_temp, 6)), requires_grad=True)
        else:
            self.As = nn.Parameter(torch.from_numpy(As), requires_grad=False)
            self.At = nn.Parameter(torch.from_numpy(self.temporal_real_adj(len_temp, 6)), requires_grad=True)

        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, g1, h2=None):
        if not self.adaptive:
            As = self.As.cuda(g1.get_device())
            At = self.At.cuda(g1.get_device())
        else:
            As = self.As
            At = self.At

        # g2 = None
        # for i in range(6):
        #     z = self.sst_gcn_blocks[i](g1, h2, As[i], At[i])
        #     g2 = g2 + z if g2 is not None else z

        z1, A1 = self.sst_gcn_blocks[0](g1, h2, As[0], At[0])
        z2, A2 = self.sst_gcn_blocks[1](g1, h2, As[1], At[1])
        z3, A3 = self.sst_gcn_blocks[2](g1, h2, As[2], At[2])
        z4, A4 = self.sst_gcn_blocks[3](g1, h2, As[3], At[3])
        z5, A5 = self.sst_gcn_blocks[4](g1, h2, As[4], At[4])
        z6, A6 = self.sst_gcn_blocks[5](g1, h2, As[5], At[5])

        g2 = z1 + z2 + z3 + z4 + z5 + z6

        g2 = self.bn(g2)
        g2 = self.relu(g2)

        return g2, A1

    @staticmethod
    def temporal_real_adj(len_temp, num_head):
        adj = np.zeros((num_head, len_temp, len_temp), dtype=np.float32)
        for o in range(num_head):
            for i in range(len_temp):
                adj[o, 0:min(len_temp, i + 2), i] = 1

        return adj


if __name__ == '__main__':
    model = unit_gcn(1, 1, np.ones((6, 25, 25)))
