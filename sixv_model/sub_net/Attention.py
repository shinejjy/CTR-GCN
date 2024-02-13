import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from inspect import isfunction


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class Cross_MultiAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads, dropout=0, gated_ff=True):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.ff = FeedForward(emb_dim, dropout=dropout, glu=gated_ff)

    def forward(self, x_t, pad_mask=None):
        '''

        :param x_t: [batch_size, c, t, num_joint]
        :param x_j: [batch_size, num_joint, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        b, c, t, v = x_t.shape
        x_j = x_t.mean(-2).permute(0, 2, 1)
        x_t = x_t.view(b, t * v, c)  # [3, 60*25, 120]

        Q = self.Wq(self.norm1(x_t))  # [batch_size, t, emb_dim] = [3, 60*25, 120]
        K = self.Wk(x_j)  # [batch_szie, num_joint, emb_dim] = [3, 25, 120]
        V = self.Wv(x_j)

        Q = Q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, t * v // num_heads, depth]
        K = K.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, num_joint // num_heads, depth]
        V = V.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, num_joint // num_heads, depth]

        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        x_att = torch.einsum('bnij, bnjd -> bnid', att_weights, V)  # [batch_size, num_heads, t * v // num_heads, emb_dim]
        x_att = x_att.view(b, -1, self.emb_dim)  # [batch_size, t * v, emb_dim]

        x_t = x_att + x_t

        x_t = self.ff(self.norm2(x_t)) + x_t
        x_t = x_t.view(b, t, v, c).permute(0, 3, 1, 2).contiguous()

        return x_t


class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        # 维度必须能被num_head 整除
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        # 定义线性变换矩阵
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att


if __name__ == '__main__':
    # 设置随机种子，保证每次运行结果一致
    torch.manual_seed(42)

    # 定义模块
    in_channels = 120
    emb_dim = 120
    num_heads = 4
    attention_module = Cross_MultiAttention(in_channels, emb_dim, num_heads)

    # 生成随机输入数据
    batch_size = 3
    t = 60
    num_joint = 25
    x_t = torch.randn(batch_size, t, emb_dim)
    x_j = torch.randn(batch_size, num_joint, emb_dim)

    # # 随机生成一个 pad_mask，假设 seq_len = t
    # pad_mask = torch.randint(2, size=(batch_size, t, t), dtype=torch.bool)

    # 模块前向传播
    output, attention_weights = attention_module(x_t, x_j)

    # 打印输出形状
    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)

    # 检查 softmax 是否正常工作（每行和为1）
    print("Attention weights sum along the last dimension:", attention_weights.sum(dim=-1))

    # 打印输出和注意力权重
    print("Output:", output.shape)
    print("Attention weights:", attention_weights.shape)
