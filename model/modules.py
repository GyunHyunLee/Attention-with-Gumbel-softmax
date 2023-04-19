import math

import torch
import numpy as np

import torch.nn.functional as F
from torch import nn, einsum
from torch.autograd import Variable

from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from einops import rearrange, repeat


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(UnitTCN, self).__init__()
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


class SelfAttention(nn.Module):
    def __init__(self, in_channels, in_frames, hidden_dim, n_heads):
        super(SelfAttention, self).__init__()
        self.scale = hidden_dim ** -0.5
        inner_dim = hidden_dim * n_heads
        self.to_qk = nn.Linear(in_channels, inner_dim * 2)
        # self.to_qk_spa = nn.Linear(int(in_channels*in_frames), inner_dim*2)
        # self.to_qk_tem = nn.Linear(in_channels*25, inner_dim * 2)
        self.n_heads = n_heads
        self.ln = nn.LayerNorm(in_channels)
        nn.init.normal_(self.to_qk.weight, 0, 1)
        # self.ln_spa = nn.LayerNorm(int(in_channels*in_frames))
        # self.ln_tem = nn.LayerNorm(in_channels*25)
        # nn.init.normal_(self.to_qk_spa.weight, 0, 1)
        # nn.init.normal_(self.to_qk_tem.weight, 0, 1)

    def forward(self, x, tau, spa_tem):
        y = rearrange(x, 'n c t v -> n t v c').contiguous()
        y = self.ln(y)
        y = self.to_qk(y)
        qk = y.chunk(2, dim=-1)
        # q, k = map(lambda t: rearrange(t, 'b t v (h d) -> (b t) h v d', h=self.n_heads), qk)

        if spa_tem == 'spatial':
            # y = rearrange(x, 'n c t v -> n v (c t)')
            # y = self.ln_spa(y)
            # y = self.to_qk_spa(y)
            # qk = y.chunk(2, dim=-1)
            # q, k = map(lambda t: rearrange(t, 'b v (h d) -> b h v d', h=self.n_heads), qk)
            q, k = map(lambda t: rearrange(t, 'b t v (h d) -> b h v (t d)', h=self.n_heads), qk)

            # attention
            dots = einsum('b h i d, b h j d -> b h i j', q, k)*self.scale
            attn = F.gumbel_softmax(dots, tau=tau, hard=False, dim=-1).cuda()
        elif spa_tem == 'temporal':
            # y = rearrange(x, 'n c t v -> n t (v c)').contiguous()
            # y = self.ln_tem(y)
            # y = self.to_qk_tem(y)
            # qk = y.chunk(2, dim=-1)
            # q, k = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=self.n_heads), qk)
            q, k = map(lambda t: rearrange(t, 'b t v (h d) -> b h t (v d)', h=self.n_heads), qk)

            # attention
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            attn = dots.softmax(dim=-1).float()

        return attn

class SA_GC(nn.Module):
    def __init__(self, in_channels, in_frames, out_channels, A):
        super(SA_GC, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head = A.shape[0]
        self.shared_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)

        rel_channels = in_channels // 8
        self.attn = SelfAttention(in_channels, in_frames, rel_channels, self.num_head)


    def forward(self, x, epoch, spa_attn=None, tem_attn=None):
        if epoch < 20:
            gumbel_tau = 1000
        elif epoch < 30:
            gumbel_tau = 100
        else:
            gumbel_tau = 10

        N, C, T, V = x.size()

        # x_weighted = None
        # if tem_attn is None:
        #     tem_attn = self.attn(x, gumbel_tau, 'temporal')
        # for h in range(self.num_head):
        #     A_h = tem_attn[:, h, :, :] # ntt
        #     A_h = A_h.expand(V, N, T, T)
        #     A_h = rearrange(A_h, 'v n u t -> (n v) u t').contiguous()
        #     feature = rearrange(x, 'n c t v -> (n v) t c')
        #     z = A_h@feature
        #     z = rearrange(z, '(n v) t c-> n c t v', v=V).contiguous()
        #     x_weighted = z + x_weighted if x_weighted is not None else z

        out = None
        if spa_attn is None:
            spa_attn = self.attn(x, gumbel_tau, 'spatial')
            # spa_attn = self.attn(x_weighted, gumbel_tau, 'spatial')
        for h in range(self.num_head):
            A_h = spa_attn[:, h, :, :] # nvv
            A_h = A_h.expand(T, N, V, V)
            A_h = rearrange(A_h, 't n w v -> (n t) w v').contiguous()
            feature = rearrange(x, 'n c t v -> (n t) v c')
            z = A_h@feature
            z = rearrange(z, '(n t) v c-> n c t v', t=T).contiguous()
            z = self.conv_d[h](z)
            out = z + out if out is not None else z

        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)

        return out

class EncodingBlock(nn.Module):
    def __init__(self, in_channels, in_frames, out_channels, A, stride=1, residual=True):
        super(EncodingBlock, self).__init__()
        self.agcn = SA_GC(in_channels, in_frames, out_channels, A)
        self.tcn = MS_TCN(out_channels, out_channels, kernel_size=5, stride=stride,
                         dilations=[1, 2], residual=False)

        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, epoch, attn=None):
        y = self.relu(self.tcn(self.agcn(x, epoch, attn)) + self.residual(x))
        return y

