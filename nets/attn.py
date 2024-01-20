import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class WindowAttention_conv2qkv(nn.Module):
    def __init__(self, dim, window_size, num_heads, conv_size,
                 qkv_bias=False, attn_drop=0., proj_drop=0.,
                 pos_emb=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.pos_emb = pos_emb
        self.conv_size = conv_size
        if conv_size == 1:
            padding = 0
            stride = 1
            self.window_size = (window_size, window_size)
            self.up = nn.Sequential()
        elif conv_size == 3:
            padding = 1
            stride = 2
            self.window_size = (window_size // 2, window_size // 2)
            self.up = nn.Upsample(scale_factor=2)
        else:
            print('conv_size is not 1 or 3', conv_size)

        self.qkv = nn.Conv2d(dim, dim * 3, (conv_size, conv_size), stride=(stride, stride),
                             padding=(padding, padding), bias=qkv_bias)
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])  # 0,1,...,window_size[0]
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm1 = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, height, width = x.size()
        N = self.window_size[0] * self.window_size[1]

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.pos_emb:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)

        x = x.permute(0, 2, 1).view(B, C, self.window_size[0], self.window_size[1])
        x = self.up(x)
        x = self.norm1(x)
        return x


class WindowAttention_gfc(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=False, attn_drop=0., proj_drop=0.,
                 pos_emb=False):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.pos_emb = pos_emb

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])  # 0,1,...,window_size[0]
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        groups = self.window_size[1]
        self.dim_ch2 = self.window_size[0] * self.window_size[1]
        self.q = nn.Conv1d(dim, dim, 1, groups=groups, bias=qkv_bias)
        self.k = nn.Conv1d(dim, dim, 1, groups=groups, bias=qkv_bias)
        self.v = nn.Conv1d(dim, dim, 1, groups=groups, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = nn.BatchNorm2d(dim)

    def forward(self, x):
        B_, C, height, width = x.size()
        N = self.window_size[0] * self.window_size[1]

        x = x.view(B_, C, N)
        q = self.q(x)
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.pos_emb:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(x)

        x = x.permute(0, 2, 1).view(B_, C, self.window_size[0], self.window_size[1])
        x = self.norm1(x)
        return x