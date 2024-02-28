import torch
import torch.nn as nn
import math
from torch.nn.init import trunc_normal_


class WindowAttention_conv2qkv(nn.Module):
    def __init__(self, dim, window_size, num_heads, conv_size,
                 qkv_bias=False, attn_drop=0., proj_drop=0., pos_emb=True,
                 vit_hmap=False):
        super().__init__()
        self.attn2hmap_org = None
        self.vit_hmap=vit_hmap

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
        # if self.vit_hmap and not self.training:    # if self.trainval == 'val' or self.trainval == 'test':
        if self.vit_hmap:
            self.attn2hmap_org = attn  # 用作热图可视化
            self.attn2hmap_org = self.attn2hmap_org[0, :, 0, :].reshape(self.num_heads, -1)
            self.attn2hmap_org = self.attn2hmap_org.reshape(self.num_heads, self.window_size[1], self.window_size[0])

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)

        x = x.permute(0, 2, 1).view(B, C, self.window_size[0], self.window_size[1])
        x = self.up(x)
        x = self.norm1(x)
        return x

class LightNet_cat(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True, groups=2):
        super(LightNet_cat, self).__init__()
        self.shuffle = False
        self.groups = groups
        if inp_dim == out_dim and stride == 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False,
                          groups=math.gcd(inp_dim, out_dim)),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_dim)
            )
            self.skip = nn.Sequential()
        else:
            cent_dim = out_dim // 2
            self.shuffle = True
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp_dim, inp_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(inp_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp_dim, cent_dim, kernel_size=3, stride=(stride, stride), padding=1, bias=False,
                          groups=math.gcd(inp_dim, cent_dim)),
                nn.BatchNorm2d(cent_dim),
                nn.Conv2d(cent_dim, cent_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(cent_dim),
                nn.ReLU(inplace=True)
            )
            self.skip = nn.Sequential(
                nn.Conv2d(inp_dim, cent_dim, kernel_size=3, stride=(stride, stride), padding=1, bias=False,
                          groups=math.gcd(inp_dim, cent_dim)),
                nn.BatchNorm2d(cent_dim),
                nn.Conv2d(cent_dim, cent_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(cent_dim),
                nn.ReLU(inplace=True)
            )
        self.last = nn.ReLU(inplace=True)

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % groups == 0)
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        branch = self.branch1(x)
        skip = self.skip(x)
        if self.shuffle:
            out = self.channel_shuffle(torch.cat((branch, skip), 1), self.groups)
        else:
            out = self.last(branch + skip)
        return out