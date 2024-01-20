import torch
import math
import torch.nn as nn


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
