import torch
import torch.nn as nn
from modules import LightNet_cat as residual
from modules import WindowAttention_conv2qkv as Attention


class attn_model(nn.Module):
    def __init__(self, dim, window_hw, dropout_factor=0.1,
                 vit_hmap=False):
        super(attn_model, self).__init__()
        self.attn2hmap_org_1 = None
        self.window_hw = window_hw
        self.vit_hmap = vit_hmap

        self.dropout_factor = dropout_factor
        self.dropout = nn.Dropout(dropout_factor)
        self.self_attn = Attention(dim=dim, window_size=window_hw,
                                   conv_size=3, num_heads=8,
                                   attn_drop=self.dropout_factor, pos_emb=True,
                                   vit_hmap=vit_hmap)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        attn = self.self_attn(x)

        if self.vit_hmap:
            self.attn2hmap_org_1 = self.self_attn.attn2hmap_org

        out = self.dropout(attn) + x
        out = self.norm2(out)
        return out


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


# inp_dim -> out_dim -> ... -> out_dim
def make_layer(kernel_size, inp_dim, out_dim, modules, layer, stride=1):
    layers = [layer(kernel_size, inp_dim, out_dim, stride=stride)]
    layers += [layer(kernel_size, out_dim, out_dim) for _ in range(modules - 1)]
    return nn.Sequential(*layers)


# def make_attn_layer(kernel_size, inp_dim, out_dim,
#                     modules, layer, stride=1, window_hw=64):
#     layers = [layer(kernel_size, inp_dim, out_dim, stride=stride)]
#     if window_hw == 64 or window_hw == 16:
#         layers += [attn_model(dim=out_dim, window_hw=window_hw)]
#     layers += [layer(kernel_size, out_dim, out_dim) for _ in range(modules - 1)]
#     return nn.Sequential(*layers)

# inp_dim -> out_dim -> ... -> out_dim
class make_attn_layer(nn.Module):
    def __init__(self, kernel_size, inp_dim, out_dim,
                 modules, layer, stride=1, window_hw=64,
                 vit_hmap=False):
        super(make_attn_layer, self).__init__()
        self.attn2hmap_org_2 = None
        self.vit_hmap=vit_hmap
        self.window_hw=window_hw

        self.layer1 = layer(kernel_size, inp_dim, out_dim, stride=stride)
        if window_hw == 64:
            self.layer2 = attn_model(dim=out_dim, window_hw=window_hw, vit_hmap=vit_hmap)
        elif window_hw == 16:
            self.layer2 = attn_model(dim=out_dim, window_hw=window_hw)
        else:
            self.layer2 = nn.Sequential()
        layer3 = [layer(kernel_size, out_dim, out_dim) for _ in range(modules - 1)]
        self.layer3 = nn.Sequential(*layer3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        if self.window_hw == 64 and self.vit_hmap:
            self.attn2hmap_org_2 = self.layer2.attn2hmap_org_1
        x = self.layer3(x)
        return x


# inp_dim -> inp_dim -> ... -> inp_dim -> out_dim
def make_layer_revr(kernel_size, inp_dim, out_dim, modules, layer):
    layers = [layer(kernel_size, inp_dim, inp_dim) for _ in range(modules - 1)]
    layers.append(layer(kernel_size, inp_dim, out_dim))
    return nn.Sequential(*layers)


# key point layer
def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False),
                         nn.Conv2d(curr_dim, out_dim, (1, 1)))


class kp_module(nn.Module):
    def __init__(self, n, dims, modules, window_hw=128,
                 vit_hmap=False):
        super(kp_module, self).__init__()
        self.attn2hmap_org_3 = None
        self.vit_hmap = vit_hmap

        self.n = n
        curr_modules = modules[0]
        next_modules = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        # curr_mod x residual，curr_dim -> curr_dim -> ... -> curr_dim
        self.top = make_layer(3, curr_dim, curr_dim, curr_modules, layer=residual)
        self.down = nn.Sequential()
        # curr_mod x residual，curr_dim -> next_dim -> ... -> next_dim
        self.low1 = make_attn_layer(3, curr_dim, next_dim, curr_modules,
                                    layer=residual, stride=2, window_hw=window_hw // 2,
                                    vit_hmap=vit_hmap)
        # next_mod x residual，next_dim -> next_dim -> ... -> next_dim
        if self.n > 1:
            self.low2 = kp_module(n - 1, dims[1:], modules[1:], window_hw=window_hw // 2,
                                  vit_hmap=vit_hmap)
        else:
            self.low2 = make_layer(3, next_dim, next_dim, next_modules, layer=residual)
        # curr_mod x residual，next_dim -> next_dim -> ... -> next_dim -> curr_dim
        self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_modules, layer=residual)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = self.top(x)
        down = self.down(x)
        low1 = self.low1(down)
        if self.attn2hmap_org_3 == None and self.vit_hmap:
            self.attn2hmap_org_3 = self.low1.attn2hmap_org_2
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up(low3)
        return up1 + up2


class exkp(nn.Module):
    def __init__(self, n, nstack, dims, modules, cnv_dim=256, num_classes=8,
                 vit_hmap=False):
        super(exkp, self).__init__()
        self.attn2hmap_org_4 = None
        self.vit_hmap = vit_hmap

        self.nstack = nstack
        self.num_classes = num_classes
        curr_dim = dims[0]
        self.pre = nn.Sequential(convolution(7, 3, 128, stride=2),
                                 residual(3, 128, curr_dim, stride=2))

        self.kps = nn.ModuleList([kp_module(n, dims, modules, vit_hmap=vit_hmap) for _ in range(nstack)])

        self.cnvs = nn.ModuleList([convolution(3, curr_dim, cnv_dim) for _ in range(nstack)])

        self.inters = nn.ModuleList([residual(3, curr_dim, curr_dim) for _ in range(nstack - 1)])

        self.inters_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                                                    nn.BatchNorm2d(curr_dim))
                                      for _ in range(nstack - 1)])
        self.cnvs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                                                  nn.BatchNorm2d(curr_dim))
                                    for _ in range(nstack - 1)])
        # heatmap layers
        self.hmap = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])
        for hmap in self.hmap:
            hmap[-1].bias.data.fill_(-2.19)

        # regression layers
        self.regs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
        self.w_h_ = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        inter = self.pre(image)

        outs = []
        for ind in range(self.nstack):
            kp = self.kps[ind](inter)
            if self.attn2hmap_org_4==None and self.vit_hmap and not self.training:
                self.attn2hmap_org_4 = self.kps[ind].attn2hmap_org_3
            cnv = self.cnvs[ind](kp)

            if self.training or ind == self.nstack - 1:
                outs.append([self.hmap[ind](cnv), self.regs[ind](cnv), self.w_h_[ind](cnv)])

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs


get_hg_attn_light = \
    {'large_hg_attn_light':
         exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4], vit_hmap=True),
     'small_hg_attn_light':
         exkp(n=5, nstack=1, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4], vit_hmap=True)}

if __name__ == '__main__':
    def count_parameters(model):
        num_paras = [v.numel() / 1e6 for k, v in model.named_parameters() if 'aux' not in k]
        print("Total num of param = %f M" % sum(num_paras))


    def count_flops(model, input_size=384):
        flops = []
        handles = []

        def conv_hook(self, input, output):
            flops.append(output.shape[2] ** 2 *
                         self.kernel_size[0] ** 2 *
                         self.in_channels *
                         self.out_channels /
                         self.groups / 1e6)

        def fc_hook(self, input, output):
            flops.append(self.in_features * self.out_features / 1e6)

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                handles.append(m.register_forward_hook(conv_hook))
            if isinstance(m, nn.Linear):
                handles.append(m.register_forward_hook(fc_hook))

        with torch.no_grad():
            _ = model(torch.randn(1, 3, input_size, input_size))
        print("Total FLOPs = %f M" % sum(flops))

        for h in handles:
            h.remove()


    net = get_hg_attn_light['small_hg_attn_light']
    x = torch.randn(1, 3, 512, 512)
    net.training = False

    # count_parameters(net)
    # count_flops(net, input_size=512)

    with torch.no_grad():
        y = net(x)
        print(net.attn2hmap_org_4.shape)
