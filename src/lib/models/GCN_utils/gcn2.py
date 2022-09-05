import math
import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .graph import Graph


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
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class change_channels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(change_channels, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        # print(self.A.shape)
        self.num_subset = num_subset
        self.alpha = nn.Parameter(torch.zeros(1))

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        # self.bn = nn.BatchNorm2d(out_channels)
        self.bn = nn.LayerNorm(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        B, C, HW, V = x.size()  # B, C, H*W, V
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 2, 3, 1).contiguous().view(B*HW, V, self.inter_c)
            A2 = self.conv_b[i](x).permute(0, 2, 1, 3).contiguous().view(B*HW, self.inter_c, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # B*HW V V

            A1 = A1 * self.alpha + A[i]

            A2 = x.permute(0, 2, 1, 3).contiguous().view(B * HW, C, V)
            z = self.conv_d[i](torch.bmm(A2, A1).view(B, HW, C, V).permute(0, 2, 1, 3).contiguous())
            # z=x
            y = z + y if y is not None else z

        y = self.bn(y.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous()
        y += self.down(x)
        return self.relu(y)


class GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, residual=True):
        super(GCN_unit, self).__init__()
        self.gcn = unit_gcn(in_channels, out_channels, A)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = change_channels(in_channels, out_channels)

    def forward(self, x):
        x = self.gcn(x) + self.residual(x)
        return self.relu(x)


class GCN(nn.Module):
    def __init__(self, in_channels, num_point=7):
        super(GCN, self).__init__()

        self.graph = Graph()
        A = self.graph.A

        # self.data_bn = nn.BatchNorm1d(in_channels * num_point)
        self.data_bn = nn.LayerNorm(in_channels)
        # bn_init(self.data_bn, 1)

        # self.l1 = GCN_unit(in_channels, 64, A, residual=False)
        self.l1 = GCN_unit(in_channels, 64, A, residual=True)
        # self.l2 = GCN_unit(64, 64, A)
        # self.l3 = GCN_unit(64, 64, A)

    def forward(self, x):
        V, B, C, H, W = x.size()
        x = x.flatten(3).permute(0, 1, 3, 2).contiguous()   # ??  (v,b,hw,c)
        x = self.data_bn(x)
        x = x.permute(1, 3, 2, 0).contiguous()   # (B,C,HW,V)

        x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)

        c_new=x.size(1)
        x = x.view(B,c_new,H,W,V).permute(4, 0, 1, 2, 3).contiguous()
        return x

    # def forward(self, x):
    #     V, B, C, H, W = x.size()
    #     x = x.permute(1, 0, 2, 3, 4).contiguous().view(B, V*C, H*W)   # ??  (B,V,C,H,W)
    #     x = self.data_bn(x)
    #     x = x.view(B, V, C, H, W).permute(0, 2, 3, 4, 1).contiguous().view(B, C, H*W, V)   # (B,C,H,W,V)

    #     x = self.l1(x)
    #     # x = self.l2(x)
    #     # x = self.l3(x)

    #     c_new=x.size(1)
    #     x = x.view(B,c_new,H,W,V).permute(4, 0, 1, 2, 3).contiguous()
    #     return x


# torch.set_default_tensor_type(torch.DoubleTensor)
# x = torch.tensor(np.random.random((7,10,3,5,6))).to(device='cuda:1')
# model = GCN(in_channels=3).to(device='cuda:1')
# out = model(x)