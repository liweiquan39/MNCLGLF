import torch
import torch.nn as nn
from core.utils import dist as link
from core.utils.misc import get_bn

from itertools import combinations
from core.utils.nn_memory_bank import NNMemoryBankModule
device = torch.device("cuda:1")


# class projection_MLP(nn.Module):
#     def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
#         super(projection_MLP, self).__init__()
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#
#         self.linear1 = nn.Linear(512, 2048)
#         self.bn1 = BN(2048)
#
#         self.linear2 = nn.Linear(2048, 2048)
#         # self.bn2 = BN(hidden_dim, affine=True)
#         #
#         # self.linear3 = nn.Linear(hidden_dim, out_dim)
#         # self.bn3 = BN(out_dim, affine=True)
#
#         self.activation = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = torch.flatten(x, 1)
#
#         x = self.linear1(x)
#         x = self.bn1(x)
#         x = self.activation(x)
#
#         x = self.linear2(x)
#         # x = self.bn2(x)
#         # x = self.activation(x)
#         #
#         # x = self.linear3(x)
#         # x = self.bn3(x)
#
#         return x
#
#
# class prediction_MLP(nn.Module):
#     def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
#         super(prediction_MLP, self).__init__()
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#
#         self.linear1 = nn.Linear(2048, 512)
#         self.bn1 = BN(512)
#
#         self.layer2 = nn.Linear(512, 2048)
#
#         self.activation = nn.ReLU(inplace=True)
#
#     def forward(self, input):
#         # layer 1
#         x = self.linear1(input)
#         x = self.bn1(x)
#         hidden = self.activation(x)
#         # N C
#         x = self.layer2(hidden)
#         return x

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super(projection_MLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = BN(hidden_dim)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = BN(hidden_dim, affine=True)

        self.linear3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = BN(out_dim, affine=True)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.linear3(x)
        x = self.bn3(x)

        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super(prediction_MLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = BN(hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, out_dim)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        # layer 1
        x = self.linear1(input)
        x = self.bn1(x)
        hidden = self.activation(x)
        # N C
        x = self.layer2(hidden)
        return x


class FastMoCo(nn.Module):
    def __init__(self, backbone, bn, projector=None, predictor=None, ema=False, m=0.99, backbone_ema=None, arch='comb_patch',
                 split_num=2, combs=0):
        super(FastMoCo, self).__init__()
        projector = {} if projector is None else projector
        predictor = {} if predictor is None else predictor
        global BN

        BN = get_bn(bn)

        self.world_size = link.get_world_size()
        self.rank = link.get_rank()

        self.split_num = split_num
        self.m = m
        self.ema = ema
        self.arch = arch

        self.combs = combs

        self.dim_fc = dim_fc = backbone.fc.weight.shape[1]
        self.memory_bank = NNMemoryBankModule(size=16384).to(device)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1]).to(device)
        _projector = projection_MLP(in_dim=dim_fc, **projector).to(device)
        self.encoder = nn.Sequential(
            self.backbone,
            _projector
        ).to(device)

        if self.ema:
            self.bbone_ema = nn.Sequential(*list(backbone_ema.children())[:-1]).to(device)
            _projector_ema = projection_MLP(in_dim=dim_fc, **projector).to(device)
            self.encoder_ema = nn.Sequential(
                self.bbone_ema,
                _projector_ema
            ).to(device)

        self.predictor = prediction_MLP(**predictor).to(device)

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_q, param_t in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
            param_q.to(device)
            param_t.to(device)
            param_t.data = param_t.data.mul_(self.m).add_(param_q.data, alpha=1. - self.m).to(device)

    def _local_split(self, x):     # NxCxHxW --> 4NxCx(H/2)x(W/2)
        _side_indent = x.size(2) // self.split_num, x.size(3) // self.split_num
        cols = x.split(_side_indent[1], dim=3)
        xs_1 = []
        xs = []

        for _x in cols:
            # print(_x.shape)
            xs += _x.split(_side_indent[0], dim=2)

        # for i in range(len(xs)):
        #     if xs[i].shape[2] == 74 and xs[i].shape[3] == 74:
        #         # print(xs[i].shape)
        #         xs_1 += xs[i]
        #
        # for i in range(len(xs_1)):
        #     xs_1[i] = torch.unsqueeze(xs_1[i], dim=0)
        # print(xs[0].shape)
        # x = torch.cat(xs_1, dim=0).to(device)
        x = torch.cat(xs, dim=0).to(device)
        # print(x.shape)
        # assert False
        return x

    def forward(self, input):
        x1, x2, x3 = torch.split(input, [3, 3, 3], dim=1)

        f, h = self.encoder.to(device), self.predictor.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        if self.arch == 'comb_patch':
            z1 = f(x1).to(device)
            z2 = f(x2).to(device)
            z3 = f(x3).to(device)
            p1 = h(z1).to(device)
            p2 = h(z2).to(device)
            p3 = h(z3).to(device)

        else:
            raise NotImplementedError
        # --> NxC, NxC
        if self.ema:
            with torch.no_grad():
                self._momentum_update_target_encoder()

                x1_in_form = self._local_split(x1).to(device)
                x2_in_form = self._local_split(x2).to(device)
                k1_pre, k2_pre = f[0](x1_in_form).to(device), f[0](x2_in_form).to(device)
                k1_splits = list(k1_pre.split(k1_pre.size(0) // self.split_num ** 2, dim=0))  # 4b x c x
                k2_splits = list(k2_pre.split(k2_pre.size(0) // self.split_num ** 2, dim=0))
                k1_orthmix = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(k1_splits, r=self.combs)))), dim=0).to(device)  # 6 of 2combs / 4 of 3combs
                k2_orthmix = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(k2_splits, r=self.combs)))), dim=0).to(device)  # 6 of 2combs / 4 of 3combs
                # 两两组合、或三三组合，然后把每个组合的特征值直接求平均值
                k1, k2 = f[1](k1_orthmix).to(device), f[1](k2_orthmix).to(device)
                # k1, k2 = f(x1).to(device), f(x2).to(device)

                j1 = self.encoder_ema(x1).to(device)
                j3 = self.encoder_ema(x3).to(device)
                j1 = self.memory_bank(j1, update=False).to(device)
                j3 = self.memory_bank(j3, update=True).to(device)

        else:
            assert False
            # with torch.no_grad():
            #     z1, z3 = f(x1), f(x3)

        return p1, k1, p2, k2, p3, j3, j1
        # 两张图片轮换走在线分支和目标分支，p是online分支，z是target分支
        # 1和2构成原图和分割的对比，1和3构成原图和支持集的对比
