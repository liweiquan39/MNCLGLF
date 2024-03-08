import torch
import torch.nn.functional as F
import torch.distributed as dist

from core.utils import dist as link
from core.utils.ntx_ent_loss import NTXentLoss
from torch.nn.modules.loss import _Loss


class InfoNCE(_Loss):
    def __init__(self, temperature):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    @staticmethod
    def cosine_similarity(p, z):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)
        # [N E] [N E] -> [N] -> [1]
        return (p * z).sum(dim=1).mean()  # dot product & batch coeff normalization

    def loss1(self, p_gather, k):
        # [N, E]
        k = k / k.norm(dim=-1, keepdim=True)

        offset = link.get_rank() * p_gather.shape[0]
        labels = torch.arange(offset, offset + p_gather.shape[0], dtype=torch.long).cuda()
        p_z_m = p_gather.mm(k.T) / self.temperature  # [N_local, N]

        return F.cross_entropy(p_z_m, labels)

    def loss2(self, j, p):
        return NTXentLoss(temperature=1)(j, p)

    def forward(self, p1, k1, p2, k2, p3, j3, j1):
        a = 6
        b = 0.5

        k1_s = k1.split(p2.size(0), dim=0)
        k2_s = k2.split(p1.size(0), dim=0)

        p1 = p1 / p1.norm(dim=-1, keepdim=True)
        p2 = p2 / p2.norm(dim=-1, keepdim=True)

        p1_gather = concat_all_gather(p1.detach())
        p2_gather = concat_all_gather(p2.detach())

        loss = 0
        for k in k1_s:
            loss = loss + self.loss1(p2_gather, k)
        for k in k2_s:
            loss = loss + self.loss1(p1_gather, k)

        loss1 = loss / (len(k1_s) + len(k2_s))

        # p1_s = p1.split(k2.size(0), dim=0)
        # p2_s = p2.split(k1.size(0), dim=0)
        #
        # k1 = k1 / k1.norm(dim=-1, keepdim=True)
        # k2 = k2 / k2.norm(dim=-1, keepdim=True)
        #
        # k1_gather = concat_all_gather(k1.detach())
        # k2_gather = concat_all_gather(k2.detach())
        #
        # loss = 0
        # for p in p1_s:
        #     loss = loss + self.loss1(p, k2_gather)
        # for p in p2_s:
        #     loss = loss + self.loss1(p, k1_gather)
        #
        # loss1 = loss / (len(p1_s) + len(p2_s))

        # loss2 = self.loss2(j3, p1) + self.loss2(j1, p3)
        loss2 = 0.5 * self.loss2(j3, p1) + 0.5 * self.loss2(j1, p3)

        # loss_total = b * loss1 + (1-b) * loss2
        # loss_total = b * loss1 + (1 - b) * a * loss2
        loss_total = loss1 + a * loss2

        return loss_total


@torch.no_grad()
def concat_all_gather(tensor):
    """gather the given tensor"""
    tensors_gather = [torch.ones_like(tensor) for _ in range(link.get_world_size())]
    dist.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output
