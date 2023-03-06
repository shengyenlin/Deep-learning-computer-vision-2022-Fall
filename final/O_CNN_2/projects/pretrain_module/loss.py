import torch
from torch import nn
import torch.nn.functional as F


def simsiam_loss(p1, p2, z1, z2):
    return -.5 * (F.cosine_similarity(p1, z2.detach(), dim=-1).mean() +
                  F.cosine_similarity(p2, z1.detach(), dim=-1).mean())
