from utils.config import config
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.training_utils import GatherLayer

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size, margin):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        positive_mask = self.mask_correlated_samples(batch_size, world_size)
        self.register_buffer("diagonal_mask", torch.zeros_like(positive_mask).fill_diagonal_(1).detach())
        self.register_buffer("masked_margin", positive_mask * margin)
        # get the positive label position from the mask.
        N = 2 * self.batch_size * self.world_size
        self.register_buffer("labels", torch.masked_select(torch.arange(N).repeat(N, 1), positive_mask).long().detach())

        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2, eps=1e-6)

    def mask_correlated_samples(self, batch_size, world_size):
        N = batch_size * world_size
        ones_N = torch.ones(N)
        mask = torch.diag(ones_N, N) + torch.diag(ones_N, -N)
        return mask.type(torch.bool)

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        if self.world_size > 1:
            z_i = GatherLayer.apply(z_i)
            z_j = GatherLayer.apply(z_j)

        z = torch.cat((z_i, z_j), dim=0)

        sim = (self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) - self.masked_margin) / self.temperature

        # zi-zi gets reduced by a large number to make it exponent to 0.
        sim = sim - (self.diagonal_mask * 1e9)

        loss = self.criterion(sim, self.labels)
        return loss

class SimSiamLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()