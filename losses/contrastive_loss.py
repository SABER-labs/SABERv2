from utils.config import config
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from utils.training_utils import GatherLayer

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.negative_mask = self.mask_correlated_samples(batch_size, world_size).cuda()
        self.positive_mask = torch.bitwise_not(self.negative_mask).fill_diagonal_(0)
        self.diagonal_mask = torch.zeros_like(self.negative_mask).fill_diagonal_(1)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        M = batch_size
        main = torch.ones((N,N))
        ones_M = torch.ones(M)
        mask = 1 - (torch.diag(torch.ones(2 * M)) + torch.diag(ones_M, batch_size) + torch.diag(ones_M, -batch_size))
        for i in range(world_size):
            main[i*2*batch_size:(i+1)*2*batch_size, i*2*batch_size:(i+1)*2*batch_size] *= mask
        return main.type(torch.bool)

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # zi-zi gets reduced by a large number to make it exponent to 0.
        sim = sim - (self.diagonal_mask * 1e9)

        # get the positive label position from the mask.
        labels = torch.masked_select(torch.arange(N).repeat(N, 1), self.positive_mask).to(device=z.device).long()

        loss = self.criterion(sim, labels)
        loss /= N
        return loss