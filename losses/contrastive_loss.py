from utils.config import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.training_utils import GatherLayer
from utils.training_utils import off_diagonal


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size, margin):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        positive_mask = self.mask_correlated_samples(batch_size, world_size)
        self.register_buffer("diagonal_mask", torch.zeros_like(
            positive_mask).fill_diagonal_(1).detach())
        self.register_buffer("masked_margin", positive_mask * margin)
        # get the positive label position from the mask.
        N = 2 * self.batch_size * self.world_size
        self.register_buffer("labels", torch.masked_select(
            torch.arange(N).repeat(N, 1), positive_mask).long().detach())

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

        sim = (self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) -
               self.masked_margin) / self.temperature

        # zi-zi gets reduced by a large number to make it exponent to 0.
        sim = sim - (self.diagonal_mask * 1e9)

        loss = self.criterion(sim, self.labels)
        return loss


class SimSiamLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()


class BarlowTwinsLoss(torch.nn.Module):

    def __init__(
            self,
            lambda_param=config.barlow_twins.lambda_param,
            scale_loss=config.barlow_twins.scale_loss):
        super().__init__()
        self.lambda_param = lambda_param
        self.scale_loss = scale_loss
        self.bn = nn.BatchNorm1d(
            config.barlow_twins.projection_out_dim, affine=False)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        N, D = z_a.size()
        c = self.bn(z_a).T @ self.bn(z_b)  # DxD
        c.div_(N)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
        loss = on_diag + self.lambda_param * off_diag
        return loss


class AggregatedCrossEntropyLoss(torch.nn.Module):

    def __init__(
            self,
            stride=1,
            gumbel_temperature=config.aggregated_ce.gumbel_temperature):
        super().__init__()
        self.gumbel_temperature = gumbel_temperature
        self.stride = stride

    def _get_lengths(self, lengths):
        return lengths.div(self.stride).ceil_()

    def forward(
            self,
            g_a: torch.Tensor,
            g_b: torch.Tensor,
            img1_len: torch.Tensor,
            img2_len: torch.Tensor):
        quantized_g_a = F.gumbel_softmax(
            g_a, tau=self.gumbel_temperature, hard=True, dim=1)
        quantized_g_b = F.gumbel_softmax(
            g_b, tau=self.gumbel_temperature, hard=True, dim=1)

        C = g_a.size(1)

        char_counts_ga = quantized_g_a.sum(dim=2)
        char_counts_gb = quantized_g_b.sum(dim=2)

        entropy_ga = (
            torch.softmax(
                g_a,
                dim=1) *
            torch.log_softmax(
                g_a,
                dim=1)).sum(
            dim=(
                1,
                2)).div_(
            C *
            self._get_lengths(img1_len)).mean()
        entropy_gb = (
            torch.softmax(
                g_b,
                dim=1) *
            torch.log_softmax(
                g_b,
                dim=1)).sum(
            dim=(
                1,
                2)).div_(
            C *
            self._get_lengths(img2_len)).mean()

        return (
            F.l1_loss(
                char_counts_ga,
                char_counts_gb),
            entropy_ga +
            entropy_gb)
