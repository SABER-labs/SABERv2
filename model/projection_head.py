import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import config

class Projection(nn.Module):

    def __init__(self, hidden_dim=config.simclr.projection_head_dim, final_embedding_dim=config.simclr.final_embedding_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.Hardswish(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.Hardswish(),
            nn.Linear(hidden_dim, final_embedding_dim, bias=False)
        )

    def forward(self, x):
        x = torch.mean(x, dim=2)
        x = self.model(x)
        return x

class BarlowTwinsProjection(nn.Module):

    def __init__(self, in_dim=config.model.output_dim, hid_dim=config.barlow_twins.projection_hid_dim, out_dim=config.barlow_twins.projection_out_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False), nn.BatchNorm1d(hid_dim), nn.Hardswish(inplace=True),
            nn.Linear(hid_dim, hid_dim, bias=False), nn.BatchNorm1d(hid_dim), nn.Hardswish(inplace=True),
            nn.Linear(hid_dim, out_dim, bias=False)
        )

    def forward(self, x):
        x = torch.mean(x, dim=2)
        x = self.model(x)
        return x

class SimSiamProjection(nn.Module):

    def __init__(self, in_dim=config.model.output_dim, hid_dim=config.simsiam.projection_hid_dim, out_dim=config.simsiam.projection_hid_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.BatchNorm1d(hid_dim), nn.Hardswish(inplace=True),
            nn.Linear(hid_dim, hid_dim), nn.BatchNorm1d(hid_dim), nn.Hardswish(inplace=True),
            nn.Linear(hid_dim, out_dim), nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = torch.mean(x, dim=2)
        x = self.model(x)
        return x

class SimSiamPrediction(nn.Module):

    def __init__(self, in_dim=config.simsiam.projection_hid_dim, hid_dim=config.simsiam.prediction_hid_dim, out_dim=config.simsiam.prediction_out_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.BatchNorm1d(hid_dim), nn.Hardswish(inplace=True),
            nn.Linear(hid_dim, out_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class AggregatedProjection(nn.Module):

    def __init__(self, in_dim=config.model.output_dim, hid_dim=config.aggregated_ce.projection_hid_dim, out_dim=config.aggregated_ce.n_characters):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_dim, hid_dim, 1, bias=False), nn.BatchNorm1d(hid_dim), nn.Hardswish(inplace=True),
            nn.Conv1d(hid_dim, out_dim, 1, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return x