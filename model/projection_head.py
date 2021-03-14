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