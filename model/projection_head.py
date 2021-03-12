import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import config

class Projection(nn.Module):

    def __init__(self, hidden_dim=config.simclr.projection_head_dim, num_heads=config.simclr.num_projection_heads):
        super().__init__()
        self.hidden_dim = hidden_dim

        heads = []
        for _ in range(num_heads - 1):
            heads.extend([nn.Linear(self.hidden_dim, self.hidden_dim, bias=False), nn.ReLU()])

        heads += [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)]
        self.model = nn.Sequential(*heads)

    def forward(self, x):
        x = torch.mean(x, dim=2)
        x = self.model(x)
        return F.normalize(x, dim=1)