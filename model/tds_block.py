import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import config


class TDSBlock(nn.Module):

    def __init__(self, channels, kernel_size, width, dropout, right_padding):
        super().__init__()

        self.channels = channels
        self.width = width

        assert(right_padding >= 0)

        self.conv_block = nn.Sequential(
            torch.nn.ConstantPad2d(
                        (kernel_size - 1 - right_padding, right_padding, 0, 0), 0),
            torch.nn.Conv2d(
                        channels, channels, (1, kernel_size), 1, (0, 0)),
            nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout)
        )

        linear_dim = channels * width

        self.linear_block = nn.Sequential(
            torch.nn.Linear(linear_dim, linear_dim),
            nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(linear_dim, linear_dim),
            torch.nn.Dropout(dropout)
        )

        self.conv_layerN = torch.nn.LayerNorm([channels, width])
        self.linear_layerN = torch.nn.LayerNorm([channels, width])

    def forward(self, x):
        # X is B, C, W, T
        out = self.conv_block(x) + x
        out = out.permute(0, 3, 1, 2) # B, T, C, W
        out = self.conv_layerN(out)
        B, T, C, W = out.shape
        out = out.view((B, T, 1, C*W))
        out = self.linear_block(out) + out
        out = out.view(B, T, C, W)
        out = self.linear_layerN(out)
        out = out.permute(0, 2, 3, 1) # B, C, W, T
        return out

if __name__ == "__main__":
    model = TDSBlock(15, 10, 80, 0.1, 1)
    x = torch.rand(8, 15, 80, 400)
    import time
    start = time.perf_counter()
    model(x)
    end = time.perf_counter()
    print(f"Time taken: {(end-start)*1000:.3f}ms")