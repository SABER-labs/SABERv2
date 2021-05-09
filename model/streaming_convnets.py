import torch
import torch.nn.functional as F
import torch.nn as nn
from model.tds_block import TDSBlock


class FBLayerNorm(nn.Module):

    def __init__(self, channel, width):
        super().__init__()
        self.norm = nn.LayerNorm([channel, width])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Streaming_convnets(nn.Module):

    def __init__(self, dropout, n_mels, input_channels):

        super().__init__()

        conv_block1 = nn.Sequential(
            nn.ConstantPad2d((5, 3, 0, 0), 0),
            nn.Conv2d(input_channels, 15, (1, 10), (1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            FBLayerNorm(15, n_mels),
            TDSBlock(15, 9, n_mels, 0.1, 1),
            TDSBlock(15, 9, n_mels, 0.1, 1)
        )

        conv_block2 = nn.Sequential(
            nn.ConstantPad2d((7, 1, 0, 0), 0),
            nn.Conv2d(15, 19, (1, 10), (1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            FBLayerNorm(19, n_mels),
            TDSBlock(19, 9, n_mels, 0.1, 1),
            TDSBlock(19, 9, n_mels, 0.1, 1),
            TDSBlock(19, 9, n_mels, 0.1, 1)
        )

        conv_block3 = nn.Sequential(
            nn.ConstantPad2d((9, 1, 0, 0), 0),
            nn.Conv2d(19, 23, (1, 12), (1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            FBLayerNorm(23, n_mels),
            TDSBlock(23, 11, n_mels, 0.1, 1),
            TDSBlock(23, 11, n_mels, 0.1, 1),
            TDSBlock(23, 11, n_mels, 0.1, 1),
            # Note,
            # https://github.com/flashlight/wav2letter/blob/master/recipes/streaming_convnets/librispeech/am_500ms_future_context.arch#L26
            TDSBlock(23, 11, n_mels, 0.1, 0)
        )

        conv_block4 = nn.Sequential(
            nn.ConstantPad2d((10, 0, 0, 0), 0),
            nn.Conv2d(23, 27, (1, 11), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            FBLayerNorm(27, n_mels),
            TDSBlock(27, 11, n_mels, 0.1, 0),
            TDSBlock(27, 11, n_mels, 0.1, 0),
            TDSBlock(27, 11, n_mels, 0.1, 0),
            TDSBlock(27, 11, n_mels, 0.1, 0),
            TDSBlock(27, 11, n_mels, 0.1, 0)
        )

        self.layers = nn.Sequential(
            conv_block1,
            conv_block2,
            conv_block3,
            conv_block4
        )

        self.final_feature_count = 27 * n_mels

    def forward(self, x):
        x = self.layers(x)
        return x.view(x.shape[0], self.final_feature_count, -1)


if __name__ == "__main__":
    model = Streaming_convnets(0.1, 80, 1)
    from torchscan import summary
    summary(model, (1, 80, 400))
