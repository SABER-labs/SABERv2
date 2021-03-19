import torch
import torch.nn as nn
import torch.nn.functional as F
class TDSBlock(nn.Module):

    def __init__(self, channels, kernel_size, width, dropout , inner_linearDim, right_padding, time_dim=0):
        super().__init__()
        self.channels = channels
        self.width = width
        self.time_dim = time_dim
        """ Input Dimensions: Batch * Channel * Freuency * Time """
        conv_padding = int ((kernel_size - 1)/2 + 0.5)
        self.total_padding = 0
        if right_padding != -1:
            self.total_padding = kernel_size - 1

            assert self.total_padding < right_padding, "right padding exceeds the 'SAME' padding required for TDSBlock"
            conv_padding = 0
        
        self.conv_padding = torch.nn.ConstantPad2d((self.total_padding-right_padding, right_padding, 0, 0), 0)
        self.conv_layer = torch.nn.Conv2d(channels, channels, (kernel_size,1), 1, (0, conv_padding))   

        assert dropout >= 0, "dropout cannot be less than 0"

        self.dropout_conv = torch.nn.Dropout(dropout)

        self.linear_dim = channels * width
        
        if inner_linearDim == 0:
            inner_linearDim = self.linear_dim

        self.linear1 = torch.nn.Linear(self.linear_dim, inner_linearDim)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(inner_linearDim, self.linear_dim)
        self.dropout2 = torch.nn.Dropout(dropout)

        if (time_dim):
            self.conv_layerN = torch.nn.LayerNorm([channels, width, time_dim])
            self.linear_layerN = torch.nn.LayerNorm([channels, width, time_dim])
        else: 
            self.conv_layerN = torch.nn.LayerNorm([channels, width])
            self.linear_layerN = torch.nn.LayerNorm([channels, width])

    def forward(self, x):

        out = self.conv_padding(x)
        out = self.conv_layer(out)
        out = torch.relu(out)
        out = self.dropout_conv(out)

        out = out + x
        if self.time_dim == 0:
            out = out.permute(0, 3, 2, 1)
            out = self.conv_layerN(x)
            x = out.permute(0, 3, 2, 1)
        else:
            x = self.conv_layerN(x)

        out = x.view((-1, self.time_dim, 1, self.linear_dim))
        out = self.linear1(out)
        out = torch.relu(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = torch.relu(out)
        out = self.dropout2(out)
        out = out.permute((0, 3, 2, 1))
        out = out.view((-1, self.channels, self.width, self.time_dim))

        out = out + x
        if self.time_dim == 0:
            out = out.permute(0, 3, 2, 1)
            out = self.linear_layerN(x)
            out = out.permute(0, 3, 2, 1)
        else:
            out = self.linear_layerN(x)

        return out








        





