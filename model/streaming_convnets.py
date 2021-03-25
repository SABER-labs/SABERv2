import torch
import torch.nn.functional as F
import torch.nn as nn
from model.tds_block import TDSBlock
class Streaming_convnets(nn.Module):

    def __init__(self, dropout, n_mels, input_channels):

        super().__init__()

        self.width = n_mels

        self.pd1 = nn.ConstantPad2d((5,3, 0, 0),0)
        self.c1 = nn.Conv2d(input_channels, 15, (1, 10), (1, 2))
        self.dropout1 = nn.Dropout(dropout)
        
        self.tds1 = TDSBlock(15, 9, n_mels, 0.1, 0, 1, 0)
        self.tds2 = TDSBlock(15, 9, n_mels, 0.1, 0, 1, 0)

        self.pd2 = nn.ConstantPad2d((7,1, 0, 0),0)
        self.c2 = nn.Conv2d(15, 19, (1, 10), (1, 2))
        self.dropout2 = nn.Dropout(dropout)

        self.tds3 = TDSBlock(19, 9, n_mels, 0.1, 0, 1, 0)
        self.tds4 = TDSBlock(19, 9, n_mels, 0.1, 0, 1, 0)
        self.tds5 = TDSBlock(19, 9, n_mels, 0.1, 0, 1, 0)

        self.pd3 = nn.ConstantPad2d((9,1, 0, 0),0)
        self.c3 = nn.Conv2d(19, 23, (1, 12), (1, 2))
        self.dropout3 = nn.Dropout(dropout)

        self.tds6 = TDSBlock(23, 11, n_mels, 0.1, 0, 1, 0)
        self.tds7 = TDSBlock(23, 11, n_mels, 0.1, 0, 1, 0)
        self.tds8 = TDSBlock(23, 11, n_mels, 0.1, 0, 1, 0)
        self.tds9 = TDSBlock(23, 11, n_mels, 0.1, 0, 1, 0)

        self.pd4 = nn.ConstantPad2d((10, 0, 0, 0),0)
        self.c4 = nn.Conv2d(23, 27, (1, 11), (1, 1))
        self.dropout4 = nn.Dropout(dropout)

        self.tds10 = TDSBlock(27, 11, n_mels, 0.1, 0, 0, 0)
        self.tds11 = TDSBlock(27, 11, n_mels, 0.1, 0, 0, 0)
        self.tds12 = TDSBlock(27, 11, n_mels, 0.1, 0, 0, 0)
        self.tds13 = TDSBlock(27, 11, n_mels, 0.1, 0, 0, 0)
        self.tds14 = TDSBlock(27, 11, n_mels, 0.1, 0, 0, 0)


    def forward(self, x):

        x = self.pd1(x)
        x = self.c1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = x.permute(0, 3, 1, 2)
        x = F.layer_norm(x, [x.shape[2], x.shape[3]])
        x = x.permute(0, 2, 3, 1)

        x = self.tds1(x)
        x = self.tds2(x)

        x = self.pd2(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = x.permute(0, 3, 1, 2)
        x = F.layer_norm(x, [x.shape[2], x.shape[3]])
        x = x.permute(0, 2, 3, 1)

        x = self.tds3(x)
        x = self.tds4(x)
        x = self.tds5(x)

        x = self.pd3(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = x.permute(0, 3, 1, 2)
        x = F.layer_norm(x, [x.shape[2], x.shape[3]])
        x = x.permute(0, 2, 3, 1)

        x = self.tds6(x)
        x = self.tds7(x)
        x = self.tds8(x)
        x = self.tds9(x)

        x = self.pd4(x)
        x = self.c4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = x.permute(0, 3, 1, 2)
        x = F.layer_norm(x, [x.shape[2], x.shape[3]])
        x = x.permute(0, 2, 3, 1)

        x = self.tds10(x)
        x = self.tds11(x)
        x = self.tds12(x)
        x = self.tds13(x)
        x = self.tds14(x)

        x = x.permute(0, 3, 1, 2)
        x = x.view(-1, x.shape[1], 1, self.width*27)
         
        return x








    