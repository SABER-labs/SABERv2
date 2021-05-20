import torch
import torch.nn.functional as F
import torch.nn as nn

# blocks


def conv_bn_act(in_size, out_size, kernel_size, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv1d(in_size, out_size, kernel_size, stride=stride, dilation=dilation),
        nn.GroupNorm(1, out_size),
        nn.Hardswish(inplace=True)
    )


def sepconv_bn(
        in_size,
        out_size,
        kernel_size,
        stride=1,
        dilation=1,
        padding=None):
    if padding is None:
        padding = (kernel_size) // 2
    # print(f"Conv1d set in in_ch={in_size}, out_ch={in_size}, kw={kernel_size}, s={stride}, d={dilation}, g={in_size}, p={padding}")
    return nn.Sequential(
        torch.nn.Conv1d(in_size, in_size, kernel_size=kernel_size,
                        stride=stride, dilation=dilation, groups=in_size,
                        padding=padding),
        torch.nn.Conv1d(in_size, out_size, kernel_size=1),
        nn.GroupNorm(1, out_size)
    )

class QnetBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride=1,
                 R=5):
        super().__init__()

        self.layers = nn.ModuleList(sepconv_bn(
            in_size, out_size, kernel_size, stride))
        for _ in range(R - 1):
            self.layers.append(nn.Hardswish(inplace=True))
            self.layers.append(sepconv_bn(
                out_size, out_size, kernel_size, 1))
        self.layers = nn.Sequential(*self.layers)

        self.residual = nn.ModuleList()
        self.residual.append(torch.nn.Conv1d(in_size, out_size, kernel_size=1, stride=stride))
        self.residual.append(torch.nn.GroupNorm(1, out_size))
        self.residual = nn.Sequential(*self.residual)

    def forward(self, x):
        return F.hardswish(self.residual(x) + self.layers(x))


class QuartzNet(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.c1 = sepconv_bn(n_mels, 256, kernel_size=33, stride=2)
        self.blocks = nn.Sequential(
            #         in   out  k   s  R
            QnetBlock(256, 256, 33, 2, R=5),
            QnetBlock(256, 256, 33, 1, R=5),
            QnetBlock(256, 256, 33, 1, R=5),
            QnetBlock(256, 256, 39, 1, R=5),
            QnetBlock(256, 256, 39, 1, R=5),
            QnetBlock(256, 256, 39, 1, R=5),
            QnetBlock(256, 512, 51, 1, R=5),
            QnetBlock(512, 512, 51, 1, R=5),
            QnetBlock(512, 512, 51, 1, R=5),
            QnetBlock(512, 512, 63, 1, R=5),
            QnetBlock(512, 512, 63, 1, R=5),
            QnetBlock(512, 512, 63, 1, R=5),
            QnetBlock(512, 512, 75, 1, R=5),
            QnetBlock(512, 512, 75, 1, R=5),
            QnetBlock(512, 512, 75, 1, R=5)
        )
        self.c2 = sepconv_bn(512, 512, kernel_size=87, dilation=2, padding=86)
        self.c3 = conv_bn_act(512, 1024, kernel_size=1)

    def model_stride(self):
        return 4

    def forward(self, x):
        c1 = F.hardswish(self.c1(x))
        blocks = self.blocks(c1)
        c2 = F.hardswish(self.c2(blocks))
        c3 = self.c3(c2)
        return c3


if __name__ == "__main__":
    model = QuartzNet(80)
    from torchscan import summary
    summary(model, (80, 400), receptive_field=True)
    print(f"Output shape: {model(torch.rand(1, 80, 400)).shape}")
