import torch
import torch.nn as nn


class ConvBlock(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        pad = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


    def forward(self, x):
        return self.block(x)


class ContractionBlock(nn.Module):


    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.path1 = nn.Sequential(
            ConvBlock(in_channels, out_channels, 3),
            ConvBlock(out_channels, out_channels, 1),
            ConvBlock(out_channels, out_channels, 3)
        )

        self.path2 = ConvBlock(in_channels, in_channels, 1)

        self.max_pool = nn.MaxPool2d(2)


    def forward(self, x):
        path1_out = self.path1(x)
        path2_out = self.path2(x)

        path2_out_c = torch.cat([x, path2_out], dim=1)
        
        paths_sum = path1_out + path2_out_c
        out = self.max_pool(paths_sum)

        return out


class FeedForwardBlock(nn.Module):


    def __init__(self, in_channels):
        super().__init__()


    def forward(self, x):
        pass


class UpsamplingBlock(nn.Module):


    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = ConvBlock(in_channels, out_channels, 1)


    def forward(self, x):
        out = self.upsampler(x)
        out = self.conv(out)

        return out


class ExpansionBlock(nn.Module):


    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

        self.convs = nn.Sequential(
            ConvBlock(out_channels, out_channels, 3),
            ConvBlock(out_channels, out_channels, 3)
        )


    def forward(self, x, ff_input, contr_input):
        upscaled = self.conv_t(x)

        convs_in = torch.cat([ff_input, upscaled], dim=1)
        convs_out = self.convs(convs_in)

        out = contr_input + convs_out + upscaled
        return out


if __name__ == '__main__':
    block = ContractionBlock(3, 6)
    print(block)

    num_params = sum(p.numel() for p in block.parameters())
    print(f'# of parameters: {num_params}')

    x = torch.rand(1, 3, 64, 64)
    y = block(x)
    print(f'{x.shape} -> {y.shape}')