import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os


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


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        pad = kernel_size // 2

        self.main_path, self.shortcut_path = None, None

        main_path_layers = []

        main_path_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias))
        main_path_layers.append(nn.BatchNorm2d(out_channels))
        main_path_layers.append(nn.ReLU())
        self.main_path = nn.Sequential(*main_path_layers)

        skip_layers = []
        if in_channels != out_channels:
            skip_layers.append(nn.Conv2d(in_channels, out_channels, 1, bias=False))
        self.shortcut_path = nn.Sequential(*skip_layers)

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ContractionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        if residual:
            self.path1 = nn.Sequential(
                ResidualBlock(in_channels, out_channels, 3),
                ConvBlock(out_channels, out_channels, 1),
                ResidualBlock(out_channels, out_channels, 3)
            )

            self.path2 = ResidualBlock(in_channels, in_channels, 1)
        else:
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

    def __init__(self, in_channels, n):
        super().__init__()
        self.pooling_layers = []
        self.n = n
        for i in range(n):
            stride = 2 ** (n-i)
            pooling_block = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
            self.pooling_layers.append(pooling_block)

        self.pooling_layers = nn.ModuleList(self.pooling_layers)

    def forward(self, inp_list):
        n = self.n
        pooled_inp_list = [inp_list[-1]]
        for i, inp in enumerate(inp_list[:-1]):
            factor = 2 ** (n-i)
            cat_list = []
            for j in range(factor):
                cat_list.append(inp)
            cat_inp = torch.cat(cat_list, dim=1)
            pooled_inp = self.pooling_layers[i].forward(cat_inp)
            pooled_inp_list.append(pooled_inp)
        output = sum(pooled_inp_list)
        return output


class ExpansionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()

        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        if residual:
            self.convs = nn.Sequential(
                ConvBlock(in_channels, out_channels, 3),
                ResidualBlock(out_channels, out_channels, 3),
                ResidualBlock(out_channels, out_channels, 3)
            )
        else:
            self.convs = nn.Sequential(
                ConvBlock(in_channels, out_channels, 3),
                ConvBlock(out_channels, out_channels, 3),
                ConvBlock(out_channels, out_channels, 3)
            )

    def forward(self, x, ff_input, contr_input):
        upscaled = ff_input
        if x is not None:
            upscaled = self.conv_t(x)

        convs_in = torch.cat([ff_input, upscaled], dim=1)

        convs_out = self.convs(convs_in)

        out = contr_input + convs_out + upscaled
        return out


class UpsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()

        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.upsampler(x)
        out = self.conv(out)

        return out


class CloudNetPlus(nn.Module):

    def __init__(self, input_channels=4, inception_depth=6, residual=False, softmax=False, sigmoid=True):
        super().__init__()
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.depth = inception_depth
        self.contr_layers = []
        self.ff_layers = []
        self.exp_layers = []
        self.upsample_layers = []

        # Contraction layers
        channels = input_channels
        for i in range(inception_depth):
            contr_block = ContractionBlock(channels, 2 * channels, residual=residual)
            channels *= 2
            self.contr_layers.append(contr_block)
        self.contr_layers = nn.ModuleList(self.contr_layers)

        # FeedForward layers
        channels = input_channels * 2
        for i in range(1, inception_depth):
            channels *= 2
            ff_block = FeedForwardBlock(channels, i)
            self.ff_layers.append(ff_block)
        self.ff_layers = nn.ModuleList(self.ff_layers)

        # Expansion layers
        for i in reversed(range(1, inception_depth)):
            factor = 2 ** (i + 1)
            exp_block = ExpansionBlock(input_channels * factor * 2, input_channels * factor, residual=residual)
            self.exp_layers.append(exp_block)
        self.exp_layers = nn.ModuleList(self.exp_layers)

        # Upsampling layers
        out_channels = 2 if softmax or not sigmoid else 1
        factor = 0
        for i in range(0, inception_depth - 1):
            factor = 2 ** (i + 2)
            upsampling_block = UpsamplingBlock(input_channels * factor, out_channels, factor)
            self.upsample_layers.append(upsampling_block)
        self.upsample_layers.append(UpsamplingBlock(input_channels * factor, out_channels, factor))
        self.upsample_layers = nn.ModuleList(self.upsample_layers)

        if sigmoid or softmax:
            self.final_layer = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                             nn.Softmax2d() if softmax else nn.Sigmoid())
        else:
            self.final_layer = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        contr_output = []
        ff_output = []
        exp_output = []
        upsample_output = []

        # Contraction forward pass
        output = x
        for contr_block in self.contr_layers:
            output = contr_block(output)
            contr_output.append(output)

        # FeedForward forward pass
        for i, ff_block in enumerate(self.ff_layers):
            output = ff_block(contr_output[:i+2])
            ff_output.append(output)

        # Expansion forward pass
        output = None
        for i, exp_block in enumerate(self.exp_layers):
            n = self.depth - i - 1
            output = exp_block.forward(output, ff_output[n - 1], contr_output[n])
            exp_output.append(output)
        exp_output = list(reversed(exp_output))

        # Upsampling forward pass
        for i, upsampling_block in enumerate(self.upsample_layers):
            if i == self.depth - 1:
                input = contr_output[i]
            else:
                input = exp_output[i]
            output = upsampling_block.forward(input)
            upsample_output.append(output)

        # Sum all outputs
        output = sum(upsample_output)

        # Final layer
        output = self.final_layer(output)
        if self.sigmoid:
            output = output.squeeze(1)
            
        return output


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)

    print(dev)

    gt = torch.Tensor(io.imread(r'..\data\gt_1.TIF') / 255).to(device)
    red = torch.Tensor(io.imread(r'..\data\red_1.TIF') / 65535).unsqueeze(0)
    green = torch.Tensor(io.imread(r'..\data\green_1.TIF') / 65535).unsqueeze(0)
    blue = torch.Tensor(io.imread(r'..\data\blue_1.TIF')/ 65535).unsqueeze(0)
    nir = torch.Tensor(io.imread(r'..\data\nir_1.TIF')/ 65535).unsqueeze(0)

    rgb = torch.cat([red, green, blue, nir], dim=0).unsqueeze(0).to(device)
    print(rgb.shape)

    block = CloudNetPlus(4, 6).to(device)
    print(block)
    num_params = sum(p.numel() for p in block.parameters())
    print(f'# of parameters: {num_params}')

    x = torch.rand(1, 4, 192,192)
    y = block.forward(rgb)
    print(f'{rgb.shape} -> {y.shape}')

    parameters = [x for x in block.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=100e-4)

    from losses import FilteredJaccardLoss
    loss_func = FilteredJaccardLoss()

    for i in range(1000):
        optimizer.zero_grad()

        out = block(rgb)
        if i % 100 == 0:
            gray = out.detach().clone()
            gray[out>=0.5] = 1
            gray[out < 0.5] = 0
            plt.imshow(gray.squeeze().cpu().detach().numpy(), cmap='gray')
            plt.show()

        loss = loss_func(out, gt)
        print(f'iteration {i} loss : {loss.item()}')
        loss.backward()
        optimizer.step()
        
    out = block(rgb)
    plt.imshow(out)
    plt.show()


