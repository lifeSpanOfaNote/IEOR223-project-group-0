import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, ngpu,  edge_channels=1, ngf = 64, nz = 100, nc = 3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.nz = nz
        self.nc = nc
        # 噪声向量的处理层
        self.noise_process = nn.Sequential(
            nn.Linear(self.nz, self.ngf * 4 * 4),
            nn.BatchNorm1d(self.ngf * 4 * 4),
            nn.ReLU(True)
        )

        # 边缘图像的初始处理层
        self.edge_process = nn.Sequential(
            nn.Conv2d(edge_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, self.ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )

        # 主生成器模块，添加了ResNet块
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 2, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            ResNetBlock(self.ngf * 8),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            ResNetBlock(self.ngf * 4),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            ResNetBlock(self.ngf * 2),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            ResNetBlock(self.ngf),
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, edge_image):
        noise = noise.view(-1, self.nz)
        noise_feature = self.noise_process(noise)
        noise_feature = noise_feature.view(-1, self.ngf, 4, 4)
        edge_feature = self.edge_process(edge_image)
        combined_feature = torch.cat((noise_feature, edge_feature), 1)
        output = self.main(combined_feature)
        return output