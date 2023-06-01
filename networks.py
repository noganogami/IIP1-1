import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        return self.relu(out)


class BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BaseBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(in_channels, out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        return self.relu(out)


class InceptionModule(nn.Module):
    def __init__(
        self, in_channels, ch_1x1, ch_3x3red, ch_3x3, ch_5x5red, ch_5x5, ch_pool
    ):
        super(InceptionModule, self).__init__()

        self.branch1x1 = ConvBlock(in_channels, ch_1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            ConvBlock(in_channels, ch_3x3red, kernel_size=1),
            ConvBlock(ch_3x3red, ch_3x3, kernel_size=3, padding=1),
        )
        self.branch5x5 = nn.Sequential(
            ConvBlock(in_channels, ch_5x5red, kernel_size=1),
            ConvBlock(ch_5x5red, ch_5x5, kernel_size=5, padding=2),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, ch_pool, kernel_size=1),
        )

        self.out_channels = ch_1x1 + ch_3x3 + ch_5x5 + ch_pool

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        return torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)


class ResInceptionModule(nn.Module):
    def __init__(
        self, in_channels, ch_1x1, ch_3x3red, ch_3x3, ch_5x5red, ch_5x5, ch_pool
    ):
        super(ResInceptionModule, self).__init__()

        self.in_channels = in_channels

        self.inception = InceptionModule(
            in_channels, ch_1x1, ch_3x3red, ch_3x3, ch_5x5red, ch_5x5, ch_pool
        )

        self.out_channels = self.inception.out_channels
        self.projection = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x if self.in_channels == self.out_channels else self.projection(x)
        out = self.inception(x)
        out += shortcut
        return self.relu(out)


class Baseline(nn.Module):
    def __init__(self, num_classes):
        super(Baseline, self).__init__()
        self.num_classes = num_classes
        self.block1 = ConvBlock(3, 64, kernel_size=7, padding=3, stride=2)
        self.block2 = BaseBlock(64, 64)
        self.block3 = BaseBlock(64, 64)
        self.block4 = BaseBlock(64, 128)
        self.block5 = BaseBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.ave_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out = self.pool(out)

        out = self.block4(out)
        out = self.block5(out)

        out = self.ave_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class CustomInception(nn.Module):
    def __init__(self, num_classes):
        super(CustomInception, self).__init__()
        self.num_classes = num_classes
        self.block1 = ConvBlock(3, 64, kernel_size=7, padding=3, stride=2)
        self.block2 = InceptionModule(
            in_channels=64,
            ch_1x1=16,
            ch_3x3red=24,
            ch_3x3=32,
            ch_5x5red=4,
            ch_5x5=8,
            ch_pool=8,
        )
        self.block3 = InceptionModule(
            in_channels=self.block2.out_channels,
            ch_1x1=8,
            ch_3x3red=24,
            ch_3x3=40,
            ch_5x5red=4,
            ch_5x5=8,
            ch_pool=8,
        )
        self.block4 = InceptionModule(
            in_channels=self.block3.out_channels,
            ch_1x1=32,
            ch_3x3red=48,
            ch_3x3=64,
            ch_5x5red=8,
            ch_5x5=16,
            ch_pool=16,
        )
        self.block5 = InceptionModule(
            in_channels=self.block4.out_channels,
            ch_1x1=64,
            ch_3x3red=96,
            ch_3x3=128,
            ch_5x5red=16,
            ch_5x5=32,
            ch_pool=32,
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.ave_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out = self.pool(out)

        out = self.block4(out)
        out = self.block5(out)

        out = self.ave_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class ResInception(nn.Module):
    def __init__(self, num_classes):
        super(ResInception, self).__init__()
        self.num_classes = num_classes
        self.block1 = ConvBlock(3, 64, kernel_size=7, padding=3, stride=2)
        self.block2 = ResInceptionModule(
            in_channels=64,
            ch_1x1=16,
            ch_3x3red=24,
            ch_3x3=32,
            ch_5x5red=4,
            ch_5x5=8,
            ch_pool=8,
        )
        self.block3 = ResInceptionModule(
            in_channels=self.block2.out_channels,
            ch_1x1=8,
            ch_3x3red=24,
            ch_3x3=40,
            ch_5x5red=4,
            ch_5x5=8,
            ch_pool=8,
        )
        self.block4 = ResInceptionModule(
            in_channels=self.block3.out_channels,
            ch_1x1=32,
            ch_3x3red=48,
            ch_3x3=64,
            ch_5x5red=8,
            ch_5x5=16,
            ch_pool=16,
        )
        self.block5 = ResInceptionModule(
            in_channels=self.block4.out_channels,
            ch_1x1=64,
            ch_3x3red=96,
            ch_3x3=128,
            ch_5x5red=16,
            ch_5x5=32,
            ch_pool=32,
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.ave_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out = self.pool(out)

        out = self.block4(out)
        out = self.block5(out)

        out = self.ave_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
