"""Same CIFARResNet architecture as cifar100-cnnbuild_mixup.ipynb."""
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class CIFARResNet(nn.Module):
    def __init__(self, num_classes=100, blocks_per_stage=3, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.stage1 = self.make_stage(32, 32, blocks_per_stage, 1)
        self.stage2 = self.make_stage(32, 64, blocks_per_stage, 2)
        self.stage3 = self.make_stage(64, 128, blocks_per_stage, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)

    @staticmethod
    def make_stage(in_channels, out_channels, count, stride):
        return nn.Sequential(ResidualBlock(in_channels, out_channels, stride),
                             *[ResidualBlock(out_channels, out_channels) for _ in range(count - 1)])

    def forward(self, x):
        return self.classifier(self.pool(self.stage3(self.stage2(self.stage1(self.stem(x))))))
