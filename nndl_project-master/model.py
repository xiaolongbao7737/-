# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    model.py
# Version:      ver2_0
# Created:      2024/06/17
# Description:  本文件定义了CustomNet类，用于定义神经网络模型
#               改进版：双卷积 + BatchNorm + AdaptiveAvgPool
# -----------------------------------------------------------------------

import torch
from torch import nn


class CustomNet(nn.Module):
    """改进版CNN模型（v2.0）。

    相比原版改进：
    - 每个stage使用2层卷积（提升表示能力）
    - 添加BatchNormalization（加速收敛、稳定训练）
    - 通道数从 3→16→32→64→128 提升至 3→64→128→256→512
    - 全局平均池化替代Flatten（适配任意输入尺寸）
    - 双层Dropout（0.5+0.3）防止过拟合

    模型结构：
      Block1: Conv3-64×2 → BN → ReLU → MaxPool (64×32×32)
      Block2: Conv64-128×2 → BN → ReLU → MaxPool (128×16×16)
      Block3: Conv128-256×2 → BN → ReLU → MaxPool (256×8×8)
      Block4: Conv256-512×2 → BN → ReLU → MaxPool (512×4×4)
      Classifier: GlobalAvgPool → Dropout → FC(512→256) → Dropout → FC(256→10)

    输入：3×64×64 (RGB图片)
    输出：10 (数字0~9分类)
    参数量：~3.2M
    """

    def __init__(self, num_classes=10):
        """初始化方法。"""
        super(CustomNet, self).__init__()

        # START----------------------------------------------------------
        # 第1阶段: 3 → 64, 64×32×32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 第2阶段: 64 → 128, 128×16×16
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 第3阶段: 128 → 256, 256×8×8
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 第4阶段: 256 → 512, 512×4×4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 分类器: 全局平均池化 + 双Dropout
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → 512
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        # END------------------------------------------------------------

    def forward(self, x):
        """前向传播过程。"""
        # START----------------------------------------------------------
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x
        # END------------------------------------------------------------


class ResNetGesture(nn.Module):
    """ResNet18迁移学习模型（v3.0）。

    使用在ImageNet上预训练的ResNet18，在大规模通用视觉特征基础上微调。
    相比从零训练的CNN，预训练模型能大幅提升实拍照片的泛化能力。

    修改：
    - 将首层7×7卷积改为3×3（适配64×64小图）
    - 去掉初始MaxPool（保留更多空间信息）
    - 替换全连接层为双Dropout分类头

    输入：3×64×64 (RGB图片)
    输出：10 (数字0~9分类)
    参数量：~11.7M（其中99%来自预训练权重）
    """

    def __init__(self, num_classes=10):
        super(ResNetGesture, self).__init__()
        import torchvision.models as models

        # 加载预训练ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 修改首层：原7×7 stride=2 → 3×3 stride=1（适配64×64输入）
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 去掉初始池化（64×64太小，再池化只剩32×32，损失太多信息）
        self.backbone.maxpool = nn.Identity()

        # 替换全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    # 测试
    from dataset import CustomDataset
    from torchvision.transforms import ToTensor

    c = CustomDataset('./images/train.txt', './images/train', ToTensor)
    for name, NetClass in [('CustomNet', CustomNet), ('ResNetGesture', ResNetGesture)]:
        net = NetClass()
        x = torch.unsqueeze(c[10]['image'], 0)
        y = net(x)
        print(f"{name}: 输入{x.shape} → 输出{y.shape}  |  参数量: {sum(p.numel() for p in net.parameters()):,}")
