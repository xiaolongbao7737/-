# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    train_resnet.py
# Version:      ver3_0
# Created:      2026/06/29
# Description:  使用ResNet18迁移学习训练手势识别模型
#               用ImageNet预训练权重初始化，只微调全连接层+部分卷积层
#               更适合实拍照片的泛化识别
# -----------------------------------------------------------------------

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import (
    Compose, ToTensor, Normalize, Resize,
    RandomRotation, RandomAffine, ColorJitter,
    RandomPerspective, GaussianBlur, RandomGrayscale,
    RandomErasing, RandomHorizontalFlip,
)
from dataset import CustomDataset
from model import ResNetGesture


def train_resnet(epoch, train_loader, val_loader, model, loss_fn,
                 optimizer, scheduler, device, save_path='./models/model.pkl'):
    """训练ResNet迁移学习模型。"""
    best_val_loss = float('inf')

    for epoch_idx in range(epoch):
        # ====== 训练 ======
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, data in enumerate(train_loader):
            images = data['image'].to(device)
            labels = data['label'].to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch_idx+1}/{epoch}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        train_acc = 100.0 * correct / total
        avg_train_loss = train_loss / len(train_loader)

        # ====== 验证 ======
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data in val_loader:
                images = data['image'].to(device)
                labels = data['label'].to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch_idx+1}/{epoch}] | '
              f'Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}% | '
              f'LR: {current_lr:.2e}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model, save_path)
            print(f'  [OK] 验证损失下降，模型已保存')

    # 最终保存
    torch.save(model, save_path)
    print(f'\n训练完成！最佳验证损失: {best_val_loss:.4f}')


if __name__ == "__main__":
    # ====== 超参数 ======
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4          # 迁移学习用较小学习率
    WEIGHT_DECAY = 5e-5           # 更轻的正则化
    EPOCH = 100                    # ResNet收敛更快
    VAL_SPLIT = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[设备] 使用: {device}')

    # ====== 数据增强（配合ResNet使用ImageNet标准化）======
    # ImageNet标准化参数
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = Compose([
        # 几何变换
        RandomRotation(degrees=30),
        RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.75, 1.25),
            shear=15,
        ),
        RandomPerspective(distortion_scale=0.2, p=0.5),
        RandomHorizontalFlip(p=0.3),  # 水平翻转（左右手互换）

        # 颜色/模糊变换
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        RandomGrayscale(p=0.15),

        # 最终处理
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        RandomErasing(p=0.3, scale=(0.02, 0.15)),  # 随机遮挡（防过拟合）
    ])

    val_transform = Compose([
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ====== 数据集 ======
    print('[数据] 加载训练数据...')
    train_dataset_aug = CustomDataset('./images/train.txt', './images/train',
                                      lambda: train_transform)
    val_dataset_raw = CustomDataset('./images/train.txt', './images/train',
                                    lambda: val_transform)

    n = len(train_dataset_aug)
    indices = torch.randperm(n)
    split = int(n * (1 - VAL_SPLIT))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = Subset(train_dataset_aug, train_indices.tolist())
    val_dataset = Subset(val_dataset_raw, val_indices.tolist())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f'  训练集: {len(train_dataset)} 张')
    print(f'  验证集: {len(val_dataset)} 张')

    # ====== 模型 ======
    print('[模型] 初始化 ResNetGesture（ResNet18迁移学习）...')
    model = ResNetGesture(num_classes=10)
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  总参数量: {total:,}  可训练: {trainable:,}')

    # ====== 损失 & 优化器 ======
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8,
        min_lr=1e-6,
    )

    # ====== 训练 ======
    print(f'[训练] 开始训练，共 {EPOCH} 轮...')
    print('=' * 70)
    train_resnet(EPOCH, train_loader, val_loader, model, loss_fn,
                 optimizer, scheduler, device)

    # ====== 测试 ======
    print('\n[测试] 在测试集上评估...')
    from test import test

    test_transform = Compose([
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_loader = DataLoader(
        CustomDataset('./images/test.txt', './images/test', lambda: test_transform),
        batch_size=32,
    )
    test(test_loader, model, device)
