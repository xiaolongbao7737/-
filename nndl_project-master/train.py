# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    train.py
# Version:      ver2_0
# Created:      2024/06/17
# Description:  本文件定义了模型的训练流程（v2.0 增强版）
#               改进：更强的数据增强 + 验证集 + 余弦退火LR + 权重衰减
# -----------------------------------------------------------------------

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import (
    Compose, ToTensor, Normalize,
    RandomRotation, RandomAffine, ColorJitter,
    RandomPerspective, GaussianBlur, RandomGrayscale,
    Resize,
)
from dataset import CustomDataset
from model import CustomNet


def train_loop(epoch, train_loader, val_loader, model, loss_fn, optimizer,
               scheduler, device, save_path='./models/model.pkl'):
    """定义增强版训练流程。

    包含：
    - 每个epoch在验证集上评估
    - 学习率调度（ReduceLROnPlateau）
    - 自动保存最佳模型（基于验证损失）
    """
    model.train()
    best_val_loss = float('inf')

    for epoch_idx in range(epoch):
        # ====== 训练阶段 ======
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, data in enumerate(train_loader):
            images = data['image'].to(device)
            labels = data['label'].to(device)

            # 前向传播
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch_idx+1}/{epoch}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        train_acc = 100.0 * correct / total
        avg_train_loss = train_loss / len(train_loader)

        # ====== 验证阶段 ======
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

        # ====== 学习率调度 ======
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # 打印
        print(f'Epoch [{epoch_idx+1}/{epoch}] | '
              f'Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}% | '
              f'LR: {current_lr:.2e}')

        # 保存最佳模型（基于验证损失）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model, save_path)
            print(f'  [OK] 验证损失下降，模型已保存至 {save_path}')

    # 训练结束，确保保存最终模型
    torch.save(model, save_path)
    print(f'\n训练完成！最佳验证损失: {best_val_loss:.4f}')
    print(f'最终模型已保存至 {save_path}')


if __name__ == "__main__":
    # ====== 超参数 ======
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4           # L2正则化，防止过拟合
    EPOCH = 200                    # 更多训练轮次
    VAL_SPLIT = 0.1               # 10% 训练数据用于验证

    # ====== 设备 ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    # ====== 数据增强（更强版，模拟真实场景变化）======
    train_transform = Compose([
        # --- 几何变换（模拟不同角度、位置、距离）---
        RandomRotation(degrees=30),
        RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),    # 随机平移 ±15%
            scale=(0.75, 1.25),        # 随机缩放 75%~125%
            shear=15,                   # 随机错切 ±15°
        ),
        RandomPerspective(distortion_scale=0.2, p=0.5),  # 随机透视变换

        # --- 颜色变换（模拟不同光照）---
        ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        ),
        GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),    # 模拟相机对焦模糊
        RandomGrayscale(p=0.1),                            # 偶尔转灰度，降低颜色依赖

        # --- 最终处理 ---
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 验证集仅标准化（不用增强）
    val_transform = Compose([
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # ====== 数据集 ======
    print("[数据] 加载训练数据...")
    train_dataset_aug = CustomDataset('./images/train.txt', './images/train',
                                      lambda: train_transform)
    train_dataset_raw = CustomDataset('./images/train.txt', './images/train',
                                      lambda: val_transform)

    n = len(train_dataset_aug)
    indices = torch.randperm(n)
    split = int(n * (1 - VAL_SPLIT))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = Subset(train_dataset_aug, train_indices.tolist())
    val_dataset = Subset(train_dataset_raw, val_indices.tolist())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  验证集: {len(val_dataset)} 张")

    # ====== 模型 ======
    print("[模型] 初始化 CustomNet v2.0...")
    model = CustomNet(num_classes=10)
    model.to(device)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ====== 损失函数 & 优化器 ======
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ====== 学习率调度 ======
    # 验证损失不下降时，学习率减半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
        min_lr=1e-6, verbose=True,
    )

    # ====== 开始训练 ======
    print(f"[训练] 开始训练，共 {EPOCH} 轮...")
    print("=" * 70)
    train_loop(EPOCH, train_loader, val_loader, model, loss_fn,
               optimizer, scheduler, device)

    # ====== 最终测试 ======
    print("\n[测试] 在测试集上评估最终模型...")
    from test import test
    from torchvision.transforms import ToTensor

    test_transform = Compose([
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_loader = DataLoader(
        CustomDataset('./images/test.txt', './images/test', lambda: test_transform),
        batch_size=32,
    )
    test(test_loader, model, device)
