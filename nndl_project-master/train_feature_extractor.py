# -*- coding: utf-8 -*- #
"""
ResNet18特征提取器 — 冻结主干，只训练分类头。

核心思路：
  - 使用完全未修改的ResNet18（224x224输入，原始ImageNet权重）
  - 冻结所有卷积层，保留ImageNet通用视觉特征
  - 只训练最后的FC层（2048→10）
  - 这样模型不会"遗忘"通用的形状、纹理识别能力
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import *
from torchvision.models import resnet18, ResNet18_Weights
from dataset import CustomDataset

# ====== 超参数 ======
BATCH_SIZE = 16          # 224x224更大，降低batch size
EPOCH = 30
LR = 1e-3
device = torch.device('cpu')
print(f'设备: {device}')

# ImageNet标准化
M, S = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# 训练：Resize到224x224（ResNet原生尺寸）
train_tfm = Compose([
    RandomRotation(30),
    RandomAffine(0, translate=(0.15,0.15), scale=(0.7,1.3), shear=15),
    RandomPerspective(0.2, p=0.5),
    ColorJitter(0.3, 0.3, 0.3, 0.1),
    GaussianBlur(3, (0.1, 1.5)),
    RandomGrayscale(0.15),
    Resize((224, 224)),
    ToTensor(),
    Normalize(M, S),
    RandomErasing(p=0.25, scale=(0.02, 0.15)),
])
val_tfm = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(M, S),
])

# 数据
print('[数据] 加载数据（224x224）...')
train_aug = CustomDataset('./images/train.txt', './images/train', lambda: train_tfm)
val_raw = CustomDataset('./images/train.txt', './images/train', lambda: val_tfm)
n = len(train_aug)
idx = torch.randperm(n)
sp = int(0.9 * n)
train_ds = Subset(train_aug, idx[:sp].tolist())
val_ds = Subset(val_raw, idx[sp:].tolist())
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, BATCH_SIZE)

# ====== 模型：完整的ResNet18（不做任何修改）======
print('[模型] 加载预训练ResNet18...')
backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# 冻结所有层（不计算梯度）
for param in backbone.parameters():
    param.requires_grad = False

# 替换分类头（只有这部分可训练）
in_features = backbone.fc.in_features
backbone.fc = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(128, 10),
)

# 统计参数量
total = sum(p.numel() for p in backbone.parameters())
trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f'总参数量: {total:,}  可训练（仅分类头）: {trainable:,}')

backbone.to(device)

# ====== 训练 ======
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, backbone.parameters()),
                               lr=LR, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_val = float('inf')
for ep in range(EPOCH):
    # 训练
    backbone.train()
    t_loss, correct, total = 0, 0, 0
    for data in train_loader:
        imgs, lbls = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        out = backbone(imgs)
        loss = loss_fn(out, lbls)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        _, pred = torch.max(out, 1)
        total += lbls.size(0)
        correct += (pred == lbls).sum().item()

    # 验证
    backbone.eval()
    v_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for data in val_loader:
            imgs, lbls = data['image'].to(device), data['label'].to(device)
            out = backbone(imgs)
            loss = loss_fn(out, lbls)
            v_loss += loss.item()
            _, pred = torch.max(out, 1)
            v_total += lbls.size(0)
            v_correct += (pred == lbls).sum().item()

    avg_v = v_loss / len(val_loader)
    scheduler.step(avg_v)

    print(f'Epoch {ep+1}/{EPOCH} | '
          f'Train: {100.0*correct/total:.1f}% ({correct}/{total}) | '
          f'Val: {100.0*v_correct/v_total:.1f}% ({v_correct}/{v_total}) | '
          f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

    if avg_v < best_val:
        best_val = avg_v
        torch.save(backbone, './models/model.pkl')
        print(f'  [OK] 保存模型')

# 测试
print('\n[测试]')
test_tfm = Compose([Resize((224,224)), ToTensor(), Normalize(M, S)])
test_loader = DataLoader(CustomDataset('./images/test.txt', './images/test', lambda: test_tfm), BATCH_SIZE)
backbone.eval()
correct, total = 0, 0
with torch.no_grad():
    for data in test_loader:
        imgs, lbls = data['image'].to(device), data['label'].to(device)
        _, pred = torch.max(backbone(imgs), 1)
        total += lbls.size(0)
        correct += (pred == lbls).sum().item()
print(f'测试集: {correct}/{total} = {100.0*correct/total:.2f}%')
