# -*- coding: utf-8 -*- #
"""
CustomNet v2 + 极端数据增强（背景随机化）
核心改进：随机替换背景区域，强制模型关注手势形状而非背景
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import *
from PIL import Image, ImageDraw, ImageFilter
import random
import numpy as np
from dataset import CustomDataset
from model import CustomNet


class RandomBackground:
    """随机替换图像背景，破坏背景与手势的关联性

    将图像边缘区域替换为随机颜色/图案，
    只保留中心区域（手势主体），
    迫使模型关注手势形状而非背景。
    """
    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size

        # 随机生成背景
        bg_type = random.randint(0, 3)
        if bg_type == 0:
            bg = Image.new('RGB', (w, h), tuple(random.randint(0,255) for _ in range(3)))
        elif bg_type == 1:
            bg = Image.new('RGB', (w, h), (random.randint(0,60),) * 3)
            draw = ImageDraw.Draw(bg)
            for _ in range(random.randint(10, 30)):
                x = random.randint(0, w-1)
                y = random.randint(0, h-1)
                rw, rh = random.randint(5, 40), random.randint(5, 40)
                draw.rectangle([
                    min(x, w-rw-1), min(y, h-rh-1),
                    min(x+rw, w-1), min(y+rh, h-1)
                ], fill=tuple(random.randint(100,255) for _ in range(3)))
        elif bg_type == 2:
            # 随机渐变
            bg = Image.new('RGB', (w, h))
            c1, c2 = [random.randint(0,255) for _ in range(3)], [random.randint(0,255) for _ in range(3)]
            for y in range(h):
                r = int(c1[0] + (c2[0]-c1[0]) * y / h)
                g = int(c1[1] + (c2[1]-c1[1]) * y / h)
                b = int(c1[2] + (c2[2]-c1[2]) * y / h)
                draw = ImageDraw.Draw(bg)
                draw.line([(0,y), (w-1,y)], fill=(r,g,b))
        else:
            bg = Image.new('RGB', (w, h), (0,0,0))
            for _ in range(100):
                x, y = random.randint(0,w-1), random.randint(0,h-1)
                bg.putpixel((x,y), tuple(random.randint(0,255) for _ in range(3)))

        # 创建椭圆遮罩：保留中心区域（手势），替换边缘（背景）
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        cx, cy = w // 2, h // 2
        rx, ry = int(w * 0.40), int(h * 0.40)
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=8))

        return Image.composite(img, bg, mask)


# 超参数
BATCH_SIZE = 64
LR = 5e-4
EPOCH = 300
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# ====== 极端数据增强 ======
train_tfm = Compose([
    RandomBackground(p=0.7),         # 随机替换背景（核心改进）
    RandomRotation(degrees=45),      # 大角度旋转
    RandomAffine(
        degrees=0,
        translate=(0.2, 0.2),       # 大幅平移
        scale=(0.6, 1.4),           # 大幅缩放
        shear=20,                    # 错切
    ),
    RandomPerspective(distortion_scale=0.3, p=0.6),
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    RandomGrayscale(p=0.2),
    Resize((64, 64)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    RandomErasing(p=0.3, scale=(0.02, 0.2)),
])

val_tfm = Compose([
    Resize((64, 64)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 数据
print('[数据] 加载数据...')
train_aug = CustomDataset('./images/train.txt', './images/train', lambda: train_tfm)
val_raw = CustomDataset('./images/train.txt', './images/train', lambda: val_tfm)
n = len(train_aug)
idx = torch.randperm(n)
sp = int(0.9 * n)
train_ds = Subset(train_aug, idx[:sp].tolist())
val_ds = Subset(val_raw, idx[sp:].tolist())
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, BATCH_SIZE)

print(f'训练: {len(train_ds)}  验证: {len(val_ds)}')

# 模型
model = CustomNet(10).to(device)
print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)

best_val = float('inf')
for ep in range(EPOCH):
    # 训练
    model.train()
    t_loss, correct, total = 0, 0, 0
    for data in train_loader:
        imgs, lbls = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(imgs), lbls)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        _, pred = torch.max(model(imgs), 1)
        total += lbls.size(0)
        correct += (pred == lbls).sum().item()

    # 验证
    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for data in val_loader:
            imgs, lbls = data['image'].to(device), data['label'].to(device)
            out = model(imgs)
            v_loss += loss_fn(out, lbls).item()
            _, pred = torch.max(out, 1)
            v_total += lbls.size(0)
            v_correct += (pred == lbls).sum().item()

    avg_v = v_loss / len(val_loader)
    scheduler.step(avg_v)

    print(f'Epoch {ep+1}/{EPOCH} | '
          f'Train: {100.0*correct/total:.1f}% | '
          f'Val: {100.0*v_correct/v_total:.1f}% | '
          f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

    if avg_v < best_val:
        best_val = avg_v
        torch.save(model, './models/model.pkl')
        print(f'  [OK] 保存模型（验证损失: {avg_v:.4f}）')

# 测试
print('\n[测试]')
from test import test
test_loader = DataLoader(
    CustomDataset('./images/test.txt', './images/test', lambda: val_tfm),
    BATCH_SIZE)
test(test_loader, model, device)
