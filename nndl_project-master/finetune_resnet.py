# -*- coding: utf-8 -*- #
"""快速微调ResNet18，只需15轮"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import *
from dataset import CustomDataset
from model import ResNetGesture

# 用更短的时间（15轮）微调
EPOCH = 15
BATCH_SIZE = 32
LR = 3e-4

device = torch.device('cpu')
print(f'设备: {device}')

# ImageNet标准化
M, S = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tfm = Compose([
    RandomRotation(25),
    RandomAffine(0, translate=(0.15,0.15), scale=(0.8,1.2), shear=10),
    RandomPerspective(0.15, p=0.4),
    ColorJitter(0.25, 0.25, 0.25, 0.08),
    GaussianBlur(3, (0.1,1.0)),
    RandomGrayscale(0.1),
    Resize((64,64)),
    ToTensor(),
    Normalize(M, S),
])
val_tfm = Compose([Resize((64,64)), ToTensor(), Normalize(M, S)])

# 数据
train_aug = CustomDataset('./images/train.txt', './images/train', lambda: train_tfm)
val_raw = CustomDataset('./images/train.txt', './images/train', lambda: val_tfm)

n = len(train_aug)
idx = torch.randperm(n)
sp = int(0.9 * n)
train_ds = Subset(train_aug, idx[:sp].tolist())
val_ds = Subset(val_raw, idx[sp:].tolist())

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, BATCH_SIZE)

# 模型
model = ResNetGesture(10)
# 如果已有部分训练的权重，继续微调
try:
    old = torch.load('./models/model.pkl', map_location='cpu', weights_only=False)
    if isinstance(old, ResNetGesture):
        model.load_state_dict(old.state_dict())
        print('加载了已有权重继续微调')
except:
    print('从头开始')

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-5)
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

best_val = float('inf')
for ep in range(EPOCH):
    # 训练
    model.train()
    train_loss, correct, total = 0, 0, 0
    for data in train_loader:
        imgs, lbls = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(imgs), lbls)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = torch.max(model(imgs), 1)
        total += lbls.size(0)
        correct += (pred == lbls).sum().item()

    # 验证
    model.eval()
    val_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for data in val_loader:
            imgs, lbls = data['image'].to(device), data['label'].to(device)
            out = model(imgs)
            loss = loss_fn(out, lbls)
            val_loss += loss.item()
            _, pred = torch.max(out, 1)
            v_total += lbls.size(0)
            v_correct += (pred == lbls).sum().item()

    avg_val = val_loss / len(val_loader)
    scheduler.step(avg_val)

    print(f'Epoch {ep+1}/{EPOCH} | Train: {100.0*correct/total:.1f}% | Val: {100.0*v_correct/v_total:.1f}% | LR: {optimizer.param_groups[0]["lr"]:.2e}')

    if avg_val < best_val:
        best_val = avg_val
        torch.save(model, './models/model.pkl')
        print(f'  [OK] 保存模型')

# 测试
test_tfm = Compose([Resize((64,64)), ToTensor(), Normalize(M, S)])
test_loader = DataLoader(CustomDataset('./images/test.txt', './images/test', lambda: test_tfm), 32)
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for data in test_loader:
        imgs, lbls = data['image'].to(device), data['label'].to(device)
        _, pred = torch.max(model(imgs), 1)
        total += lbls.size(0)
        correct += (pred == lbls).sum().item()
print(f'\n测试集: {correct}/{total} = {100.0*correct/total:.2f}%')
