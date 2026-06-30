# -*- coding: utf-8 -*- #
"""
混合训练：原始数据 + 实拍照片
CPU优化版：更快、更轻量、实时输出进度
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import *
from dataset import CustomDataset
from model import CustomNet
import os
import sys

# ====== 进度写入 ======
PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_progress.txt')

def log(msg):
    """同时输出到终端和进度文件"""
    print(msg)
    sys.stdout.flush()
    try:
        with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
            f.flush()
    except:
        pass


def train():
    # ====== CPU优化参数 ======
    BATCH_SIZE = 128            # 增大batch减少迭代次数
    LR = 5e-4
    EPOCH_BASE = 10             # 原始数据训练
    EPOCH_FINETUNE = 10         # 实拍微调
    VAL_EVERY = 2               # 每N轮验证一次（节省CPU时间）

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_TXT = os.path.join(SCRIPT_DIR, 'images', 'train.txt')
    real_photos_file = os.path.join(SCRIPT_DIR, 'images', 'real_photos.txt')

    log('=' * 50)
    log('开始训练流程')
    log(f'设备: {device}  {"[GPU]" if torch.cuda.is_available() else "[CPU] 无GPU"}')
    log(f'Batch: {BATCH_SIZE}  Epoch: {EPOCH_BASE+EPOCH_FINETUNE}')
    log('=' * 50)

    # ====== 数据增强 ======
    base_tfm = Compose([
        RandomRotation(degrees=25),
        RandomAffine(0, translate=(0.12,0.12), scale=(0.8,1.2), shear=10),
        RandomPerspective(0.15, p=0.4),
        ColorJitter(0.25, 0.25, 0.25, 0.08),
        GaussianBlur(3, (0.1, 1.0)),
        RandomGrayscale(0.1),
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])

    real_tfm = Compose([
        RandomRotation(degrees=35),
        RandomAffine(0, translate=(0.15,0.15), scale=(0.7,1.3), shear=15),
        RandomPerspective(0.2, p=0.5),
        ColorJitter(0.3, 0.3, 0.3, 0.1),
        GaussianBlur(3, (0.1, 1.5)),
        RandomGrayscale(0.15),
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    val_tfm = Compose([Resize((64,64)), ToTensor(), Normalize([0.5]*3, [0.5]*3)])

    # ====== 数据加载 ======
    log('\n[1/5] 加载数据...')
    base_dataset = CustomDataset(TRAIN_TXT, os.path.join(SCRIPT_DIR, 'images', 'train'), lambda: base_tfm)
    has_real = os.path.exists(real_photos_file) and os.path.getsize(real_photos_file) > 0
    total_epochs = EPOCH_BASE + (EPOCH_FINETUNE if has_real else 0)

    log(f'原始数据: {len(base_dataset)} 张')

    if has_real:
        real_dataset = CustomDataset(real_photos_file, SCRIPT_DIR, lambda: real_tfm)
        n_real = len(real_dataset)
        # 合理重复：最多重复3倍，避免训练集过大
        repeat = min(3, max(1, len(base_dataset) // n_real // 10))
        from torch.utils.data import ConcatDataset
        combined = ConcatDataset([base_dataset] + [real_dataset] * repeat)
        log(f'实拍照片: {n_real} 张（重复{repeat}倍）')
    else:
        combined = base_dataset
        log('未发现实拍照片')

    # 验证集（只取10%）
    val_dataset = CustomDataset(TRAIN_TXT, os.path.join(SCRIPT_DIR, 'images', 'train'), lambda: val_tfm)
    n = len(val_dataset)
    idx = torch.randperm(n)
    sp = int(0.9 * n)
    val_ds = Subset(val_dataset, idx[sp:].tolist())

    train_loader = DataLoader(combined, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE * 2, shuffle=False, num_workers=0)

    log(f'每轮训练: {len(combined)} 样本 / {len(train_loader)} batch')
    log(f'验证集: {len(val_ds)} 样本')

    # ====== 模型 ======
    log('\n[2/5] 初始化模型...')
    model = CustomNet(10).to(device)
    log(f'CustomNet v2, 参数量: {sum(p.numel() for p in model.parameters()):,}')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # ====== 训练 ======
    log('\n[3/5] 开始训练...')
    best_val = float('inf')
    steps = 0

    for ep in range(total_epochs):
        phase = '基础训练' if ep < EPOCH_BASE else '实拍微调'

        # ====== 训练 ======
        model.train()
        correct = total = 0
        train_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            imgs, lbls = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(model(imgs), 1)
            total += lbls.size(0)
            correct += (pred == lbls).sum().item()
            steps += 1

            # 每50个batch打印一次进度
            if (batch_idx + 1) % 50 == 0:
                avg = 100.0 * correct / total if total > 0 else 0
                log(f'  Epoch {ep+1}/{total_epochs} [{phase}] batch {batch_idx+1}/{len(train_loader)} | 当前准确率: {avg:.1f}%')

        train_acc = 100.0 * correct / total

        # ====== 验证（每VAL_EVERY轮做一次）======
        if (ep + 1) % VAL_EVERY == 0 or ep == total_epochs - 1:
            model.eval()
            v_loss = v_correct = v_total = 0
            with torch.no_grad():
                for data in val_loader:
                    imgs, lbls = data['image'].to(device), data['label'].to(device)
                    out = model(imgs)
                    v_loss += loss_fn(out, lbls).item()
                    _, pred = torch.max(out, 1)
                    v_total += lbls.size(0)
                    v_correct += (pred == lbls).sum().item()

            avg_v = v_loss / len(val_loader)
            val_acc = 100.0 * v_correct / v_total
            lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_v)

            log(f'  >> Epoch {ep+1}/{total_epochs} [{phase}] | '
                f'训练: {train_acc:.1f}% | 验证: {val_acc:.1f}% | LR: {lr:.2e}')

            if avg_v < best_val:
                best_val = avg_v
                torch.save(model, './models/model.pkl')
                log(f'     -> 模型已保存 (验证损失: {avg_v:.4f})')
        else:
            log(f'  >> Epoch {ep+1}/{total_epochs} [{phase}] | 训练: {train_acc:.1f}% (下轮验证)')

    # ====== 测试 ======
    log('\n[4/5] 测试集评估...')
    test_loader = DataLoader(
        CustomDataset('./images/test.txt', './images/test', lambda: val_tfm),
        BATCH_SIZE, num_workers=0)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, lbls = data['image'].to(device), data['label'].to(device)
            _, pred = torch.max(model(imgs), 1)
            total += lbls.size(0)
            correct += (pred == lbls).sum().item()
    test_acc = 100.0 * correct / total
    log(f'测试集准确率: {correct}/{total} = {test_acc:.2f}%')

    # ====== 完成 ======
    log('\n[5/5] 训练完成！')
    log(f'模型已保存至 models/model.pkl')
    with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
        f.write('__TRAINING_COMPLETE__\n')


if __name__ == '__main__':
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    try:
        train()
    except Exception as e:
        import traceback
        log(f'\n[错误] 训练失败: {e}')
        log(traceback.format_exc())
        with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
            f.write(f'__TRAINING_FAILED__: {e}\n')
