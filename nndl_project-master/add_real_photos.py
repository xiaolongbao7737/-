# -*- coding: utf-8 -*- #
"""
将实拍手势照片加入训练数据集

用法：
  python add_real_photos.py                    # 从 ../实拍手势照片/ 添加
  python add_real_photos.py --dir 照片目录     # 从指定目录添加
  python add_real_photos.py --clear            # 清空已添加的实拍照片

照片命名规则：文件名第一个数字 = 实际手势数字
  - 1.jpg, 1 (2).jpg → 数字 1
  - 2.jpg, 2 (1).jpg → 数字 2
  - 以此类推
"""

import os
import sys
import shutil
import argparse
from PIL import Image

# 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(SCRIPT_DIR, 'images', 'train', 'real')
LABELS_FILE = os.path.join(SCRIPT_DIR, 'images', 'real_photos.txt')
TRAIN_TXT = os.path.join(SCRIPT_DIR, 'images', 'train.txt')


def extract_digit_from_name(filename):
    """从文件名中提取数字（文件名第一个数字字符）"""
    name = os.path.splitext(filename)[0]
    for ch in name:
        if ch.isdigit() and '0' <= ch <= '9':
            return int(ch)
    return None


def add_photos(source_dir, copy=True):
    """将目录中的照片添加到训练集

    Args:
        source_dir: 源照片目录
        copy: True=复制, False=移动
    """
    if not os.path.exists(source_dir):
        print(f"[错误] 目录不存在: {source_dir}")
        return 0

    os.makedirs(REAL_DIR, exist_ok=True)

    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    added = 0
    skipped = 0
    new_entries = []

    for fname in sorted(os.listdir(source_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported:
            continue

        digit = extract_digit_from_name(fname)
        if digit is None:
            print(f"  ⚠ 跳过 {fname}: 无法从文件名识别手势数字")
            skipped += 1
            continue

        src_path = os.path.join(source_dir, fname)
        dst_name = f'{digit}_{added:04d}_{fname}'
        dst_path = os.path.join(REAL_DIR, dst_name)

        try:
            # 验证图片
            img = Image.open(src_path)
            img.verify()

            if copy:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

            new_entries.append(f'./images/train/real/{dst_name} {digit}')
            added += 1
            print(f'  [OK] 数字{digit}: {fname} -> {dst_name}')

        except Exception as e:
            print(f'  [FAIL] {fname}: {e}')
            skipped += 1

    # 写入标注文件
    if new_entries:
        with open(LABELS_FILE, 'a', encoding='utf-8') as f:
            for entry in new_entries:
                f.write(entry + '\n')
        print(f'\n已添加 {added} 张实拍照片到 {REAL_DIR}')
        print(f'标注文件: {LABELS_FILE}')
    else:
        print('\n没有新照片可添加')

    return added


def merge_into_trainset():
    """将实拍照片标注合并到 train.txt 中（去重）"""
    if not os.path.exists(LABELS_FILE):
        print("[提示] 没有实拍照片标注文件，跳过合并")
        return

    # 读取现有 train.txt
    with open(TRAIN_TXT, 'r', encoding='utf-8') as f:
        existing = set(line.strip() for line in f.readlines())

    # 读取实拍照片标注
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        real_entries = [line.strip() for line in f.readlines() if line.strip()]

    # 去重合并
    new_count = 0
    with open(TRAIN_TXT, 'a', encoding='utf-8') as f:
        for entry in real_entries:
            if entry not in existing:
                f.write(entry + '\n')
                new_count += 1

    if new_count > 0:
        print(f'已将 {new_count} 条实拍记录合并到 {TRAIN_TXT}')
    else:
        print('所有实拍记录已存在，无需重复添加')


def clear_real_photos():
    """清空已添加的实拍照片"""
    if os.path.exists(REAL_DIR):
        for f in os.listdir(REAL_DIR):
            os.remove(os.path.join(REAL_DIR, f))
        os.rmdir(REAL_DIR)
        print(f'已清空 {REAL_DIR}')

    if os.path.exists(LABELS_FILE):
        os.remove(LABELS_FILE)
        print(f'已删除 {LABELS_FILE}')

    # 从 train.txt 中移除实拍照片条目
    if os.path.exists(TRAIN_TXT):
        with open(TRAIN_TXT, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(TRAIN_TXT, 'w', encoding='utf-8') as f:
            for line in lines:
                if '/real/' not in line:
                    f.write(line)
        print(f'已从 {TRAIN_TXT} 中移除实拍照片条目')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='添加实拍手势照片到训练集')
    parser.add_argument('--dir', default=os.path.join(SCRIPT_DIR, '..', '实拍手势照片'),
                        help='照片目录路径')
    parser.add_argument('--clear', action='store_true',
                        help='清空已添加的实拍照片')
    parser.add_argument('--no-merge', action='store_true',
                        help='不合并到train.txt')
    args = parser.parse_args()

    if args.clear:
        clear_real_photos()
        sys.exit(0)

    # 添加照片
    source = os.path.abspath(args.dir)
    print(f'扫描目录: {source}')
    count = add_photos(source, copy=True)

    if count > 0 and not args.no_merge:
        print('\n--- 合并到训练集 ---')
        merge_into_trainset()

    print(f'\n完成！共添加 {count} 张实拍照片')
