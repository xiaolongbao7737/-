# -*- coding: utf-8 -*- #
"""
实拍照片预处理 — 中心裁剪手部区域，替换为统一背景
让输入图片风格更接近训练数据集
"""
import cv2
import numpy as np
from PIL import Image


def extract_hand(image, target_size=(64, 64)):
    """从实拍照片中提取手部，放置在统一背景上

    策略：
    1. 先尝试肤色检测找出最大手部区域
    2. 若失败则用中心裁剪（默认手在照片中心）
    3. 提取到手部后缩放到合适大小，居中放在统一背景上
    """
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 肤色检测（放宽HSV范围覆盖不同光照）
    lower_skin = np.array([0, 15, 50], dtype=np.uint8)
    upper_skin = np.array([30, 180, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 找最大轮廓（手部）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 8000:
        # 肤色检测成功 → 取手部边界框 + padding
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        pad = 30
        x = max(0, x - pad)
        y = max(0, y - pad)
        bw = min(w - x, bw + 2*pad)
        bh = min(h - y, bh + 2*pad)
        crop = img[y:y+bh, x:x+bw]
    else:
        # 回退：中心裁剪（取中间40%区域）
        cx, cy = w // 2, h // 2
        s = min(w, h) // 3
        crop = img[cy-s:cy+s, cx-s:cx+s]

    # 缩放到合适大小，居中放在深色背景上
    bg = np.full((target_size[0], target_size[1], 3), (35, 50, 75), dtype=np.uint8)
    ch, cw = crop.shape[:2]
    scale = min(target_size[0]*0.65/cw, target_size[1]*0.65/ch)
    nw, nh = int(cw*scale), int(ch*scale)
    resized = cv2.resize(crop, (nw, nh))
    xo, yo = (target_size[1]-nw)//2, (target_size[0]-nh)//2
    bg[yo:yo+nh, xo:xo+nw] = resized

    return Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = extract_hand(Image.open(sys.argv[1]))
        result.save("hand_extracted.png")
        print(f"预处理完成，已保存 hand_extracted.png")
