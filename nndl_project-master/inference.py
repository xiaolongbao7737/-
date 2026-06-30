# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    inference.py
# Version:      ver2_0
# Created:      2024/06/17
# Description:  本文件定义了用于在模型应用端进行推理，返回模型输出的流程
#               改进：添加标准化，保证推理输入与训练数据分布一致
# -----------------------------------------------------------------------

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


def inference(image_path, model, device):
    """定义模型推理应用的流程。
    :param image_path: 输入图片的路径
    :param model: 训练好的模型
    :param device: 模型推理使用的设备
    """
    model.eval()

    # START----------------------------------------------------------
    # 1. 加载图片并转换为RGB格式
    image = Image.open(image_path).convert('RGB')

    # 2. 预处理：缩放 + 张量化 + 标准化（与训练时一致）
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(image)

    # 3. 添加batch维度: [C, H, W] -> [1, C, H, W]
    x = torch.unsqueeze(x, 0)

    # 4. 将数据移至指定设备
    x = x.to(device)

    # 5. 前向传播（不计算梯度）
    with torch.no_grad():
        output = model(x)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # 6. 输出预测结果
    predicted_digit = predicted.item()
    confidence_pct = confidence.item() * 100
    print(f'Predicted digit: {predicted_digit} (confidence: {confidence_pct:.2f}%)')

    return predicted_digit, confidence_pct
    # END------------------------------------------------------------


if __name__ == "__main__":
    # 指定图片路径
    image_path = "./images/test/signs/img_0006.png"

    # 加载训练好的模型
    model = torch.load('./models/model.pkl')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 显示图片，输出预测结果
    inference(image_path, model, device)
