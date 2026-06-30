# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    test.py
# Version:      ver2_0
# Created:      2024/06/17
# Description:  本文件定义了模型的测试流程
#               改进：添加标准化，确保测试集评估准确
# -----------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from dataset import CustomDataset


def test(dataloader, model, device):
    """定义测试流程。"""
    model.eval()

    size = len(dataloader.dataset)
    correct_num = 0

    # START----------------------------------------------------------
    with torch.no_grad():
        for data in dataloader:
            images = data['image'].to(device)
            labels = data['label'].to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            correct_num += (predicted == labels).sum().item()

    accuracy = 100.0 * correct_num / size
    print(f'Test set: {correct_num}/{size} correct, Accuracy: {accuracy:.2f}%')
    # END------------------------------------------------------------


if __name__ == "__main__":
    # 加载训练好的模型
    model = torch.load('./models/model.pkl')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 测试集也需要标准化（与训练一致）
    test_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataloader = DataLoader(
        CustomDataset('./images/test.txt', './images/test', lambda: test_transform),
        batch_size=32,
    )
    test(test_dataloader, model, device)
