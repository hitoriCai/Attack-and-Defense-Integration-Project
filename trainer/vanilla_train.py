# 实现正常的训练过程

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append("..")
from dataset import ImageNetLoader
from models import resnet50, NormalizeByChannelMeanStd, ProcessedModel

def train(model, dataloader, criterion, optimizer, num_epochs=3):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs} Loss: {running_loss / len(dataloader)}')

def main():
    # 设备配置
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    # 数据目录和加载器
    data_dir = 'path/to/your/imagenet/data'
    train_loader = ImageNetLoader(data_dir, batch_size=32, train=True).load_data()

    # 初始化模型
    base_model = resnet50(pretrained=True)
    data_normalizer = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = ProcessedModel(base_model, data_normalize=data_normalizer).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练模型
    train(model, train_loader, criterion, optimizer)

if __name__ == '__main__':
    main()
