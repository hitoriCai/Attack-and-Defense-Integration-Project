# 实现fgsm对抗训练与pgd对抗训练，二者的区别在于迭代次数、步长、初始化等

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ImageNetLoader
from model import resnet50, NormalizeByChannelMeanStd, ProcessedModel

import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建对抗性例子
    perturbed_image = image + epsilon * sign_data_grad
    # 添加剪切以维持数据范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_attack(model, images, labels, device, epsilon, alpha, iters=3):
    # 初始化扰动
    perturbed_images = images + torch.randn_like(images) * epsilon
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    for _ in range(iters):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        with torch.no_grad():
            # 通过梯度符号进行攻击
            perturbed_images = perturbed_images + alpha * perturbed_images.grad.sign()
            # 将扰动量限制在epsilon范围内并进行剪裁
            perturbed_images = torch.max(torch.min(perturbed_images, images + epsilon), images - epsilon)
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images

def train(model, device, train_loader, optimizer, epoch, epsilon, attack_type='fgsm'):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 如果是对抗训练，先攻击
        if attack_type == 'fgsm':
            data_grad = data.clone().detach().requires_grad_(True)
            output = model(data_grad)
            loss = F.cross_entropy(output, target)
            model.zero_grad()
            loss.backward()
            data = fgsm_attack(data_grad, epsilon, data_grad.grad.data)
        elif attack_type == 'pgd':
            data = pgd_attack(model, data, target, device, epsilon, alpha=epsilon/10, iters=3)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化数据加载器
    train_loader = ImageNetLoader('path/to/your/imagenet/data', batch_size=32, train=True).load_data()

    # 初始化模型
    model = resnet50(pretrained=True)
    model = ProcessedModel(model, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 对抗训练参数
    epsilon = 0.03  # 攻击强度

    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch, epsilon, attack_type='fgsm')
        train(model, device, train_loader, optimizer, epoch, epsilon, attack_type='pgd')

if __name__ == '__main__':
    main()
