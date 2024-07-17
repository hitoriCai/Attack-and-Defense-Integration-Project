# 实现fgsm对抗训练与pgd对抗训练，二者的区别在于迭代次数、步长、初始化等

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from dataset import ImageNetLoader
# from model import resnet50, NormalizeByChannelMeanStd, ProcessedModel

# import torch.nn.functional as F
# import attack


# def train(model, device, train_loader, optimizer, epoch, epsilon, attack_type='fgsm'):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)

#         # 如果是对抗训练，先攻击
#         if attack_type == 'fgsm':
#             data_grad = data.clone().detach().requires_grad_(True)
#             output = model(data_grad)
#             loss = F.cross_entropy(output, target)
#             model.zero_grad()
#             loss.backward()
#             data = fgsm_attack(data_grad, epsilon, data_grad.grad.data)
#         elif attack_type == 'pgd':
#             data = pgd_attack(model, data, target, device, epsilon, alpha=epsilon/10, iters=40)

#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.cross_entropy(output, target)
#         loss.backward()
#         optimizer.step()

#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 初始化数据加载器
#     train_loader = ImageNetLoader('path/to/your/imagenet/data', batch_size=32, train=True).load_data()

#     # 初始化模型
#     model = resnet50(pretrained=True)
#     model = ProcessedModel(model, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).to(device)

#     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#     # 对抗训练参数
#     epsilon = 0.03  # 攻击强度

#     for epoch in range(1, 11):
#         train(model, device, train_loader, optimizer, epoch, epsilon, attack_type='fgsm')
#         train(model, device, train_loader, optimizer, epoch, epsilon, attack_type='pgd')

# if __name__ == '__main__':
#     main()




#     #可以通过这种方式得到FGSM的对抗样本
#     attack = attack.FGSM(net, eps=8 / 255)
#     image = attack(image, label)

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ImageNetLoader
from models import resnet50, NormalizeByChannelMeanStd, ProcessedModel
import attack 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(model, device, train_loader, optimizer, epoch, eps):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 使用您的attack库生成FGSM对抗样本
        fgsm = attack.FGSM(model, eps=eps)
        perturbed_data = fgsm(data, target)

        # 清零梯度
        optimizer.zero_grad()
        # 使用对抗样本进行训练
        output = model(perturbed_data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on {device}')

    data_dir = '/data0/mxy/imagenet/ILSVRC2012'
    train_loader = ImageNetLoader(data_dir, batch_size=32, train=True).load_data()

    model = ProcessedModel(resnet50(pretrained=True), NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    num_epochs = 3
    epsilon = 8 / 255  # 对抗强度

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, epsilon)

if __name__ == '__main__':
    main()
