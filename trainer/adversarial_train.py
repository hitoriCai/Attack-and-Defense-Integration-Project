# 实现fgsm对抗训练与pgd对抗训练，二者的区别在于迭代次数、步长、初始化等

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import ImageNetLoader
from models import resnet18, NormalizeByChannelMeanStd, ProcessedModel
from attack import white_box
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(model, device, train_loader, optimizer, scheduler, epoch, eps, warmup=False):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        #FGSM
        fgsm = white_box.FGSM(model, eps=eps)
        perturbed_data = fgsm(data, target)

        # #使用PGD
        # fgsm = white_box.PGD(resnet18, eps=eps, alpha=1 / 255, steps=10, random_start=True)  # Linf
        # perturbed_data = fgsm(data, target)

        # Zero out the gradients
        optimizer.zero_grad()
        # Train using adversarial examples
        output = model(perturbed_data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            accuracy = 100. * correct / total
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.6f}, Accuracy: {accuracy:.2f}%')
            correct = 0
            total = 0  # Reset for the next 100 batches

        # Update scheduler only after warmup
        if warmup and epoch > 1:
            scheduler.step()

def test(model, device, dataloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(dataloader.dataset)
    accuracy = 100. * correct / total
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on {device}')

    data_dir = '/opt/data/common/ILSVRC2012/'
    train_loader = ImageNetLoader(data_dir, batch_size=128, train=True).load_data()
    test_loader = ImageNetLoader(data_dir, batch_size=128, train=False).load_data()  # Assuming a similar function for test data

    model = ProcessedModel(resnet18, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    for epoch in range(1, 11):  # Update the number of epochs here if needed
        train(model, device, train_loader, optimizer, scheduler, epoch, 8 / 255)  # EPSILON defined earlier
        test(model, device, test_loader, criterion)

        model_path = '/opt/data/private/trained_model/FGSM_train/'
        torch.save(model.module.state_dict(), os.path.join(model_path, f'model_epoch_{epoch}.pth'))

if __name__ == '__main__':
    main()



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from dataset import ImageNetLoader
# from models import resnet18, NormalizeByChannelMeanStd, ProcessedModel
# from attack import white_box 
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# def train(model, device, train_loader, optimizer, epoch, eps):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)

#         # 使用您的attack库生成FGSM对抗样本
#         fgsm = white_box.FGSM(model, eps=eps)
#         perturbed_data = fgsm(data, target)

#         # #使用PGD
#         # fgsm = white_box.PGD(net, eps=eps, alpha=1 / 255, steps=10, random_start=True)  # Linf
#         # perturbed_data = PGD(data, target)


#         # 清零梯度
#         optimizer.zero_grad()
#         # 使用对抗样本进行训练
#         output = model(perturbed_data)
#         loss = nn.CrossEntropyLoss()(output, target)
#         loss.backward()
#         optimizer.step()

#         if batch_idx % 100 == 0:
#             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f'Training on {device}')

#     data_dir = '/data0/mxy/imagenet/ILSVRC2012'
#     train_loader = ImageNetLoader(data_dir, batch_size=128, train=True).load_data()

#     model = ProcessedModel(resnet18, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).to(device)

#     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#     num_epochs = 10
#     epsilon = 8 / 255  # 对抗强度

#     for epoch in range(1, num_epochs + 1):
#         train(model, device, train_loader, optimizer, epoch, epsilon)

# if __name__ == '__main__':
#     main()


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from dataset import ImageNetLoader
# from model import resnet18, NormalizeByChannelMeanStd, ProcessedModel

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
#     model = resnet18(pretrained=True)
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
