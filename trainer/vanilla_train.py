# 实现正常的训练过程

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ImageNetLoader
from models import resnet18, NormalizeByChannelMeanStd, ProcessedModel
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(model, device, dataloader, criterion, optimizer, scheduler, epoch, warmup, total_steps):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update the learning rate with warmup
        if warmup > 0:
            warmup_step_lr(optimizer, epoch, batch_idx, len(dataloader), warmup, total_steps)
        else:
            scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            current = batch_idx * len(images)
            total_images = len(dataloader.dataset)
            percent = 100. * batch_idx / len(dataloader)
            print(f'Train Epoch: {epoch} [{current}/{total_images} ({percent:.0f}%)]\tLoss: {loss.item():.6f}, Accuracy: {100. * correct / total:.2f}%')

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def warmup_step_lr(optimizer, epoch, batch_idx, total_batches, warmup, total_steps):
    """ Update learning rate of optims[i] for warmup during training."""
    current_step = epoch * total_batches + batch_idx + 1
    warmup_steps = total_batches * warmup
    if current_step < warmup_steps:
        lr_scale = (current_step / warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_scale * param_group['initial_lr']

def test(model, device, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    data_dir = '/opt/data/common/ILSVRC2012/'
    train_loader = ImageNetLoader(data_dir, batch_size=128, train=True).load_data()
    test_loader = ImageNetLoader(data_dir, batch_size=128, train=False).load_data()

    base_model = resnet18
    data_normalizer = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = ProcessedModel(base_model, data_normalize=data_normalizer)
    model = nn.DataParallel(model).to(device)  # Wrap the model for multi-GPU training

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.06, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    num_epochs = 10
    warmup = 5
    total_steps = num_epochs * len(train_loader)

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, scheduler, epoch, warmup, total_steps)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        
        test_loss, test_acc = test(model, device, test_loader, criterion)
        print(f'Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        
        model_path = '/opt/data/private/trained_model/vanilla_train/'
        torch.save(model.module.state_dict(), os.path.join(model_path, f'model_epoch_{epoch+1}.pth'))


if __name__ == '__main__':
    main()



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from dataset import ImageNetLoader
# from models import resnet50, NormalizeByChannelMeanStd, ProcessedModel
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# def train(model, device, dataloader, criterion, optimizer, scheduler, epoch, warmup, total_steps):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (images, labels) in enumerate(dataloader):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # Update the learning rate with warmup
#         if warmup > 0:
#             warmup_step_lr(optimizer, epoch, batch_idx, len(dataloader), warmup, total_steps)
#         else:
#             scheduler.step()

#         total_loss += loss.item()
#         _, predicted = outputs.max(1)
#         correct += predicted.eq(labels).sum().item()
#         total += labels.size(0)

#         if (batch_idx + 1) % 100 == 0:
#             current = batch_idx * len(images)
#             total_images = len(dataloader.dataset)
#             percent = 100. * batch_idx / len(dataloader)
#             print(f'Train Epoch: {epoch} [{current}/{total_images} ({percent:.0f}%)]\tLoss: {loss.item():.6f}, Accuracy: {100. * correct / total:.2f}%')

#     avg_loss = total_loss / len(dataloader)
#     accuracy = 100. * correct / total
#     return avg_loss, accuracy

# def warmup_step_lr(optimizer, epoch, batch_idx, total_batches, warmup, total_steps):
#     """ Update learning rate of optims[i] for warmup during training."""
#     current_step = epoch * total_batches + batch_idx + 1
#     warmup_steps = total_batches * warmup
#     if current_step < warmup_steps:
#         # Scale the learning rate linearly with the number of steps.
#         lr_scale = (current_step / warmup_steps)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr_scale * param_group['initial_lr']

# def test(model, device, dataloader, criterion):
#     model.eval()  # 设置模型为评估模式
#     total_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():  # 在评估时不计算梯度
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             _, predicted = outputs.max(1)
#             correct += predicted.eq(labels).sum().item()
#             total += labels.size(0)
    
#     avg_loss = total_loss / len(dataloader)
#     accuracy = 100. * correct / total
#     return avg_loss, accuracy

# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'Training on {device}')

#     data_dir = '/opt/data/common/ILSVRC2012/'
#     train_loader = ImageNetLoader(data_dir, batch_size=64, train=True).load_data()
#     test_loader = ImageNetLoader(data_dir, batch_size=64, train=False).load_data()

#     base_model = resnet50(pretrained=False)
#     data_normalizer = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     model = ProcessedModel(base_model, data_normalize=data_normalizer).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.06, momentum=0.9)
#     scheduler = CosineAnnealingLR(optimizer, T_max=10)  # T_max is typically set to the number of epochs

#     num_epochs = 10
#     warmup = 5  # Number of epochs to warm up
#     total_steps = num_epochs * len(train_loader)

#     for epoch in range(num_epochs):
#         train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, scheduler, epoch, warmup, total_steps)
#         print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        
#         test_loss, test_acc = test(model, device, test_loader, criterion)
#         print(f'Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        
#         torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

# if __name__ == '__main__':
#     main()




# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# from dataset import ImageNetLoader
# from models import resnet50, NormalizeByChannelMeanStd, ProcessedModel
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True



# def train(model, device, dataloader, criterion, optimizer, scheduler, epoch):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (images, labels) in enumerate(dataloader):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#         total_loss += loss.item()
#         _, predicted = outputs.max(1)
#         correct += predicted.eq(labels).sum().item()
#         total += labels.size(0)

#         if (batch_idx + 1) % 100 == 0:  # 每100个batch打印一次
#             current = batch_idx * len(images)  # 计算已处理的图像总数
#             total_images = len(dataloader.dataset)  # 数据集中的总图像数
#             percent = 100. * batch_idx / len(dataloader)  # 计算百分比
#             print(f'Train Epoch: {epoch} [{current}/{total_images} ({percent:.0f}%)]\tLoss: {loss.item():.6f}, Accuracy: {100. * correct / total:.2f}%')

#     avg_loss = total_loss / len(dataloader)
#     accuracy = 100. * correct / total
#     return avg_loss, accuracy

# def test(model, device, dataloader, criterion):
#     model.eval()  # 设置模型为评估模式
#     total_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():  # 在评估时不计算梯度
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             _, predicted = outputs.max(1)
#             correct += predicted.eq(labels).sum().item()
#             total += labels.size(0)
    
#     avg_loss = total_loss / len(dataloader)
#     accuracy = 100. * correct / total
#     return avg_loss, accuracy



# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'Training on {device}')

#     data_dir = '/opt/data/common/ILSVRC2012/'
#     train_loader = ImageNetLoader(data_dir, batch_size=64, train=True).load_data()
#     test_loader = ImageNetLoader(data_dir, batch_size=64, train=False).load_data()

#     base_model = resnet50(pretrained=False)
#     data_normalizer = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     model = ProcessedModel(base_model, data_normalize=data_normalizer).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.06, momentum=0.9)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#     num_epochs = 10
#     for epoch in range(num_epochs):
#         train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, scheduler, epoch)
#         print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        
#         test_loss, test_acc = test(model, device, test_loader, criterion)
#         print(f'Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        
        
#         # Save the model
#         torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

# if __name__ == '__main__':
#     main()



