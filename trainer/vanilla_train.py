# 实现正常的训练过程

import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataset import ImageNetLoader
from models import resnet50, NormalizeByChannelMeanStd, ProcessedModel

def train(model, device, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

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
    train_loader = ImageNetLoader(data_dir, batch_size=32, train=True).load_data()
    test_loader = ImageNetLoader(data_dir, batch_size=32, train=False).load_data()

    base_model = resnet50(pretrained=True)
    data_normalizer = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = ProcessedModel(base_model, data_normalize=data_normalizer).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        
        test_loss, test_acc = test(model, device, test_loader, criterion)
        print(f'Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        
        scheduler.step()
        
        # Save the model
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()

