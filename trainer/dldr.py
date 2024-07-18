# 实现低维训练，包括对抗训练与普通训练
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import ImageNetLoader
from models import resnet18, NormalizeByChannelMeanStd, ProcessedModel
import numpy as np
from sklearn.decomposition import PCA
from PIL import ImageFile
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

def sub_AT_attack(model, images, labels, epsilon, n_components=5):
    model.eval()
    images = images.detach().cpu().numpy()
    orig_shape = images.shape
    images_flat = images.reshape(images.shape[0], -1)
    pca = PCA(n_components=n_components)
    pca.fit(images_flat)
    images_pca = pca.transform(images_flat)
    perturbations = epsilon * np.sign(images_pca)
    perturbed_images_pca = images_pca + perturbations
    perturbed_images_flat = pca.inverse_transform(perturbed_images_pca)
    perturbed_images = perturbed_images_flat.reshape(orig_shape)
    perturbed_images = np.clip(perturbed_images, 0, 1)
    perturbed_images_tensor = torch.tensor(perturbed_images).float().to(images.device)
    model.train()
    return perturbed_images_tensor

def parse_args():
    parser = argparse.ArgumentParser(description='Low Dimensional Training with Normal and Adversarial Training')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Perturbation magnitude for adversarial training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for dataset')
    return parser.parse_args()

def train_normal(model, device, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
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
    print(f"Normal Training: Loss: {total_loss / len(dataloader)}, Accuracy: {correct / total:.2f}")

def train_sub_AT(model, device, dataloader, criterion, optimizer, epsilon):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        adv_images = sub_AT_attack(model, images, labels, epsilon)
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    print(f"Adversarial Training: Loss: {total_loss / len(dataloader)}, Accuracy: {correct / total:.2f}")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    data_dir = '/data0/mxy/imagenet/ILSVRC2012'
    train_loader = ImageNetLoader(data_dir, batch_size=128, train=True).load_data()


    model = ProcessedModel(resnet18, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)

    # Warm-up phase
    warmup_epochs = 3
    for epoch in range(args.epochs):
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scale * args.lr

        print(f"Epoch {epoch+1}")
        train_normal(model, device, train_loader, criterion, optimizer)
        # train_sub_AT(model, device, train_loader, criterion, optimizer, args.epsilon)
        scheduler.step()

        model_path = '/path/to/save/models/'
        torch.save(model.module.state_dict(), os.path.join(model_path, f'model_epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    main()

