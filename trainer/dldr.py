# 实现低维训练，包括对抗训练与普通训练

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models import resnet50, NormalizeByChannelMeanStd, ProcessedModel
import numpy as np
from sklearn.decomposition import PCA
from torch.autograd import Variable


def sub_at_attack(model, images, labels, epsilon, n_components=5):
    """
    Subspace Adversarial Training attack using PCA to find the main directions.
    
    Args:
        model (torch.nn.Module): The neural network model.
        images (torch.Tensor): Batch of images.
        labels (torch.Tensor): Batch of labels.
        epsilon (float): Maximum perturbation.
        n_components (int): Number of principal components for PCA.

    Returns:
        torch.Tensor: Adversarial examples.
    """
    model.eval()
    images = images.detach().cpu().numpy()
    
    # Flatten images to 2D for PCA
    orig_shape = images.shape
    images_flat = images.reshape(images.shape[0], -1)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(images_flat)
    
    # Project the images onto the principal components
    images_pca = pca.transform(images_flat)
    
    # Add perturbations along the principal components
    perturbations = epsilon * np.sign(images_pca)
    perturbed_images_pca = images_pca + perturbations
    
    # Transform back to the original space
    perturbed_images_flat = pca.inverse_transform(perturbed_images_pca)
    perturbed_images = perturbed_images_flat.reshape(orig_shape)
    
    # Clip to ensure the images are valid
    perturbed_images = np.clip(perturbed_images, 0, 1)
    
    # Convert back to tensor and copy to the same device as the model
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
    """Regular training process"""
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

def train_sub_at(model, device, dataloader, criterion, optimizer, epsilon):
    """Subspace Adversarial Training process"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        adv_images = sub_at_attack(model, images, labels, epsilon)
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
    model = get_model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}")
        train_normal(model, device, train_loader, criterion, optimizer)
        train_sub_at(model, device, train_loader, criterion, optimizer, args.epsilon)

if __name__ == "__main__":
    main()
