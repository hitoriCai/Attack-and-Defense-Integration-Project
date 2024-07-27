import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import resnet18, NormalizeByChannelMeanStd, ProcessedModel
from torch.utils.data import DataLoader 

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

def main():
    parser = argparse.ArgumentParser(description='Evaluate model accuracy')
    parser.add_argument('--data', metavar='DIR', help='path to dataset', required=True)
    parser.add_argument('--batch-size', default=256, type=int, help='mini-batch size')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--model-path', default='450.pt', type=str, help='path to the model checkpoint')
    parser.add_argument('--arch', default='resnet18', type=str, help='model architecture')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    base_model = resnet18()
    data_normalizer = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = ProcessedModel(base_model, data_normalize=data_normalizer).to(device)

    # Load model weights
    if os.path.isfile(args.model_path):
        print(f"Loading model from '{args.model_path}'")
        state_dict = torch.load(args.model_path, map_location=device)
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict)
    else:
        print(f"No model found at '{args.model_path}'")
        return

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    val_loader = DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Define loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Evaluate model
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

if __name__ == '__main__':
    main()