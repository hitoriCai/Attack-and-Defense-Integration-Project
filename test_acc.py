import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models import resnet18, resnet152, resnet101, NormalizeByChannelMeanStd, ProcessedModel

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18', 'resnet152', 'resnet101'],
                        help='model architecture: resnet18 | resnet152 | resnet101 (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    return parser

def evaluate_model(model, dataloader, criterion, device, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(dataloader), [batch_time, losses, top1, top5], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(dataloader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    return losses.avg, top1.avg

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.arch == "resnet18":
        base_model = resnet18()
    elif args.arch == "resnet152":
        base_model = resnet152()
    elif args.arch == "resnet101":
        base_model = resnet101()
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    data_normalizer = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = ProcessedModel(base_model, data_normalize=data_normalizer).to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading model from '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            checkpoint = remove_module_prefix(checkpoint)  # Remove 'module.' prefix
            model.load_state_dict(checkpoint)
        else:
            print(f"No checkpoint found at '{args.resume}'")
            return

    cudnn.benchmark = True

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

    criterion = nn.CrossEntropyLoss().to(device)

    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device, args.print_freq)

    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

if __name__ == '__main__':
    main()
