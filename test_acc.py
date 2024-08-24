import argparse
import time
import os
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.append('../models')
sys.path.append('../dataset')
from models import resnet18, NormalizeByChannelMeanStd, ProcessedModel
from dataset import get_imagenet_dataset, set_seed, Logger

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description='test_acc')
    parser.add_argument('--data', metavar='DIR', required=True,
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        help='model architecture (default: resnet18)')
    parser.add_argument('--datasets', metavar='DATASETS', default='Imagenet', type=str,
                        help='The training datasets')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use')
    parser.add_argument('--randomseed', type=int, default=1, 
                        help='random seed for training and initialization')
    parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str,
                        help='directory to save the trained models')
    parser.add_argument('--log-dir', dest='log_dir', default='log', type=str,
                        help='directory to save the log')
    parser.add_argument('--log-name', dest='log_name', default='log.txt', type=str,
                        help='name of the log file')
    return parser


def load_model(filepath, device):
    model = ProcessedModel(resnet18(), NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    checkpoint = torch.load(filepath, map_location=device)

    # Remove 'module.' prefix if exists
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    
    # Load state dict with strict=False to skip non-matching layers
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def main(parser):
    args = parser.parse_args()
    set_seed(args.randomseed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    sys.stdout = Logger(os.path.join(args.log_dir, args.log_name))
    print('test_Acc')
    print('save dir:', args.save_dir)

    # Load model
    model = load_model(os.path.join(args.save_dir, '450.pt'), device)
    cudnn.benchmark = True

    # Define loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Prepare dataloader
    _, val_dataset = get_imagenet_dataset(args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Validate
    val_loss, val_accuracy = validate(val_loader, model, criterion, device, args)
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.2f}%')


def validate(val_loader, model, criterion, device, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

    return losses.avg, top1.avg


class AverageMeter(object):
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
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


if __name__ == '__main__':
    parser = get_parser()
    main(parser)



#################resnet101##############
# import argparse
# import time
# import os
# import sys

# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn

# sys.path.append('../models')
# sys.path.append('../dataset')
# from models import resnet101, NormalizeByChannelMeanStd, ProcessedModel
# from dataset import get_imagenet_dataset, set_seed, Logger

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# import numpy as np


# def get_parser():
#     parser = argparse.ArgumentParser(description='test_acc')
#     parser.add_argument('--data', metavar='DIR', required=True,
#                         help='path to dataset')
#     parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101',
#                         help='model architecture (default: resnet101)')
#     parser.add_argument('--datasets', metavar='DATASETS', default='Imagenet', type=str,
#                         help='The training datasets')
#     parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                         help='number of data loading workers (default: 4)')
#     parser.add_argument('-b', '--batch-size', default=256, type=int,
#                         metavar='N', help='mini-batch size (default: 256)')
#     parser.add_argument('--print-freq', '-p', default=100, type=int,
#                         metavar='N', help='print frequency (default: 100)')
#     parser.add_argument('--gpu', default=0, type=int,
#                         help='GPU id to use')
#     parser.add_argument('--randomseed', type=int, default=1, 
#                         help='random seed for training and initialization')
#     parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str,
#                         help='directory to save the trained models')
#     parser.add_argument('--log-dir', dest='log_dir', default='log', type=str,
#                         help='directory to save the log')
#     parser.add_argument('--log-name', dest='log_name', default='log.txt', type=str,
#                         help='name of the log file')
#     return parser


# def load_model(filepath, device):
#     model = ProcessedModel(resnet101(), NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
#     checkpoint = torch.load(filepath, map_location=device)
    
#     # Remove 'module.' prefix
#     state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    
#     model.load_state_dict(state_dict)
#     return model.to(device)


# def main(parser):
#     args = parser.parse_args()
#     set_seed(args.randomseed)

#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#     torch.cuda.set_device(args.gpu)

#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)
#     if not os.path.exists(args.log_dir):
#         os.makedirs(args.log_dir)
#     sys.stdout = Logger(os.path.join(args.log_dir, args.log_name))
#     print('test_Acc')
#     print('save dir:', args.save_dir)

#     # Load model
#     model = load_model(os.path.join(args.save_dir, '450.pt'), device)
#     cudnn.benchmark = True

#     # Define loss function
#     criterion = nn.CrossEntropyLoss().to(device)

#     # Prepare dataloader
#     _, val_dataset = get_imagenet_dataset(args)
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=args.batch_size, shuffle=False,
#         num_workers=args.workers, pin_memory=True)

#     # Validate
#     val_loss, val_accuracy = validate(val_loader, model, criterion, device, args)
#     print(f'Validation Loss: {val_loss:.4f}')
#     print(f'Validation Accuracy: {val_accuracy:.2f}%')


# def validate(val_loader, model, criterion, device, args):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     model.eval()
#     with torch.no_grad():
#         end = time.time()
#         for i, (input, target) in enumerate(val_loader):
#             input, target = input.to(device), target.to(device)

#             output = model(input)
#             loss = criterion(output, target)

#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             losses.update(loss.item(), input.size(0))
#             top1.update(acc1[0], input.size(0))
#             top5.update(acc5[0], input.size(0))

#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % args.print_freq == 0:
#                 print(f'Test: [{i}/{len(val_loader)}]\t'
#                       f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
#                       f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                       f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

#     return losses.avg, top1.avg


# class AverageMeter(object):
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def accuracy(output, target, topk=(1,)):
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


# if __name__ == '__main__':
#     parser = get_parser()
#     main(parser)
