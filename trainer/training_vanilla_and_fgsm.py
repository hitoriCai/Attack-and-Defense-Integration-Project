import argparse
import os
import random
import shutil
import time
import warnings
import os
import sys
import numpy as np
import pickle

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.append('../models')
sys.path.append('../dataset')
from models import resnet18,resnet101,resnet152, NormalizeByChannelMeanStd, ProcessedModel
from dataset import get_dataloader_from_args
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as t_models


model_names = sorted(name for name in t_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(t_models.__dict__[name]))

def get_parser_normal():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--save_dir', metavar='save_dir', default="save_model", 
                        help="where to save the checkpoints")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--eps', default=4, type=int, metavar='N',
                        help='epsilon for adversarial training')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=1000, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    return parser





def get_model_param_vec(model):
    # Return the model parameters as a vector

    vec = []
    for name,param in model.named_parameters():
        vec.append(param.detach().cpu().reshape(-1).numpy())
    return np.concatenate(vec, 0)

def start_ddp(parser):
    
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    train_loss, train_acc, test_loss, test_acc, arr_time = [], [], [], [],[]
    best_acc1 = 0
    sample_idx = 0
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()
    '''-------------------------------------------------'''
    if args.arch == "resnet18":    
        base_model = resnet18()
    elif args.arch == "resnet152":
        base_model= resnet152()
    elif args.arch == "resnet101":
        base_model= resnet101()
    else:
        _ = 1/0
    data_normalizer = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = ProcessedModel(base_model, data_normalize=data_normalizer)   
    print(f"Create Model Finished, gpu id: {args.gpu}")
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            print("Model DDP Finished")
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print("Model DDP Finished")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    local_rank = args.rank % ngpus_per_node

    # Data loading code
    train_sampler, train_loader, val_loader = get_dataloader_from_args(args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and local_rank == 0):
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"{sample_idx}.pt"))
    
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        if local_rank == 0: print(f"Training at epoch {epoch}")

        # train for one epoch
        train_loss, train_acc, arr_time,sample_idx = train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node, train_loss, train_acc, test_loss, test_acc, arr_time, sample_idx, args.eps/255.)

        # evaluate on validation set
        acc1, test_acc, test_loss = validate(val_loader, model, criterion, args,train_loss, train_acc, test_loss, test_acc, arr_time, local_rank=local_rank)
        if args.eps>0:
            validate_robustness(val_loader, model, criterion, args,train_loss, train_acc, test_loss, test_acc, arr_time, eps=8/255., iters=4, alpha=8/255/2., local_rank=local_rank)
            
            validate_robustness(val_loader, model, criterion, args,train_loss, train_acc, test_loss, test_acc, arr_time, eps=args.eps/255., iters=4, alpha=args.eps/255/2., local_rank=local_rank)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_dir)

    print ('train loss: ', train_loss)
    print ('train acc: ', train_acc)
    print ('test loss: ', test_loss)
    print ('test acc: ', test_acc)

    print ('time: ', arr_time)


def train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node, train_loss, train_acc, test_loss, test_acc, arr_time, sample_idx, eps):
    
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    epoch_start = end
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
        
        if eps>0:
            delta_x = torch.empty_like(images).uniform_(-eps,eps).requires_grad_(True)
            model.eval()
            output = model(images+delta_x)
            loss = criterion(output, target)
            loss.backward()
            delta_x.data = torch.clamp(delta_x.data + delta_x.grad.sign()*eps, -eps, eps)
            images.data = torch.clamp(images.data + delta_x.data, 0., 1.)
            delta_x.grad = None
            model.train()
        
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed 
                and args.rank % ngpus_per_node == 0):
            
            if (i+1) % args.print_freq == 0:
                progress.display(i)

            if i > 0 and i % 1000 == 0 and i < 5000:
                sample_idx += 1
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"{sample_idx}.pt"))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed 
            and args.rank % ngpus_per_node == 0):
        sample_idx += 1
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"{sample_idx}.pt"))
    
    arr_time.append(time.time() - epoch_start)
    train_loss.append(losses.avg)
    train_acc.append(top1.avg)
    return train_loss, train_acc, arr_time, sample_idx


def validate(val_loader, model, criterion, args, train_loss, train_acc, test_loss, test_acc, arr_time, local_rank):
     
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        if local_rank==0: print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    test_acc.append(top1.avg)
    test_loss.append(losses.avg)
    return top1.avg, test_acc, test_loss


def validate_robustness(val_loader, model, criterion, args, train_loss, train_acc, test_loss, test_acc, arr_time, eps, iters, alpha, local_rank):
     
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
        
        delta_x = torch.empty_like(images).uniform_(-eps,eps).requires_grad_(True)
        for i in range(iters):
            output = model(images+delta_x)
            loss = criterion(output, target)
            loss.backward()
            delta_x.data = torch.clamp(delta_x.data + delta_x.grad.sign()*alpha, -eps, eps)
            delta_x.grad = None
        images.data = torch.clamp(images.data + delta_x.data, 0., 1.)
        delta_x.grad = None
        delta_x.requires_grad=False
        with torch.no_grad():
        # compute output
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    if local_rank==0: print("Robust Acc@eps={}/255".format(int(eps*255)) + ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
    test_acc.append(top1.avg)
    test_loss.append(losses.avg)
    return top1.avg, test_acc, test_loss


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pt'):
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    # lr = 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


if __name__ == '__main__':
    parser = get_parser_normal()
    start_ddp(parser)
