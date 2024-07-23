from .resnet import * 

# torchvision.models中实现的模型，其数据需要经过一定的预处理，具体为
# 1. 各种数据增强，例如random crop, random flip等
# 2. 归一化，即每个像素除以255，从0-255的范围变换到0-1
# 3. 标准化，按通道减去均值，除以标准差
# 在一般的深度学习场景中，这部分往往由dataset的transform变化定义，但是，在攻击中，由于需要根据原始像素限定
# 攻击范围，例如每个像素的变化不能超过8，一个合适的方法是，dataset得到的数据只完成1,2，标准化由网络完成，这里给出一种实现方式

import torch
import torch.nn as nn
import torchvision.models

import os.path as osp
import numpy as np
import torch
import models.cifar
import models.imagenet

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    
class ProcessedModel(nn.Module):
    def __init__(self, net, data_normalize):
        super(ProcessedModel, self).__init__()
        self.net = net
        self.data_normalize = data_normalize

    def forward(self, x):
        x = self.data_normalize(x)
        return self.net(x)
    





### QueryNet ######################################################

def load_weight_from_pth_checkpoint(model, fname):
    print('Load weights from {}'.format(fname))
    raw_state_dict = torch.load(fname, map_location='cpu')['state_dict']
    state_dict = dict()
    for key, val in raw_state_dict.items():
        new_key = key.replace('module.', '')
        state_dict[new_key] = val

    model.load_state_dict(state_dict)


def make_model(dataset, arch, **kwargs):
    """
    Make model, and load pre-trained weights.
    :param dataset: cifar10 or imagenet
    :param arch: arch name, e.g., alexnet_bn
    :return: model (in cpu and training mode)
    """
    assert dataset in ['cifar10', 'imagenet']
    if dataset == 'cifar10':
        if arch == 'gdas':
            assert kwargs['train_data'] == 'full'
            model = models.cifar.gdas('data/cifar10-models/gdas/seed-6293/checkpoint-cifar10-model.pth')
            model.mean = [125.3 / 255, 123.0 / 255, 113.9 / 255]
            model.std = [63.0 / 255, 62.1 / 255, 66.7 / 255]
            model.input_space = 'RGB'
            model.input_range = [0, 1]
            model.input_size = [3, 32, 32]
        elif arch == 'pyramidnet272':
            assert kwargs['train_data'] == 'full'
            model = models.cifar.pyramidnet272(num_classes=10)
            load_weight_from_pth_checkpoint(model, 'data/cifar10-models/pyramidnet272/checkpoint.pth')
            model.mean = [0.49139968, 0.48215841, 0.44653091]
            model.std = [0.24703223, 0.24348513, 0.26158784]
            model.input_space = 'RGB'
            model.input_range = [0, 1]
            model.input_size = [3, 32, 32]
        else:
            # decide weight filename prefix, suffix
            if kwargs['train_data'] in ['cifar10.1']:
                # use cifar10.1 (2,000 images) to train models
                if kwargs['train_data'] == 'cifar10.1':
                    prefix = 'data/cifar10.1-models'
                else:
                    raise NotImplementedError('Unknown train data {}'.format(kwargs['train_data']))
                if kwargs['epoch'] == 'final':
                    suffix = 'final.pth'
                elif kwargs['epoch'] == 'best':
                    suffix = 'model_best.pth'
                else:
                    raise NotImplementedError('Unknown epoch {} for train data {}'.format(
                        kwargs['epoch'], kwargs['train_data']))
            elif kwargs['train_data'] == 'full':
                # use full training set to train models
                prefix = 'data/cifar10-models'
                if kwargs['epoch'] == 'final':
                    suffix = 'checkpoint.pth.tar'
                elif kwargs['epoch'] == 'best':
                    suffix = 'model_best.pth.tar'
                else:
                    raise NotImplementedError('Unknown epoch {} for train data {}'.format(
                        kwargs['epoch'], kwargs['train_data']))
            else:
                raise NotImplementedError('Unknown train data {}'.format(kwargs['train_data']))

            if arch == 'alexnet_bn':
                model = models.cifar.alexnet_bn(num_classes=10)
            elif arch == 'vgg11_bn':
                model = models.cifar.vgg11_bn(num_classes=10)
            elif arch == 'vgg13_bn':
                model = models.cifar.vgg13_bn(num_classes=10)
            elif arch == 'vgg16_bn':
                model = models.cifar.vgg16_bn(num_classes=10)
            elif arch == 'vgg19_bn':
                model = models.cifar.vgg19_bn(num_classes=10)
            elif arch == 'wrn-28-10-drop':
                model = models.cifar.wrn(depth=28, widen_factor=10, dropRate=0.3, num_classes=10)
            else:
                raise NotImplementedError('Unknown arch {}'.format(arch))

            # load weight
            load_weight_from_pth_checkpoint(model, osp.join(prefix, arch, suffix))

            # assign meta info
            model.mean = [0.4914, 0.4822, 0.4465]
            model.std = [0.2023, 0.1994, 0.2010]
            model.input_space = 'RGB'
            model.input_range = [0, 1]
            model.input_size = [3, 32, 32]

    elif dataset == 'imagenet':

        model = eval('models.imagenet.{}(num_classes=1000, pretrained=\'imagenet\')'.format(arch))

        if kwargs['train_data'] == 'full':
            # torchvision has load correct checkpoint automatically
            pass
        elif kwargs['train_data'] == 'imagenetv2-val':
            prefix = 'data/imagenetv2-v1val45000-models'
            if kwargs['epoch'] == 'final':
                suffix = 'checkpoint.pth.tar'
            elif kwargs['epoch'] == 'best':
                suffix = 'model_best.pth.tar'
            else:
                raise NotImplementedError('Unknown epoch {} for train data {}'.format(
                    kwargs['epoch'], kwargs['train_data']))

            # load weight
            load_weight_from_pth_checkpoint(model, osp.join(prefix, arch, suffix))
        else:
            raise NotImplementedError('Unknown train data {}'.format(kwargs['train_data']))
    else:
        raise NotImplementedError('Unknown dataset {}'.format(dataset))

    return model

