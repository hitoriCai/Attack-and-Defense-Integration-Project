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
    
data_normalizer = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

resnet18 = ProcessedModel(torchvision.models.resnet50(), data_normalize=data_normalizer)