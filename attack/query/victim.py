from .utils import *
import torch
from torch import nn
import torchvision
import numpy as np
import math
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
sys.path.append("..")
from models import make_model
device = torch.device('cuda:0')


def get_margin_loss(y, logits, targeted=False, loss_type='margin_loss'):
    """ Implements the margin loss (difference between the correct and 2nd best class). """
    logits = torch.tensor(logits, dtype=torch.float32).clone()
    y = torch.tensor(y, dtype=torch.float32).clone()
    # logits = logits.clone().detach().float()
    # y = y.clone().detach().float()
    if loss_type == 'margin_loss':
        preds_correct_class = (logits * y).sum(dim=1, keepdim=True)
        diff = preds_correct_class - logits
        correct_class_indices = torch.argmax(y, dim=1)
        diff[torch.arange(len(y)), correct_class_indices] = float('inf')
        margin, _ = torch.min(diff, dim=1, keepdim=True)
        loss = -margin if targeted else margin
    elif loss_type == 'cross_entropy':
        probs = softmax(logits)
        correct_probs = (probs * y).sum(dim=1)
        loss = -torch.log(correct_probs)
        loss = -loss if not targeted else loss
    else:
        raise ValueError('Wrong loss type.')
    return loss.cpu().numpy().flatten()


class VictimCifar(nn.Module):
    """
    A StandardModel object wraps a cnn model.
    This model always accept standard image: in [0, 1] range, RGB order, un-normalized, NCHW format
    """
    def __init__(self, arch, device=device, batch_size=100, **kwargs):
        super(VictimCifar, self).__init__()
        # init cnn model
        self.cnn = make_model('cifar10', arch, **kwargs)
        self.arch = arch
        self.cnn.to(device)
        self.batch_size = batch_size
        self.device = device

        # init cnn model meta-information
        self.mean = np.reshape(self.cnn.mean, [1, 3, 1, 1])
        self.std = np.reshape(self.cnn.std, [1, 3, 1, 1])
        self.mean, self.std = self.mean.astype(np.float32), self.std.astype(np.float32)

    def __call__(self, x):
        x = np.floor(x * 255.0) / 255.0
        if hasattr(self, 'drop'): self.cnn.drop = self.drop

        # normalization
        x = (x - self.mean) / self.std
        x = x.astype(np.float32)
        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []
        with torch.no_grad():
            for i in range(n_batches):
                x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
                x_batch_torch = torch.as_tensor(x_batch, device=self.device)
                logits = self.cnn(x_batch_torch).cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits


class VictimMnist(nn.Module):
    def __init__(self, arch, device=device, batch_size=10000, **kwargs):
        super(VictimMnist, self).__init__()
        # init cnn model
        if arch == 'resnet_preact': self.cnn = torch.load('../../QueryNet/data/mnist-models/resnet_preact.pth')
        elif arch == 'wrn':         self.cnn = torch.load('data/mnist-models/wrn.pth')
        elif arch == 'densenet':    self.cnn = torch.load('../../QueryNet/data/mnist-models/densenet.pth')
        else: raise ValueError

        self.arch = arch
        self.cnn.to(device)
        self.batch_size = batch_size
        self.mean = np.array([0.1307])
        self.std = np.array([0.3081])
        self.device = device

    def __call__(self, x):
        x = np.floor(x * 255.0) / 255.0
        # normalization
        x = (x - self.mean) / self.std
        x = x.astype(np.float32)
        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []
        with torch.no_grad():
            for i in range(n_batches):
                x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
                x_batch_torch = torch.as_tensor(x_batch, device=self.device)
                logits = self.cnn(x_batch_torch).cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits


class VictimImagenet(nn.Module):
    def __init__(self, arch, device=device, batch_size=100, **kwargs):
        super(VictimImagenet, self).__init__()
        self.arch = arch
        self.cnn = getattr(torchvision.models, arch)(pretrained=True).to(device).eval()
        self.cnn.to(device)
        self.batch_size = batch_size
        self.device = device

    def __call__(self, _x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = np.floor(_x * 255.0) / 255.0
        for i in range(3): x[:, i, :, :] = (x[:, i, :, :]-mean[i])/std[i]
        if x.shape[0] <= self.batch_size: return self.cnn(torch.Tensor(x).to(self.device)).detach().cpu().numpy()
        batch_num = int(x.shape[0]/self.batch_size)
        if self.batch_size * batch_num != int(x.shape[0]): batch_num += 1
        logits = self.cnn(torch.Tensor(x[:self.batch_size]).to(self.device)).detach().cpu().numpy()
        for i in range(batch_num-1):
            new_logits = self.cnn(torch.Tensor(x[self.batch_size*(i+1):self.batch_size*(i+2)]).to(self.device)).detach().cpu().numpy()
            logits = np.concatenate((logits, new_logits), axis=0)
            del new_logits
        # return torch.from_numpy(logits).to("cuda:0")
        return logits