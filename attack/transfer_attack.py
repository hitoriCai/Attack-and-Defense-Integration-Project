import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from attack.base_attacker import Attack
from attack.gradcam.gradcam import *
from attack.gradcam.gc_utils import *


class MI():
    r"""
    MI Attack in the paper 'Boosting Adversarial Attacks with Momentum'
    Distance Measure : Linf
    Based on PGDAttack, non-targeted
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (alpha = eps*2/steps)
        steps (int): number of steps in PGD. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)
    Examples:
        >>> attack = attack.MI(net, eps=8/255 steps=10, momentum=0.9)
    """
    def __init__(self, model, eps=8/255, steps=10, momentum=0.9, random_start=True):
        self.eps = eps
        self.model = model
        self.alpha = self.eps*2 / steps
        self.steps = steps
        self.momentum = momentum
        self.random_start = random_start
        self.num_classes = 1000
        self.device = "cuda:0"

    def __call__(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()

        if self.random_start:
            # Starting at a uniformly random point
            delta_x = torch.empty_like(images).uniform_(-self.eps, self.eps)
            delta_x.data = torch.clamp(images.data + delta_x.data, min=0., max=1.) - images.data
        else:
            delta_x = torch.zeros_like(images)
        g_t = torch.zeros_like(delta_x)

        delta_x.requires_grad = True

        for _ in range(self.steps):
            outputs = self.model(images + delta_x)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, delta_x, retain_graph=False, create_graph=False)[0]

            normalized_grad = grad / grad.abs().mean(dim=[1,2,3], keepdim=True)
            g_t = self.momentum * g_t + normalized_grad

            delta_x.data = delta_x.data + g_t.sign() * self.alpha
            delta_x.data = torch.clamp(delta_x.data, -self.eps, self.eps)
            delta_x.data = torch.clamp(images.data + delta_x.data, min=0., max=1.) - images.data
            delta_x.grad = None

        return images + delta_x



# DI (Non-targeted, linf) ##################################################################
class DI():
    r"""
    DI Attack in the paper 'Improving Transferability of Adversarial Examples with Input Diversity'
    Distance Measure : Linf
    Based on PGDATtack, non-targeted
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        num_iter (int): number of iterations in DI. (Default:4)
        alpha (float): step size. (alpha = eps*2/steps, Default: 2/255)
        steps (int): number of steps in PGD. (Default: 10)
        prob (float): transformation probability. (Default:0.5)
        image_width (int): the lower bound of rnd. (Default:200)
        image_resize (int): the upper bound of rnd. (Default:224)
        random_start (bool): using random initialization of delta. (Default: True)
    Examples:
        >>> attack = attack.DI(net, eps=8/255, num_iter=4, steps=10, prob=0.5)
    """
    def __init__(self, model, eps=8/255, steps=10, prob=0.5, image_width=200, image_resize=224, random_start=True):
        self.eps = eps
        self.model = model
        self.alpha = self.eps*2 / steps
        self.steps = steps
        self.prob = prob
        self.image_width = image_width
        self.image_resize = image_resize
        self.random_start = random_start
        self.num_classes = 1000
        self.device = "cuda:0"

    def input_diversity(self, x):
        batch_size, channels, height, width = x.shape
        rnd = torch.randint(low=self.image_width, high=self.image_resize, size=(1,)).item()
        rescaled = F.interpolate(x, size=(rnd, rnd), mode='nearest')
        
        h_rem = self.image_resize - rnd
        w_rem = self.image_resize - rnd
        
        pad_top = torch.randint(0, h_rem, (1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem, (1,)).item()
        pad_right = w_rem - pad_left
        
        padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        padded = padded.view(batch_size, channels, self.image_resize, self.image_resize)
        
        # Apply with probability prob
        apply_mask = torch.rand(1) < self.prob
        output = torch.where(apply_mask.to(self.device), padded.to(self.device), x)
        return output

    def __call__(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()

        if self.random_start:
            # Starting at a uniformly random point
            delta_x = torch.empty_like(images).uniform_(-self.eps, self.eps)
            delta_x.data = torch.clamp(images.data + delta_x.data, min=0., max=1.) - images.data
        else:
            delta_x = torch.zeros_like(images)

        delta_x.requires_grad = True

        for _ in range(self.steps):
            outputs = self.model(self.input_diversity(images + delta_x))   ### DI
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, delta_x, retain_graph=False, create_graph=False)[0]

            normalized_grad = grad / grad.abs().mean(dim=[1,2,3], keepdim=True)

            delta_x.data = delta_x.data + normalized_grad.sign() * self.alpha
            delta_x.data = torch.clamp(delta_x.data, -self.eps, self.eps)
            delta_x.data = torch.clamp(images.data + delta_x.data, min=0., max=1.) - images.data
            delta_x.grad = None

        return images + delta_x


# TI (Non-targeted, linf) ##################################################################
class TI():
    r"""
    TI Attack in the paper 'Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks'
    Distance Measure : Linf
    Based on PGDATtack, non-targeted
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (alpha = eps*2/steps)
        steps (int): number of steps in PGD. (Default: 10)
        kernlen (int): size of the Gaussian kernel. (Default: 5)
        nsig (int): range of the Gaussian distribution. (Default: 5)
        random_start (bool): using random initialization of delta. (Default: True)
    Examples:
        >>> attack = attack.TI(net, eps=8/255, steps=10, kernlen=5, nsig=5)
    """
    def __init__(self, model, eps=8/255, steps=10, kernlen=5, nsig=5, random_start=True):
        self.eps = eps
        self.model = model
        self.steps = steps
        self.alpha = self.eps*2 / self.steps
        self.kernlen = kernlen
        self.random_start = random_start
        self.num_classes = 1000
        self.device = "cuda:0"

        self.kernel = self.gkern(kernlen=kernlen, nsig=nsig).astype(np.float32)
        stack_kernel = np.stack([self.kernel, self.kernel, self.kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        self.stack_kernel = torch.from_numpy(stack_kernel)

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        import scipy.stats as st
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def __call__(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()

        if self.random_start:
            # Starting at a uniformly random point
            delta_x = torch.empty_like(images).uniform_(-self.eps, self.eps)
            delta_x.data = torch.clamp(images.data + delta_x.data, min=0., max=1.) - images.data
        else:
            delta_x = torch.zeros_like(images)

        delta_x.requires_grad = True

        for _ in range(self.steps):
            outputs = self.model(images + delta_x)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, delta_x, retain_graph=False, create_graph=False)[0]

            # grad = F.conv2d(grad, self.stack_kernel.to(self.device), padding=self.kernlen//2, groups=grad.shape[1])  ### TI
            # grad = F.conv2d(grad, self.stack_kernel.to(self.device), stride=1, padding=0, groups=grad.size(1))

            # 计算填充
            kernel_size = self.stack_kernel.size(2)  # 假设卷积核的高度和宽度相同
            padding = (kernel_size - 1) // 2
            grad = F.pad(grad, (padding, padding, padding, padding), mode='constant', value=0)
            grad = F.conv2d(grad, self.stack_kernel.to(self.device), stride=1, groups=grad.size(1))

            normalized_grad = grad / grad.abs().mean(dim=[1,2,3], keepdim=True)

            delta_x.data = delta_x.data + normalized_grad.sign() * self.alpha
            delta_x.data = torch.clamp(delta_x.data, -self.eps, self.eps)
            delta_x.data = torch.clamp(images.data + delta_x.data, min=0., max=1.) - images.data
            delta_x.grad = None

        return images + delta_x


# AoA ##################################################################
class AoA():
    r"""
    AoA (Attack on Attention) in the pape 'Universal Adversarial Attack on Attention and the Resulting Dataset DAmageNetr'
    Using grad-cam to calculate the attention map h(x,y)
    Arguments:
        model (nn.Module): model to attack.
        model_dict (string):  a dictionary that contains 'type', 'arch', layer_name', 'input_size'(optional) as keys.
        type (string): 'vgg', 'resnet', 'densenet', 'alexnet', 'squeezenet'. (Default: 'resnet')
        eps (float): maximum perturbation. (Default: 8/255)
        lamb (int): a trade-off between the attack on attention and cross entropy. (Default: 1000)
        yita (int): the bound of Root Mean Squared Error. (Default: None)
        alpha (float): step size. (alpha = eps/num_iter, Default: 2/255)
        num_iter (int): number of iterations. (Default: 4)
    Examples:
        >>> attack = attack.AoA(net, eps=8/255, alpha=2, num_iter=4, lamb=1000)
    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=4, lamb=1000, layer_name="layer4"):
        self.eps = eps
        self.model = model
        self.lamb = max(lamb, 1e-4)
        self.alpha = alpha
        self.steps = steps
        self.num_classes = 1000
        self.device = "cuda:0"
        self.layer_name=layer_name

    def __call__(self, x, y):
        """
        support batch calculation
        x shape: [B,C,H,W]
        y shape: [B]
        layer_name: 是最后一个卷积层输出的特征层. (Default: 'layer4', for resnet50)
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        model_dict = dict(type='resnet', arch=self.model, layer_name=self.layer_name, input_size=(1, 3, 224, 224))
        gradcam = GradCAM(model_dict)

        with torch.no_grad():
            logits = self.model(x).detach()
        values, indices = torch.topk(logits, 2, dim=1)
        y_ori = indices[:, 0]  
        y_sec = indices[:, 1] 

        def AoAloss_batch(x:torch.tensor, y_ori:torch.tensor, y_sec:torch.tensor, target:torch.tensor):
            '''
            x shape: [1, 3, W, H]
            '''
            assert x.requires_grad , "X should need gradient"
            h_ori, logit_ori = gradcam(x, y_ori)
            h_sec, logit_sec = gradcam(x, y_sec)
            
            L_log = torch.log(torch.norm(h_ori, p=1, dim=(1,2,3)) + 1e-8) - \
                torch.log(torch.norm(h_sec, p=1, dim=(1,2,3))+ 1e-8)
            L_log = L_log.mean()
            
            # logit_ori and logit_sec shoule be the same
            L_ce = nn.CrossEntropyLoss()(logit_ori, target) * 0.5 + nn.CrossEntropyLoss()(logit_sec, target) * 0.5
            if torch.isnan(L_log).any():
                print("Loss has Nan, just using CE loss")
                L_AoA = -self.lamb * L_ce
            else:
                L_AoA = L_log - self.lamb * L_ce
            grad = torch.autograd.grad(L_AoA, x)[0].detach()
            del gradcam.activations["value"]
            gradcam.activations = dict()
            self.model.zero_grad()
            torch.cuda.empty_cache() 
            return -grad
            
        delta_x = torch.zeros_like(x).to(self.device).uniform_(-self.eps, self.eps)
        delta_x.requires_grad = True
        
        for _ in range(self.steps):
            grad = AoAloss_batch((x+delta_x), y_ori, y_sec, y)
            delta_x.data = delta_x.data + self.alpha * grad.sign().detach()
            delta_x.data = torch.clamp(delta_x.data, -self.eps, self.eps)
            delta_x.data = torch.clamp(delta_x.data + x.data, 0, 1) - x.data
            delta_x.grad = None
        return (x+delta_x).detach()
            



 