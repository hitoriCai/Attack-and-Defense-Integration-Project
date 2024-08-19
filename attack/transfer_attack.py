# AoA实现比较困难

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attack.base_attacker import Attack


# MI (Non-targeted, linf) ##################################################################
class MI(Attack):
    r"""
    MI Attack in the paper 'Boosting Adversarial Attacks with Momentum'
    Distance Measure : Linf
    Based on PGDATtack, non-targeted
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        num_iter (int): number of iterations in MI.  (Default: 4)
        alpha (float): step size. (alpha = eps/num_iter, Default: 2/255)
        steps (int): number of steps in PGD. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)
    Examples:
        >>> attack = attack.MI(net, eps=8/255, num_iter=4, steps=10, momentum=0.9)
    """
    def __init__(self, model, eps=8/255, num_iter=4, steps=10, momentum=0.9, random_start=True):
        super().__init__("MI", model)
        self.eps = eps
        self.model = model
        self.num_iter = num_iter
        self.alpha = self.eps / self.num_iter
        self.steps = steps
        self.momentum = momentum
        self.random_start = random_start
        self.num_classes = 1000
        self.device = "cuda:0"

    def mi(self, images, labels, grad):     # based on PGD attack
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)
            # Calculate loss
            cost = loss(outputs, labels)
            # Update adversarial images
            noise0 = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            noise = noise0.data
            noise = noise / noise.abs().mean(dim=[1,2,3], keepdim=True)
            noise = self.momentum * grad + noise  # 这里的noise0是上一轮的，这里的noise是本轮的grad
            adv_images = adv_images.detach() + self.alpha * noise.sign()  # 与普通的区别实际上是noise的计算里包含了momentum (**MI核心步骤**)
            adv_images.grad = None
            adv_images.noise = None
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        return adv_images, noise

    def __call__(self, images, labels):
        grad = torch.zeros([images.shape[0], images.shape[2], images.shape[3], 3]).to(self.device)
        grad = grad.permute(0, 3, 1, 2)
        for _ in range(0, self.num_iter):
            # print("begin MI, iter=", _+1, '/', self.num_iter, end='\r')
            images, grad = self.mi(images, labels, grad)
        return images    # adv_images



# DI (Non-targeted, linf) ##################################################################
class DI(Attack):
    r"""
    DI Attack in the paper 'Improving Transferability of Adversarial Examples with Input Diversity'
    Distance Measure : Linf
    Based on PGDATtack, non-targeted
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        num_iter (int): number of iterations in DI. (Default:4)
        alpha (float): step size. (alpha = eps/num_iter, Default: 2/255)
        steps (int): number of steps in PGD. (Default: 10)
        prob (float): transformation probability. (Default:0.5)
        image_width (int): the lower bound of rnd. (Default:299)
        image_resize (int): the upper bound of rnd. (Default:224)
        random_start (bool): using random initialization of delta. (Default: True)
    Examples:
        >>> attack = attack.DI(net, eps=8/255, num_iter=4, steps=10, prob=0.5)
    """
    def __init__(self, model, eps=8/255, num_iter=4, steps=10, prob=0.5, image_width=200, image_resize=224, random_start=True):
        super().__init__("MI", model)
        self.eps = eps
        self.model = model
        self.num_iter = num_iter
        self.alpha = self.eps / self.num_iter
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

    def di(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(self.input_diversity(adv_images))    # DI的关键步骤，**随机扰动**
            # Calculate loss
            cost = loss(outputs, labels)
            # Update adversarial images
            noise0 = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            noise = noise0.data
            noise = noise / noise.abs().mean(dim=[1,2,3], keepdim=True)
            # noise = self.momentum * grad + noise   # 这里先不融合 MI
            adv_images = adv_images.detach() + self.alpha * noise.sign()
            adv_images.grad = None
            adv_images.noise = None
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        return adv_images

    def __call__(self, images, labels):
        # grad = torch.zeros([images.shape[0], images.shape[2], images.shape[3], 3]).to(self.device)
        # grad = grad.permute(0, 3, 1, 2)
        for _ in range(0, self.num_iter):
            # print("begin DI, iter=", _+1, '/', self.num_iter, end='\r')
            images = self.di(images, labels)
        return images    # adv_images



# TI (Non-targeted, linf) ##################################################################
class TI(Attack):
    r"""
    TI Attack in the paper ''
    Distance Measure : Linf
    Based on PGDATtack, non-targeted
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        num_iter (int): number of iterations in MI. (Default: 4)
        alpha (float): step size. (alpha = eps/num_iter, Default: 2/255)
        steps (int): number of steps in PGD. (Default: 10)
        kernlen (int): size of the Gaussian kernel. (Default: 15)
        nsig (int): range of the Gaussian distribution. (Default: 3)
        random_start (bool): using random initialization of delta. (Default: True)
    Examples:
        >>> attack = attack.TI(net, eps=8/255, num_iter=4, steps=10, kernlen=15, nsig=3)
    """
    def __init__(self, model, eps=8/255, num_iter=4, steps=10, kernlen=15, nsig=3, random_start=True):
        super().__init__("MI", model)
        self.eps = eps
        self.model = model
        self.num_iter = num_iter
        self.alpha = self.eps / self.num_iter
        self.steps = steps
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

    def ti(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            # outputs = self.get_logits(self.input_diversity(adv_images))    # 这里先不融合DI
            outputs = self.get_logits(adv_images)
            # Calculate loss
            cost = loss(outputs, labels)
            
            # Update adversarial images
            noise0 = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            noise = noise0.data
            # 卷积 **TI关键步骤**
            noise = F.conv2d(noise, self.stack_kernel.to(self.device), padding=self.kernlen//2, groups=noise.shape[1])
            # depthwise_conv = nn.Conv2d(in_channels=noise.size(1), out_channels=noise.size(1), kernel_size = self.stack_kernel.size()[2:], groups=noise.size(1), bias=False)
            # depthwise_conv.weight.data = self.stack_kernel
            # noise = depthwise_conv(noise)
            noise = noise / noise.abs().mean(dim=[1,2,3], keepdim=True)
            # noise = self.momentum * grad + noise   # 这里先不融合 MI
            
            adv_images = adv_images.detach() + self.alpha * noise.sign()
            adv_images.grad = None
            adv_images.noise = None
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        return adv_images

    def __call__(self, images, labels):
        # grad = torch.zeros([images.shape[0], images.shape[2], images.shape[3], 3]).to(self.device)
        # grad = grad.permute(0, 3, 1, 2)
        for _ in range(0, self.num_iter):
            # print("begin TI, iter=", _+1, '/', self.num_iter, end='\r')
            images = self.ti(images, labels)
        return images    # adv_images






