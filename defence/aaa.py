import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F

def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def loss(y, logits, targeted=False, loss_type='margin_loss'):
    if loss_type == 'margin_loss':
        preds_correct_class = (logits * y).sum(1, keepdims=True)
        diff = preds_correct_class - logits
        diff[y] = np.inf
        margin = diff.min(1, keepdims=True)
        loss = margin * -1 if targeted else margin
    elif loss_type == 'cross_entropy':
        probs = softmax(logits)
        loss = -np.log(probs[y])
        loss = loss * -1 if not targeted else loss
    else:
        raise ValueError('Wrong loss.')
    return loss.flatten()

class AAALinear(nn.Module):
    def __init__(self, 
                 cnn,
                 arch,
                 device="cuda:0", 
                 attractor_interval=4, 
                 reverse_step=1, 
                 num_iter=100, 
                 calibration_loss_weight=5, 
                 optimizer_lr=0.1, 
                 **kwargs):
        super(AAALinear, self).__init__()
        self.dataset = 'imagenet'
        self.cnn = cnn
        
        self.loss = loss
        self.device = device

        self.attractor_interval = attractor_interval
        self.reverse_step = reverse_step
        self.dev = 0.5
        self.optimizer_lr = optimizer_lr
        self.calibration_loss_weight = calibration_loss_weight
        self.num_iter = num_iter
        self.arch_ori = arch
        self.arch = '%s_AAAlinear-Lr-%.1f-Ai-%d-Cw-%d' % (self.arch_ori, self.reverse_step, self.attractor_interval, self.calibration_loss_weight)
        self.temperature = 1 # 2.08333 #
        
        
    def forward(self, x):
        with torch.no_grad():
            x_curr = x
            if isinstance(x, np.ndarray): x_curr = torch.as_tensor(x_curr, device=self.device) 
            logits = self.cnn(x_curr)
        
        logits_ori = logits.detach()
        prob_ori = F.softmax(logits_ori / self.temperature, dim=1)
        prob_max_ori = prob_ori.max(1)[0] ###
        value, index_ori = torch.topk(logits_ori, k=2, dim=1)
        #"""
        mask_first = torch.zeros(logits.shape, device=self.device)
        mask_first[torch.arange(logits.shape[0]), index_ori[:, 0]] = 1
        mask_second = torch.zeros(logits.shape, device=self.device)
        mask_second[torch.arange(logits.shape[0]), index_ori[:, 1]] = 1
        #"""
        
        margin_ori = value[:, 0] - value[:, 1]
        attractor = ((margin_ori / self.attractor_interval + self.dev).round() - self.dev) * self.attractor_interval
        target = attractor - self.reverse_step * (margin_ori - attractor)
        diff_ori = (margin_ori - target)
        real_diff_ori = margin_ori - attractor
        #"""
        with torch.enable_grad():
            logits.requires_grad = True
            optimizer = torch.optim.Adam([logits], lr=self.optimizer_lr)
            i = 0 
            los_reverse_rate = 0
            prd_maintain_rate = 0
            for i in range(self.num_iter):
            #while i < self.num_iter or los_reverse_rate != 1 or prd_maintain_rate != 1:
                prob = F.softmax(logits, dim=1)
                #loss_calibration = (prob.max(1)[0] - prob_max_ori).abs().mean()
                loss_calibration = ((prob * mask_first).max(1)[0] - prob_max_ori).abs().mean() # better
                #loss_calibration = (prob - prob_ori).abs().mean()

                value, index = torch.topk(logits, k=2, dim=1) 
                margin = value[:, 0] - value[:, 1]
                #margin = (logits * mask_first).max(1)[0] - (logits * mask_second).max(1)[0]

                diff = (margin - target)
                real_diff = margin - attractor
                loss_defense = diff.abs().mean()
                
                loss = loss_defense + loss_calibration * self.calibration_loss_weight
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #i += 1
                los_reverse_rate = ((real_diff * real_diff_ori) < 0).float().mean()
                prd_maintain_rate = (index_ori[:, 0] == index[:, 0]).float().mean()
                #print('%d, %.2f, %.2f' % (i, los_reverse_rate * 100, prd_maintain_rate * 100), end='\r')
                #print('%d, %.4f, %.4f, %.4f' % (itre, loss_calibration, loss_defense, loss))
            return logits.detach().to(x.device)
            #print('main [los=%.2f, prd=%.2f], margin [ori=%.2f, tar=%.2f, fnl=%.2f], logits [ori=%.2f, fnl=%.2f], prob [tar=%.2f, fnl=%.2f]' % 
                #(los_reverse_rate * 100, prd_maintain_rate * 100, 
                #margin_ori[0], target[0], margin[0], logits_ori.max(1)[0][0], logits.max(1)[0][0], prob_max_ori[0], prob.max(1)[0][0]))
        #"""
        #logits_list.append(logits_ori.detach().cpu() / self.temperature)
