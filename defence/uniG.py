import torch
import torch.optim
import torch.nn as nn
import numpy as np

### get forward feature by Hook
# -------------------- 第一步：定义接收feature的函数 ---------------------- #
# 这里定义了一个类，类有一个接收feature的函数hook_fun。定义类是为了方便提取多个中间层。
class HookTool: 
    def __init__(self):
        self.fea = None 
    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out

# ---------- 第二步：注册hook，告诉模型我将在哪些层提取feature,比如提取'fc'后的feature，即output -------- #
def get_feas_by_hook(model, extract_module=['fc']):
    fea_hooks = []
    for n, m in model.named_modules():
        # print('name:', n)
        # # if isinstance(m, extract_module):
        # print(extract_module)
        # if n == 'avg_pool':
        #     print('True')
        if n in extract_module:
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)
            
    return fea_hooks

# ### 用法示例
# fea_hooks = get_feas_by_hook(model, extract_module=['avgpool'])
# output = model(inputs)
# features = fea_hooks[0].fea.squeeze()


class P_layer(nn.Module):
    def __init__(self, shape):
        super(P_layer, self).__init__()
        self.shape = shape
        self.p = nn.Parameter(torch.rand(shape).cuda(), requires_grad = True)
        
    def forward(self, x):
        batch = x.size(0)
        mat, _ = self.p.split([batch, self.p.size(0)-batch], dim=0)
        x = x * mat
        return x

    def init_param(self):
        # self.p.data = self.save_p
        # torch.nn.init.constant_(self.p, 1)
        # torch.nn.init.uniform_(self.p)
        torch.nn.init.normal_(self.p, mean=1, std=0.5)
        return
    

class UniGModel(nn.Module):
    def __init__(self, model, module_name, head_name, epoch, lr, delta, ifcombine):
        super(UniGModel, self).__init__()
        self.model = model
        self.epoch = epoch
        self.lr = lr
        self.delta = delta
        self.ifcombine = ifcombine
        self.module_name = module_name
        self.head_name = head_name
        self.head_module = self.search_head_module()

        # # # train data
        # train_input = np.load('data/cifar10_pre_resnet18_GS_train_imgs_0.npy')
        # self.train_input = torch.from_numpy(train_input).cuda()
        # # # train label
        # train_label = np.load('data/cifar10_pre_resnet18_GS_train_lbls_0.npy')
        # self.train_label = torch.from_numpy(train_label).cuda()
        # self.addition = 5

    def set_gs_param(self, shape):
        self.gs = P_layer(shape)
        self.gs.init_param()
        return
    def get_target(self, output):
        _, pred = output.topk(1, 1, True, True)
        target = pred.squeeze().detach()
        return target
    
    def combine(self, x):
        index = int(100 * torch.rand(size=[1]))
        input = torch.cat((x, self.train_input[index:index + self.addition]), 0)
        train_label = self.train_label[index:index+self.addition]
        return input, train_label

    def gs_loss(self, grad):
        grad_simi = 0
        for j in range(grad.size(0) - 1):
            grad_simi += torch.norm((grad[j] - grad[j + 1]))
        return grad_simi

    def cal_shape(self, inputs):
        ### 用法示例
        fea_hooks = get_feas_by_hook(self.model.net, extract_module=[self.module_name])
        output = self.model(inputs)
        features = fea_hooks[0].fea.squeeze()
        feature_dim = features.shape[-1]
        return feature_dim
    
    def search_head_module(self):
        for name, module in self.model.net.named_modules():             
            if name == self.head_name:
                head_module = module
                break
        return head_module
    def get_pred(self, fea):
        fea = self.gs(fea)
        output = self.head_module(fea)
        return output
    def forward(self, x):
        feature_dim = self.cal_shape(x)
        batch = x.size(0)
        shape = (batch, feature_dim)
        if self.ifcombine:
            x, train_label = self.combine(x)
            shape = (batch+5, feature_dim)
        
        self.set_gs_param(shape)
        self.optimizer = torch.optim.SGD([self.gs.p],
                                         lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

        x.requires_grad_()
        with torch.enable_grad():
            with torch.no_grad():
                fea_hooks = get_feas_by_hook(self.model.net, extract_module=[self.module_name])
                output2 = self.model(x)
                fea = fea_hooks[0].fea.squeeze()
                target = self.get_target(output2)
                if self.ifcombine:
                    target = torch.cat((target[0:batch], train_label), 0)
            for i in range(self.epoch):
                #### forward
                # fea = self.model.get_feature(x)
                fea.requires_grad_()
                output = self.get_pred(fea)
                # output, fea, _ = self.model(x)
                ### optimize
                #----- gs loss
                loss_ce = self.criterion(output, target)
                self.optimizer.zero_grad()
                grad = torch.autograd.grad(loss_ce, fea, create_graph=True, retain_graph=True)[0]
                if i == 0:
                    min = grad.min().detach()
                    max = grad.max().detach()
                grad = (grad - min) / (max - min)
                loss_gs = self.gs_loss(grad)
                #----- all loss
                loss = 1 * loss_gs
                #----- step
                self.optimizer.zero_grad()
                # print(loss)
                loss.backward()
                self.optimizer.step()
                self.gs.p.data = torch.clamp(self.gs.p.data, 1 - self.delta, 1 + self.delta)
                # print('epoch:{:d}, gs_loss: {:.3f}, fea_loss: {:.3f}'.format(i, loss_gs.data, loss_fea.data))
        with torch.no_grad():
            output = self.get_pred(fea)     
        return output[0:batch]

       