import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.datasets import ImageFolder
import numpy as np

import os
import argparse
import sys
sys.path.append("..")
import attack
from models import *


def test(model, testloader, attack=None, trans=None):
    correct = 0
    if hasattr(attack, 'get_xylogits'): # queryattack
        model.eval()
        x_test, y_test, logits_clean, net = attack.get_xylogits(model, testloader)
        x_adv = attack(net, x_test, y_test, logits_clean) # adv_images after 'iter' queries
        accuracy = test_query(net, x_adv, y_test, trans)
    else:
        total = 0
        if trans:
            model = trans
        model.to(device)
        model.eval()
        cnt=0
        for images, labels in testloader:
            cnt+=1
            cd = images.to(device)
            print("iter=", cnt, '/', 782, end='\r')
            images, labels = images.to(device), labels.to(device)
            if attack:
                images = attack(images, labels)

            # 检测扰动是否在范围内
            perturbation = images - cd
            l_inf_norm = torch.max(perturbation.abs())
            # print(l_inf_norm)
            if l_inf_norm > 8/255+0.0001:
                print(l_inf_norm, "扰动在范围外")
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # if cnt==50:
            #     break
        print()
        accuracy = 100 * correct / total
    return accuracy


def test_query(model, x_best, y_test, trans=None):
    batch_size = 16
    if trans:
        model = trans
    model.eval()
    correct = 0
    num_batches = (len(x_best) + batch_size - 1) // batch_size
    output_list = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(x_best))
        x_ = x_best[start_idx:end_idx]
        if x_.shape[0] == 0:
            continue
        with torch.no_grad():
            output_batch = model(x_)
        output_list.append(output_batch)
        del x_
        del output_batch
        torch.cuda.empty_cache()
    outputs = np.concatenate(output_list, axis=0)
    _, predicted = torch.max(torch.tensor(outputs), 1)
    predicted = predicted.cpu()
    y = torch.argmax(torch.tensor(y_test).to(torch.int32), dim=1)
    total = y.size(0)
    correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters for ImageNet.')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    data_dir = '/opt/data/common/ILSVRC2012'  # Specify the local path for the ImageNet dataset.
    # data_dir = '/home/datasets/ILSVRC2012'
    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    testset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
    subdataset, _ = torch.utils.data.random_split(testset, [1000, len(testset)-1000], generator=torch.Generator().manual_seed(0))
    testloader = torch.utils.data.DataLoader(subdataset, batch_size=32, shuffle=True, num_workers=10)

    # Model
    print('==> Building model..')
    # net = models.resnet18(pretrained=True)
    net = models.resnet50(pretrained=True)
    # net = models.resnet101(pretrained=True)
    trans_net = models.resnet101(pretrained=True)   # for transferability
    net = ProcessedModel(net, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).to(device)
    trans_net = ProcessedModel(trans_net, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).to(device)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/imagenet_res50_ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # FGSM:
    # attack = attack.FGSM(net, eps=8 / 255)

    # PGD:
    attack_pgd = attack.PGD(net, eps=8 / 255, alpha=1 / 255, steps=10, random_start=True)  # Linf

    # autoPGD-ce/dlr:
    # attack = attack.APGD(net, eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
    # attack = attack.APGD(net, eps=8/255, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False)

    # SquareAttack:
    # attack = attack.Square(net, eps=8 / 255, n_queries=100, n_restarts=1, p_init=.8, seed=0, verbose=False, loss='margin', resc_schedule=True)

    # QueryAttack:
    # attack = attack.QueryAttack(net, eps=8/255, num_iter=5000, num_x=10000)

    # 用 MI+PGD 攻击：
    # attack = attack.MI(net, eps=8/255, steps=10, momentum=0.9)
    
    # 用 DI+PGD 攻击：
    # attack = attack.DI(net, eps=8/255, steps=10, prob=0.5)

    # 用 TI+PGD 攻击：
    # attack = attack.TI(net, eps=8/255, steps=10, kernlen=5, nsig=5)

    # 用 AoA 攻击：
    attack_aoa = attack.AoA(net, eps=8/255, alpha=1.6/255, steps=10, lamb=10, layer_name="layer4")

    # 测试原始模型在干净测试集上的准确度
    # clean_accuracy = test(net, testloader)
    # print(f'Accuracy on clean test images: {clean_accuracy:.2f}%')

    # 测试模型在攻击后的测试集上的准确度
    # attack_accuracy = test(net, testloader, attack=attack)
    attack_accuracy = test(net, testloader, attack=attack_aoa, trans=trans_net)  # for tranferability
    print(f'Accuracy on attacked test images: {attack_accuracy:.2f}%')



