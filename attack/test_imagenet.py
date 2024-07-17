import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.datasets import ImageFolder

import os
import argparse
import sys
sys.path.append("..")
import attack


def test(model, testloader, attack=None):
    model.eval()
    correct = 0
    total = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        # 对图片进行攻击
        if attack:
            images = attack(images, labels)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    # parser.add_argument('--num_epoch', default=200, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    # 定义设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # 指定ImageNet数据集的本地路径
    data_dir = '/opt/data/common/ILSVRC2012'
    # data_dir = '/home/datasets/ILSVRC2012'
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    testset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    net = models.resnet50(pretrained=True)  # resnet50
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/imagenet_res50_ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # 用FGSM攻击
    # attack = attack.FGSM(net, eps=8 / 255)

    # 用新写的PGD进行攻击：
    # attack = attack.PGD(net, eps=8 / 255, alpha=1 / 255, steps=10, random_start=True)  # Linf
    # attack = attack.PGDL2(net, eps=2.0, alpha=0.5, steps=10, random_start=True)  # L2

    # 用autoPGD-ce/dlr攻击：
    # ce:
    # attack = attack.APGD(net, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
    # attack = attack.APGD(net, norm='L2', eps=3.0, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
    # dlr:
    # attack = attack.APGD(net, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False)
    # attack = attack.APGD(net, norm='L2', eps=3.0, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False)

    # 用autoPGD-taegeted-ce/dlr攻击：
    # ce:
    # attack = attack.APGDT(net, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False, n_classes=1000)
    # attack = attack.APGDT(net, norm='L2', eps=3.0, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False, n_classes=1000)
    # dlr:
    # attack = attack.APGDT(net, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False, n_classes=1000)
    # attack = attack.APGDT(net, norm='L2', eps=3.0, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False, n_classes=1000)

    # 用SquareAttack攻击： 这里本来torchattack的示例是 n_queries=5000, 但跑着太慢了所以改成了100, 不知道该用什么参数
    attack = attack.Square(net, norm='Linf', eps=8 / 255, n_queries=5000, n_restarts=1, p_init=.8, seed=0, verbose=False,
                           targeted=False, loss='margin', resc_schedule=True)

    # 用QueryAttack攻击：（见querynet.py）




    # 测试原始模型在干净测试集上的准确度
    # clean_accuracy = test(net, testloader)
    # print(f'Accuracy on clean test images: {clean_accuracy:.2f}%')

    # 测试模型在攻击后的测试集上的准确度
    # for s in range(num_epoch)
    attack_accuracy = test(net, testloader, attack=attack)
    print(f'Accuracy on attacked test images: {attack_accuracy:.2f}%')



