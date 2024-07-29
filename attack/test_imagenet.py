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


def test_query(model, x_test, y_test, logits_clean, attack):
    model.eval()
    correct = 0
    total = 0
    x_best_init, x_best = attack(model, x_test, y_test, logits_clean) # adv_images after 'iter' queries
    print("x_best_init:",np.linalg.norm(x_best_init.flatten(), ord=2))
    print("x_best:", np.linalg.norm(x_best.flatten(), ord=2))

    # 显存够的话，直接下面这行就可以了
    # outputs = model(torch.from_numpy(x_best).float().to(device))
    # 显存不够的话，分批计算outputs
    batch_size = 16
    num_batches = (len(x_best) + batch_size - 1) // batch_size  # 计算总批次数
    output_list = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(x_best))
        x_ = x_best[start_idx:end_idx]
        if x_.shape[0] == 0:
            continue
        x_ = torch.from_numpy(x_).float().to('cuda')
        with torch.no_grad():
            output_batch = model(x_)
        output_list.append(output_batch)
        del x_
        del output_batch
        torch.cuda.empty_cache()
    outputs = torch.cat(output_list, dim=0)

    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.cpu()
    y_test = torch.from_numpy(y_test)
    total += y_test.size(0)
    # correct += (predicted == y_test).sum().item()
    equal_pairs = predicted == y_test 
    correct = torch.sum(equal_pairs).item()  # 使用 .item() 来获取数值
    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters for ImageNet.')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--num_epoch', default=200, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

    '''
    parser.add_argument('--eps', default=8/255, type=float, help='epsilon')     # 8/255 or 3.0
    parser.add_argument('--norm', default='Linf', type=str, help='norm')    # 'Linf' or 'L2'
    parser.add_argument('--alpha', default=1/255, type=float, help='alpha')     # 1/255 or 0.5
    parser.add_argument('--steps', default=10, type=int, help='steps')
    parser.add_argument('--random_start', default=True, type=bool, help='random_start')
    parser.add_argument('--n_restarts', default=1, type=int, help='n_restarts')
    parser.add_argument('--seed', default=0, type=int, help='seed')     # queryattack---1
    parser.add_argument('--loss', default='ce', type=str, help='loss')    # 'ce' or 'dlr' (for APGD)
    parser.add_argument('--eot_iter', default=1, type=int, help='eot_iter')
    parser.add_argument('--rho', default=.75, type=float, help='rho')
    parser.add_argument('--verbose', default=False, type=bool, help='verbose')
    parser.add_argument('--n_classes', default=1000, type=int, help='n_classes')

    # QueryNet
    parser.add_argument('--query_net', default='resnext101_32x8d', type=str, help='[inception_v3, mnasnet1_0, resnext101_32x8d] for ImageNet')
    parser.add_argument('--num_x', type=int, default=10000, help='number of samples for evaluation.')
    parser.add_argument('--num_srg', type=int, default=0, help='number of surrogates.')
    parser.add_argument('--use_nas', action='store_true', help='use NAS to train the surrogate.')
    parser.add_argument('--use_square_plus', action='store_true', help='use Square+.')
    parser.add_argument('--p_init', type=float, default=0.05, help='hyperparameter of Square, the probability of changing a coordinate.')
    parser.add_argument('--run_times', type=int, default=1, help='repeated running time.')
    parser.add_argument('--l2_attack', action='store_true', help='perform l2 attack')
    parser.add_argument('--num_iter', type=int, default=10000, help='maximum query times.')
    parser.add_argument('--gpu', type=str, default='1', help='GPU number(s).')
    '''
    args = parser.parse_args()


    # 定义设备
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # 指定ImageNet数据集的本地路径
    data_dir = '/opt/data/common/ILSVRC2012'
    # data_dir = '/home/datasets/ILSVRC2012'
    # Data/opt/data/common/ILSVRC2012
    print('==> Preparing data..')
    '''
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    '''

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    testset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    # net = models.resnet50(pretrained=True)  # resnet50
    net = models.resnext101_32x8d(pretrained=True)  # resnet101
    net = net.to(device)
    if device == 'cuda:0':
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
    # attack = attack.Square(net, norm='Linf', eps=8 / 255, n_queries=5000, n_restarts=1, p_init=.8, seed=0, verbose=False,
    #                        targeted=False, loss='margin', resc_schedule=True)

    # 用QueryAttack攻击：
    attack = attack.QueryAttack(net, eps=8/255, num_iter=5000)
    x_test, y_test, logits_clean, querynet_model = attack.get_xylogits(model_names='resnext101_32x8d')
    # only for QueryAttack:
    # test_query(net, x_test, y_test, logits_clean, attack=attack)
    # attack(net, x_test, y_test, logits_clean)
    # querynet_model = querynet_model.to(device)
    attack_accuracy = test_query(net, x_test, y_test, logits_clean, attack=attack)
    print(f'After 5000 iters, accuracy on attacked test images: {attack_accuracy:.2f}%')


    # 测试原始模型在干净测试集上的准确度
    # clean_accuracy = test(net, testloader)
    # print(f'Accuracy on clean test images: {clean_accuracy:.2f}%')

    # 测试模型在攻击后的测试集上的准确度
    # for s in range(num_epoch)
    # attack_accuracy = test(net, testloader, attack=attack)
    # print(f'Accuracy on attacked test images: {attack_accuracy:.2f}%')



