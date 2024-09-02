from defence import AAALinear
from models import ProcessedModel, NormalizeByChannelMeanStd
from torchvision.models import resnet101
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from autoattack.square import SquareAttack


net = resnet101(pretrained=True)
eps = 8
bs = 1024
val_dir = "/opt/data/common/ILSVRC2012/val/"
trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ])
dataset = datasets.ImageFolder(root=val_dir, transform=trans)
dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True, num_workers=10)

square = SquareAttack(net, p_init=.8, n_queries=100, eps=eps/255., norm='Linf',
                    n_restarts=1, seed=0, verbose=False,)


def get_acc_from_dataloader(model, dataloader, attack=None):
    correct = 0 
    total = 0
    model.eval()
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        if attack is not None:
            x = attack.perturb(x)
        outputs = model(x)
        _, predicted = outputs.max(1)
        correct += (predicted == y).sum().item()
        total += x.shape[0]
        if (batch_idx+1) %1 == 0:
            print(batch_idx, correct / total)
    return correct / total

clean_acc = get_acc_from_dataloader(net, dataloader, attack=None)
square_acc = get_acc_from_dataloader(net, dataloader, attack=square)

print(f"Clean acc:{clean_acc:.3f}\tSquare acc:{square_acc:.3f}")