from defence import AAALinear, UniGModel
from models import ProcessedModel, NormalizeByChannelMeanStd
from torchvision.models import resnet101, resnet152, resnet18
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from autoattack.square import SquareAttack
from tqdm import tqdm
import torch
import argparse

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument("--method", type=str)
args = parser.parse_args()
    
model = resnet152(pretrained=True)
model = ProcessedModel(model, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).cuda()
model.eval()


### For AAA defence

if args.method == "aaa":
    model = AAALinear(
        model,
        "resnet101",
        device="cuda:0", 
        attractor_interval=4, 
        reverse_step=1, 
        num_iter=100, 
        calibration_loss_weight=5, 
        optimizer_lr=0.1, 
        )

### For UniG
elif args.method == "unig":
    model = UniGModel(
            model,
            module_name='avgpool',
            head_name='fc',
            epoch=5, 
            lr=0.01, 
            delta=0.2, 
            ifcombine=False
        )

elif args.method == "none":
    pass

else:
    raise NotImplementedError

eps = 8
bs = 256
val_dir = "/opt/data/common/ILSVRC2012/val/"
trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ])
dataset = datasets.ImageFolder(root=val_dir, transform=trans)
dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True, num_workers=10)

square = SquareAttack(model, p_init=.8, n_queries=100, eps=eps/255., norm='Linf',
                    n_restarts=1, seed=0, verbose=False,)

@torch.no_grad()
def get_acc_from_dataloader(model, dataloader, attack=None):
    correct = 0 
    total = 0
    model.eval()
    i = 0
    for (x, y) in (dataloader):
        x, y = x.cuda(), y.cuda()
        if attack is not None:
            x = attack.perturb(x)
        outputs = model(x)
        _, predicted = outputs.max(1)
        correct += (predicted == y).sum().item()
        total += x.shape[0]
        i += 1
        # tqdm.write(f'Processing item {i}')
        print(f'Current: {correct / total:.4f}') 
    return correct / total

clean_acc = get_acc_from_dataloader(model, dataloader, attack=None)
square_acc = get_acc_from_dataloader(model, dataloader, attack=square)

print(f"Clean acc:{clean_acc:.3f}\tSquare acc:{square_acc:.3f}")