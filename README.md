# Attack-and-Defense-Integration-Project
2024 Summer

## 1. 深度神经网络鲁棒性检测的开源工具1套

代码位置：``attack``

### 1.0 代码准备说明：
参考``test_imagenet.py``中的内容，测试模型需准备如下内容：

1. 测试数据的dataloader，其中使用的transform不能包含``transforms.Normalize``变换，
2. 测试网络，使用``torchvision.models``的预训练网络即可，但需要经过``ProcessedModel(net, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))``，将归一化变换加入网络输入侧
3. 攻击方法，从``attack``中导入特定的攻击方法进行测试

### 1.1 白盒攻击

#### 1.1.1 FGSM攻击
``from somewhere import some attack``
指定的参数有：
- 攻击强度 
- …

#### 1.1.2 PGD攻击与autoPGD
….

测试方法
``
``` python
from torchvision.models import resnet101

net = resnet101(pretrained=True)
net = ProcessedModel(net, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

```

### 1.2 查询攻击
#### 1.2.1 Square查询攻击

#### 1.2.2 QueryNet 查询迁移攻击

### 1.3 迁移攻击

#### 1.2.2 迁移攻击
包括AOA,TI,MI,DI






## 3. 超低维训练方法

代码位置: ``trainer``, ``normal_training.sh``, ``normal_training.py``, ``train_dldr.sh``, ``train_dldr.py``

### 3.1 提取SGD训练轨迹

参考``normal_training.sh``, 此训练脚本支持正常训练与对抗训练，可设置的参数有：

- ``datasets`` : 默认为ImageNet
- ``device`` : 单机多卡时使用的GPU编号，默认为0,1
- ``model`` : 训练模型，设置为resnet101，参数量4500万
- ``path`` : 为ImageNet数据集路径，其下应有train与val文件夹
- ``epochs`` : 训练epoch数量，默认为90
- ``DST`` : 模型保存路径, 将模型的训练轨迹保存在此路径下，默认每个epoch保存5个checkpoint
- ``eps`` :  对抗扰动大小，当指定为0时，为正常训练

  在以上设置下，执行``normal_training.sh``脚本，会训练resnet101 90个epoch，保存$90*5+1=451$(1为初始化时的网络状态)个checkpoint到DST路径。

### 3.2 使用SGD训练轨迹进行低维训练

参考``train_dldr.sh``， 此训练脚本支持正常低维训练与对抗低维训练，可设置的参数有：

- ``datasets`` : 默认为ImageNet
- ``device``:  单机多卡时使用的GPU编号，默认为0,1
- ``model`` : 训练模型，设置为resnet101，参数量4500万
- ``path`` : 为ImageNet数据集路径，其下应有train与val文件夹
- ``wd_psgd`` :  weight_decay参数, 默认为0.00001
- ``lr`` : 低维训练学习率，默认为0.3
- ``DST``:SGD训练轨迹路径，会从该路径下读取ckeckpoint文件构建训练子空间
- ``eps``: 对抗扰动大小，当指定为0时，为正常训练
- ``params_start`` : 训练轨迹起始点，设置为0
- ``params_end`` : 训练轨迹终止点，设置为301

在以上设置下，执行``train_dldr.sh``脚本，会提取``$DST``下 epoch 1到epoch60的训练轨迹，然后使用低维训练训练2个epoch，总训练epoch为62，加速比为$\frac{90-62}{90}=31.1\%$. 

> 模型保存在哪里？

