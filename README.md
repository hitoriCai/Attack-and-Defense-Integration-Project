# Attack-and-Defense-Integration-Project

2024 Summer

## 1. 深度神经网络鲁棒性检测的开源工具1套

代码位置：``attack``

### 1.0 代码准备说明：

参考``test_imagenet.py``中的内容，测试模型需准备如下内容：

1. 测试数据的dataloader，其中使用的transform不能包含``transforms.Normalize``变换，
2. 测试网络，使用``torchvision.models``的预训练网络即可，但需要经过``ProcessedModel(net, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))``，将归一化变换加入网络输入侧
3. 攻击方法，从``attack``中导入特定的攻击方法进行测试, 需要 `import torchvision.models as models` 

4. 本攻击代码用于 ImageNet, 因此要修改正确的 ImageNet 路径 `data_dir = '/your_dir/ILSVRC2012'`

5. 在攻击之前, 原图片在 `resnet50` 下的分类正确率为 Accuracy on clean test images: **76.15%**

```python
clean_accuracy = test(net, testloader)
```

6. 关于 `testloader`, 测试全数据集时 `shuffle=False`

```python
testset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=10)
```

7. 采用评估模式 `net.eval()`, `trans_net.eval()`



### 1.1 白盒攻击

参考 `attack/white_box.py`

白盒攻击测试网络均为:

```python
net = models.resnet50(pretrained=True)
net = ProcessedModel(net, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
```

#### 1.1.1 FGSM攻击

指定的参数有：

- `net`: 攻击网络, 默认为 `resnet50`

- `eps`: 攻击强度 , 默认为 8/255

测试方法:

```python
attack_fgsm = attack.FGSM(net, eps=8 / 255)
```

攻击后图片分类正确率: Accuracy on attacked test images: **7.90%**



#### 1.1.2 PGD攻击与autoPGD

**PGD攻击:**

指定的参数有：

- `net`: 攻击网络, 默认为 `resnet50`
- `eps`: 攻击强度 , 默认为 8/255

- `alpha`: 步长(step size), 默认为 2/255

测试方法:
``` python
attack_pgd = attack.PGD(net, eps=8 / 255, alpha=1 / 255, steps=10, random_start=True)  # Linf
```

攻击后图片分类正确率: Accuracy on attacked test images: **0.02%**



**autoPGD攻击(APGD):**

指定的参数有：

- `net`: 攻击网络, 默认为 `resnet50`
- `eps`: 攻击强度 , 默认为 8/255

- `alpha`: 步长(step size), 默认为 2/255
- `n_restarts`: 随机重启的次数, 默认为1
- `seed`: 起点的随机数种子, 默认为0
- `loss`: 损失函数的选择, 为`['ce', 'dlr'] `, 默认为 'ce' 
- `eot_iter`: EOT的迭代次数, 默认为1
- `rho`: 步长更新参数, 默认为0.75 
- `verbose`: 是否打印进度, 默认为False

APGD-ce 测试方法:

```python
attack_apgd = attack.APGD(net, eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
```

攻击后图片分类正确率: Accuracy on attacked test images: **0.09%**

APGD-dlr 测试方法:

```python
attack_apgd = attack.APGD(net, eps=8/255, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False)
```

攻击后图片分类正确率: Accuracy on attacked test images: **0.00%**





### 1.2 查询攻击

参考 `attack/query_attack.py`

#### 1.2.1 Square查询攻击

参考 `attack/query_attack.py` 里的 `Square` 类

指定的参数有：

- `net`: 攻击网络, 默认为 `resnet50`
- `eps`: 攻击强度 , 默认为 8/255

- `n_queries`: 最大查询次数, 默认为 100
- `n_restarts`: 随机重启的次数, 默认为 1 
- `p_init`: 控制正方形大小的参数, 默认为 0.8
- `loss`: 损失函数的选择, 为`['ce', 'margin'] `, 默认为 'margin' 
- `resc_schedule`: 根据 `n_queries` 调整 `p`, 默认值为True
- `seed`: 起点的随机数种子, 默认为0
- `verbose`: 是否打印进度, 默认为False

测试网络:

```python
net = models.resnet50(pretrained=True)
net = ProcessedModel(net, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
```

测试方法:

```python
attack_square = attack.Square(net, eps=8 / 255, n_queries=100, n_restarts=1, p_init=.8, seed=0, verbose=False, loss='margin', resc_schedule=True)
```

攻击后图片分类正确率: Accuracy on attacked test images: **32.21%**



#### 1.2.2 QueryNet 查询迁移攻击

相关代码:

- `attack/query_attack.py` 里的 `QueryNet`, `PGDGeneratorInfty`, `PGDGenerator2`, `QueryAttack` 类
- `attack/query` 文件夹
- `attack/query_attack_sub` 文件夹

指定的参数有:

- `net`: 攻击网络, 默认为 `resnet50`
- `eps`: 攻击强度 , 默认为 8/255

- `num_x`: 用于评估的样本数量, 默认为10000
- `num_srg`: 替代模型(surrogate)的数量, 默认为0
- `use_nas`: 是否使用NAS训练替代模型, 默认为 `action='store_true'` 
- `use_square_plus`: 是否使用Square+, 默认为 `action='store_true'`
- `p_init`: Square的超参数，改变一个坐标的概率, 默认为0.05
- `run_times`: 重复运行的次数, 默认为1 
- `num_iter`: 最大查询次数, 默认为10000
- `gpu`: GPU个数, 默认为'1' 

测试网络: `resnet101`

```python
net = models.resnet101(pretrained=True)
net = ProcessedModel(net, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
```

测试方法:

```python
attack_query = attack.QueryAttack(net, eps=8/255, num_iter=5000, num_x=10000)
```

输出:

```shell
+--------+-------------+-------------+-------------+-------------+-------------+
| ATTACK | DenseNet121 |   ResNet50  | DenseNet169 |   Square+   |    Square   |
+--------+-------------+-------------+-------------+-------------+-------------+
| WEIGHT |    0.835    |    0.894    |    0.858    |    0.000    |    0.000    |
| CHOSEN |    0.298    |    0.291    |    0.408    |    0.002    |    0.000    |
+--------+-------------+-------------+-------------+-------------+-------------+
+--------+-------------+-------------+-------------+-------------+-------------+
| ATTACK | DenseNet121 |   ResNet50  | DenseNet169 |   Square+   |    Square   |
+--------+-------------+-------------+-------------+-------------+-------------+
| WEIGHT |    0.229    |    0.358    |    0.335    |    0.312    |    0.000    |
| CHOSEN |    0.286    |    0.210    |    0.409    |    0.095    |    0.000    |
+--------+-------------+-------------+-------------+-------------+-------------+
... ...
+--------+-------------+-------------+-------------+-------------+-------------+
| ATTACK | DenseNet121 |   ResNet50  | DenseNet169 |   Square+   |    Square   |
+--------+-------------+-------------+-------------+-------------+-------------+
| WEIGHT |    0.068    |    0.188    |    0.113    |    0.293    |    0.452    |
| CHOSEN |    0.000    |    0.000    |    0.000    |    0.313    |    0.687    |
+--------+-------------+-------------+-------------+-------------+-------------+
```

攻击后图片分类正确率: Accuracy on attacked test images: **0.20%**

注意事项:

1. 在测试方面, 有专门用于 QueryAttack 的测试函数 `test_query`, 进入函数前先进行如下预处理:

   ```python
   if hasattr(attack, 'get_xylogits'): # queryattack
       model.eval()
       x_test, y_test, logits_clean, net = attack.get_xylogits(model, testloader)
       x_adv = attack(net, x_test, y_test, logits_clean) # adv_images after 'iter' queries
       accuracy = test_query(net, x_adv, y_test, trans)
   ```

2. 后面攻击用的 model 用 `VictimImagenet` 处理过, 但在测 clean_accuracy 的时候没有被处理过

3. 运行 `test_imagenet.py` 后得到一个文件夹, 里面有 `adv` 和 `final_adv_images` 两个图片目录, 后者是 5000 次攻击之后得到的图片(但看起来两个目录里的图片是一样的)

4. 前10轮左右的 query 比较慢，是因为用到了 square+ 等，后面的就快了

5. 在文件 `utils.py` 里, 第201行的 `'CDataI'` 写了 imagenet 的文件夹路径, 如果换位置记得改

```python
paths = {
     ...
    'CDataI': '/opt/data/common/ILSVRC2012/val',
    'CGTI':   'data/val.txt'
}
```





### 1.3 迁移攻击

参考 `attack/transfer_attack.py`, 以下迁移攻击均为 linf, non-targeted

1. 测试网络均为: 在 `resnet50` 上生成攻击图片, 在 `resnet101` 上测试迁移后攻击的正确率 :

```python
net = models.resnet50(pretrained=True)
trans_net = models.resnet101(pretrained=True)   # for transferability
net = ProcessedModel(net, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).to(device)
trans_net = ProcessedModel(trans_net, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).to(device)
```

2. 注意, 在测试时, 应当加入 `trans_net` :

```python
attack_accuracy = test(net, testloader, attack=attack, trans=trans_net)  # for tranferability
```

3. 如果不加迁移攻击, 直接测试 PGD-linf 的迁移成功率, 我们可以得到分类正确率为 **36.76%**



#### 1.3.1 MI, DI, TI 攻击

此处 MI, DI, TI 攻击均以 PGD-linf 为基础, 且分别只含 MI, DI, TI 的迁移加成. 后续可以根据自己需要来实现三种迁移攻击的叠加集成, 从而更好地提高迁移率.

**MI攻击:**

指定的参数有:

- `net`: 攻击网络, 默认为 `resnet50`
- `eps`: 攻击强度, 默认为 8/255
- `alpha`: 步长(一般 `alpha = eps*2/steps`)
- `steps`: 循环次数, 默认为10
- `random_start`: 是否使用delta的随机初始化, 默认为True

测试方法:

```python
attack_mi = attack.MI(net, eps=8/255, steps=10, momentum=0.9)
```

攻击后图片分类正确率: Accuracy on attacked test images: **17.64%**



**DI攻击:**

指定的参数有:

- `net`: 攻击网络, 默认为 `resnet50`
- `eps`: 攻击强度, 默认为 8/255

- `alpha`: 步长(一般 `alpha = eps*2/steps`)
- `steps`: 循环次数, 默认为10

- `prob`: 变换概率, 默认值为 0.5
- `image_width`: 随机数的下界, 默认为 200
- `image_resize`: 随机数的上界, 默认为 224

- `random_start`: 是否使用delta的随机初始化, 默认为True

测试方法:

```python
attack_di = attack.DI(net, eps=8/255, steps=10, prob=0.5)
```

攻击后图片分类正确率: Accuracy on attacked test images:  **18.48%**



**TI攻击:**

指定的参数有:

- `net`: 攻击网络, 默认为 `resnet50`
- `eps`: 攻击强度, 默认为 8/255

- `alpha`: 步长 (一般 `alpha = eps*2/steps`)
- `steps`: PGD中的循环次数, 默认为10

- `kernlen`: 高斯核的大小, 默认为 5
- `nsig`: 高斯分布的范围, 默认为 5

- `random_start`: 是否使用delta的随机初始化, 默认为 True

测试方法:

```python
attack_ti = attack.TI(net, eps=8/255, steps=10, kernlen=15, nsig=3)
```

攻击后图片分类正确率: Accuracy on attacked test images: **28.44%**



#### 1.3.2 AoA 攻击

相关代码:

- `attack/transfer_attack.py` 里的 `AoA` 类
- `attack/gradcam` 文件夹, 用于得到注意力计算图

指定的参数有:

- `net`: 攻击网络, 默认为 `resnet50`
- `eps`: 攻击强度, 默认为 8/255

- `model_dict`: 包含 'type', 'arch', 'layer_name', 'input_size' 作为键的字典。
- `type`: 'vgg', 'resnet', 'densenet', 'alexnet', 'squeezenet' 等模型类型, 默认为 'resnet'
- `lamb`: $\lambda$, 注意力攻击和交叉熵之间的关系, 默认为1000
- `layer_name`: 最后一个卷积层输出的特征层, 用于 gradcam 提取注意力计算图
- `alpha`: 步长 (一般 `alpha = eps/num_iter`),  默认为 2/255
- `steps`: 循环次数, 默认为10

攻击网络: `resnet50`; 测试网络: `resnet101`

测试方法: 

```python
attack_aoa = attack.AoA(net, eps=8/255, alpha=1.6/255, steps=10, lamb=10, layer_name="layer4")
```

攻击后图片分类正确率: Accuracy on attacked test images:  **30.11%**



### 1.4 投毒攻击
参考 `attack/adversarial_poisons/`文件夹, 为提高投毒准确率，均采用 targeted-attack.

1. 生成投毒样本

**AdvPosion攻击:**
参考代码：[Adversarial Examples Make Strong Poisons](https://github.com/lhfowl/adversarial_poisons)
```python
bash generate_cifar10.sh # CIFAR10
bash generate_imagenet.sh #ImageNet
```

2. 测试投毒准确率
```python
bash test.sh
```

投毒后在干净样本上的图片分类正确率: **7.36%**（CIFAR10）


## 2. 防御方法

相关代码位于``defence``路径中，测试函数参见```test_blackbox_defence.py```.

### 2.1 AAA

使用方法：包装一个model的前向函数，如下所示。

```python
from defence import AAALinear
net = models.resnet152(pretrained=True)
net = ProcessedModel(net, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
net = AAALinear(cnn=net, arch="resnet101", device="cuda:0", attractor_interval=4, reverse_step=1, num_iter=100, calibration_loss_weight=5, optimizer_lr=0.1)
```

其中，``cnn``为需要包装的网络，``device``指定tensor设备，其余参数保持默认即可。

### 2.2 UniG

使用方法：包装一个model的前向函数，如下所示。

```python
from defence import UniGModel
net = models.resnet152(pretrained=True)
net = ProcessedModel(net, NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
model = UniGModel(model=model, module_name='avgpool', head_name='fc',epoch=5, lr=0.01, delta=0.2, ifcombine=False)
```
其中, ``model``为需要包装的网络，``module_name``为插入UniG模块的前一层，``head_name``为网络线性层的名称，``epoch``为训练UniG使用的epoch数量，``lr``为对应学习率，``delta``为UniG模块参数的最大偏移量，``ifcombine``指定是否使用训练数据微调UniG，除前三个参数外，其余参数默认即可。




### 2.3 测试

测试网络为ResNet152，使用100查询的Square Attack攻击网络，测试结果如下：

|              | Clean Accuracy | Square Accuracy |
| :----------: | :------------: | :-------------: |
|  未防御模型  |     0.783      |      0.422      |
| AAA防御模型  |     0.783      |      0.670      |
| UniG防御模型 |     0.783      |      0.617      |



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

  下面给出``normal_training.sh``脚本案例：
  
  ````python
  # without --eps $eps, meaning attack epsilon is zero, which means vanilla training
  datasets=ImageNet
  device=0,1
  model=resnet101
  path=/opt/data/common/ILSVRC2012/ ## imagenet测试集保存位置
  epochs=90
  DST=eps_8_save_$model    ##模型保存位置
  eps=8    ##若eps==0，则为普通训练
  CUDA_VISIBLE_DEVICES=$device  python3 normal_training.py -a $model \
      --epochs $epochs --workers 8  --dist-url 'tcp://127.0.0.1:1234' \
      --dist-backend 'nccl' --multiprocessing-distributed \
      --world-size 1 --rank 0 $path --save_dir $DST --eps $eps
  ````

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
- ``batch_sze`` : 批次大小，默认为256。如遇爆显存可以按照2的幂次方依次减小，如256->128->64->...
- ``train_start``: 训练初始点，默认为-1
- ``log_dir``: 日志存储地址，默认与模型保存地址相同

在以上设置下，执行``train_dldr.sh``脚本，会提取``$DST``下 epoch 1到epoch60的训练轨迹，然后使用低维训练训练2个epoch，总训练epoch为62，加速比为$\frac{90-62}{90}=31.1\%$. 

> 模型保存在哪里？
> 
> 生成的模型会保存在DST的文件夹中，包含log.pt，ddp0.pt，ddp1.pt三个文件，其中log.pt为配置文件

模型输入：上述参数，SGD模型路径

模型输出：TWA模型路径

下面给出``train_dldr.sh``脚本在训练resnet101、vanilla训练时的案例：

```python
# TWA (DDP version) 60+2
# without --eps $eps, meaning attack epsilon is zero, which means vanilla training
datasets=ImageNet
device=0,1
model=resnet101
wd_psgd=0.00001
lr=0.3
path=/opt/data/common/ILSVRC2012/    ##ImageNet数据集路径
DST=/opt/data/private/checkpoint_resnet18_vanilla_save_DOT_NOT_DELETE    ##resnet101普通训练的路径点
params_start=0
params_end=301
CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node 2 train_dldr.py \
        --epochs 2 --datasets $datasets --opt SGD --schedule step --worker 8 \
        --lr $lr --params_start $params_start  --params_end $params_end  --train_start -1 --wd $wd_psgd \
        --batch-size 256 --arch $model --save-dir $DST --log-dir $DST --eps 0
```

下面给出``train_dldr.sh``脚本在训练resnet101、fgsm训练时的案例：

```python
# with --eps $eps,  meaning adversarial training
datasets=ImageNet
device=0,1
model=resnet101
wd_psgd=0.00001
lr=0.3
path=/opt/data/common/ILSVRC2012/    ##ImageNet数据集路径
DST=/opt/data/private/fgsm-at@eps:4_save_resnet18    ##resnet101 fgsm训练的路径点
CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node 2 train_dldr.py \
        --epochs 2 --datasets $datasets --opt SGD --schedule step --worker 8 \
        --lr $lr --params_start $params_start  --params_end $params_end   --train_start -1 --wd $wd_psgd \
        --batch-size 256 --arch $model --save-dir $DST --log-dir $DST --eps 4
```
