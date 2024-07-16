# 攻击部分说明

## 0. initial

```python
net = models.resnet50(pretrained=True)
testset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)
```

learning rate 默认为 0.1		batch_size 因为显存爆炸, 先调成16了

测试集在原始图片上的准确度: **76.15%**

```python
clean_accuracy = test(net, testloader)
attack_accuracy = test(net, testloader, attack=attack)
```



## 1. FGSM

Accuracy on attacked test images: **23.48%**

```python
attack = attack.FGSM(net, eps=8 / 255)
```



## 2. PGD

**Linf**		Accuracy on attacked test images:  **0.53%**

```python
attack = attack.PGD(net, eps=8 / 255, alpha=1 / 255, steps=10, random_start=True) 
```

**L2**		Accuracy on attacked test images: **34.39%**  (啊?)

```python
attack = attack.PGDL2(net, eps=1.0, alpha=0.2, steps=10, random_start=True)
```



## 3. AutoPGD

### 3.1 APGD

#### 3.1.1 ce

**Linf**		Accuracy on attacked test images: **0.27%**

```python
attack = attack.APGD(net, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
```

**L2**		Accuracy on attacked test images: **32.23%**

```python
attack = attack.APGD(net, norm='L2', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
```



#### 3.1.2 dlr

**Linf**		Accuracy on attacked test images: **0.42%**

```python
attack = attack.APGD(net, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False)
```

**L2**		Accuracy on attacked test images: **32.23%**

```python
attack = attack.APGD(net, norm='L2', eps=3.0, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False)
```



### 3.2 APGDT

(貌似因为一个 iter 后, 显存没有正确释放, 所以跑到一半炸了, 这个问题先放着)

#### 3.2.1 ce

**Linf**		

```python
attack = attack.APGDT(net, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False, n_classes=1000)
```

**L2**		

```python
attack = attack.APGDT(net, norm='L2', eps=3.0, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False, n_classes=1000)
```



#### 3.2.2 dlr

**Linf**		

```python
attack = attack.APGDT(net, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False, n_classes=10)
```

**L2**		

```python
attack = attack.APGDT(net, norm='L2', eps=3.0, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False, n_classes=1000)
```



## 4. Square

Accuracy on attacked test images: **23.76%**

```python
attack = attack.Square(net, norm='Linf', eps= 8/225, n_queries=100, n_restarts=1, p_init=.8, seed=0, verbose=False, targeted=False, loss='margin', resc_schedule=True)
```









## 5. Query

用 `querynet.py` 跑, 单独的, 用的默认参数见下

![image-20240716164803875](D:\SJTU_research\Attack-and-Defense-Integration-Project\attack\assets\image-20240716164803875.png)

```python
# imagenet
assert (not args.use_nas), 'NAS is not supported for ImageNet for resource concerns'
if not ((args.l2_attack and args.eps == 5) or (not args.l2_attack and args.eps == 12.75)):
    print('Warning: not using default eps in the paper, which is l2=5 or linfty=12.75 for ImageNet.')
batch_size = 100 if model_name != 'resnext101_32x8d' else 32
model = VictimImagenet(model_name, batch_size=batch_size) if model_name != 'easydlmnist' else VictimEasydl(arch='easydlmnist')
x_test, y_test = load_imagenet(args.num_x, model)

logits_clean = model(x_test)
corr_classified = logits_clean.argmax(1) == y_test.argmax(1)
print('Clean accuracy: {:.2%}'.format(np.mean(corr_classified)) + ' ' * 40)
y_test = dense_to_onehot(y_test.argmax(1), n_cls=10 if dataset != 'imagenet' else 1000)
for run_time in range(args.run_times):
    attack(model, x_test[corr_classified], y_test[corr_classified], logits_clean[corr_classified], dataset, batch_size, run_time, args, log)
```

在第 7235 轮之后, 达到 0 正确率

```shell
7234: Acc=0.10%, AQ_suc=45.95, MQ_suc=7.0, AQ_all=53.14, MQ_all=7.0, ALoss_all=-1.82, |D|=2000, Time=459.5s
7235: Acc=0.00%, AQ_suc=53.14, MQ_suc=7.0, AQ_all=53.14, MQ_all=7.0, ALoss_all=-1.82, |D|=2000, Time=459.6s
```

















