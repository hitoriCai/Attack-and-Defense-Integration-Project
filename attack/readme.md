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

**Linf**		

```python
attack = attack.APGD(net, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
```

**L2**		

```python
attack = attack.APGD(net, norm='L2', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
```



#### 3.1.2 dlr

**Linf**		Accuracy on attacked test images: **0.42%**

```python
attack = attack.APGD(net, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False)
```

**L2**		

```python
attack = attack.APGD(net, norm='L2', eps=8/255, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False)
```



### 3.2 APGDT

#### 3.2.1 ce

**Linf**		

```python
attack = attack.APGDT(net, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False, n_classes=10)
```

**L2**		

```python
attack = attack.APGDT(net, norm='L2', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False, n_classes=10)
```



#### 3.2.2 dlr

**Linf**		

```python
attack = attack.APGDT(net, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False, n_classes=10)
```

**L2**		

```python
attack = attack.APGDT(net, norm='L2', eps=8/255, steps=10, n_restarts=1, seed=0, loss='dlr', eot_iter=1, rho=.75, verbose=False, n_classes=10)
```



## 4. Square









## 5. Query



















