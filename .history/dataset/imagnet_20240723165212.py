# 正如在models中提到的那样，dataset 只需要实现数据增强与归一化即可，不用实现标准化

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ImageNetLoader:
    def __init__(self, data_dir, batch_size=32, train=True, num_workers=4, shuffle=True):
        self.data_dir = os.path.join(data_dir, 'train' if train else 'val')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transform = self.build_transforms(train)

    def build_transforms(self, train):
        if train:
            # 更复杂的数据增强可以在训练时应用
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            # 验证时使用简单的中心裁剪
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])

    def load_data(self):
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        # 使用num_workers来提高数据加载速度，shuffle选项由构造函数参数控制
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        return dataloader
