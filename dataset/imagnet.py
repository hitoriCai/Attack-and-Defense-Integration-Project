# 正如在models中提到的那样，dataset 只需要实现数据增强与归一化即可，不用实现标准化

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ImageNetLoader:
    def __init__(self, data_dir, batch_size=32, train=True, num_workers=4, shuffle=True, transform=None):
        self.data_dir = os.path.join(data_dir, 'train' if train else 'val')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transform = transform if transform else self.build_transforms(train)

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



# import os
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# class ImageNetLoader:
#     def __init__(self, data_dir, batch_size=32, train=True, num_workers=4, shuffle=True):
#         self.data_dir = os.path.join(data_dir, 'train' if train else 'val')
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.shuffle = shuffle
#         self.transform = self.build_transforms(train)

#     def build_transforms(self, train):
#         if train:
#             # 更复杂的数据增强可以在训练时应用
#             return transforms.Compose([
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#             ])
#         else:
#             # 验证时使用简单的中心裁剪
#             return transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#             ])

#     def load_data(self):
#         dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
#         # 使用num_workers来提高数据加载速度，shuffle选项由构造函数参数控制
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
#         return dataloader


# class ImageFolder(DatasetFolder):
#     """A generic data loader where the images are arranged in this way by default: ::

#         root/dog/xxx.png
#         root/dog/xxy.png
#         root/dog/[...]/xxz.png

#         root/cat/123.png
#         root/cat/nsdf3.png
#         root/cat/[...]/asd932_.png

#     This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
#     the same methods can be overridden to customize the dataset.

#     Args:
#         root (str or ``pathlib.Path``): Root directory path.
#         transform (callable, optional): A function/transform that takes in a PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         loader (callable, optional): A function to load an image given its path.
#         is_valid_file (callable, optional): A function that takes path of an Image file
#             and check if the file is a valid file (used to check of corrupt files)
#         allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
#             An error is raised on empty folders if False (default).

#      Attributes:
#         classes (list): List of the class names sorted alphabetically.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         imgs (list): List of (image path, class_index) tuples
#     """

#     def __init__(
#         self,
#         root: str,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         loader: Callable[[str], Any] = default_loader,
#         is_valid_file: Optional[Callable[[str], bool]] = None,
#         allow_empty: bool = False,
#     ):
#         super().__init__(
#             root,
#             loader,
#             IMG_EXTENSIONS if is_valid_file is None else None,
#             transform=transform,
#             target_transform=target_transform,
#             is_valid_file=is_valid_file,
#             allow_empty=allow_empty,
#         )
#         self.imgs = self.samples