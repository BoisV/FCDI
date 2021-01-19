# -*- coding: utf-8 -*-
# @Time : 2021/1/10 11:18
# @Author : lepold
# @File : data_interface.py

import os
from typing import Callable, Dict, Optional

from PIL import Image
from torchvision.datasets.vision import VisionDataset


class FCDIDaset(VisionDataset):

    classes = ["Motorola-X", "Motorola-Droid-Maxx", "LG-Nexus-5x", "Samsung-Galaxy-Note3",
               "Samsung-Galaxy-S4", "iPhone-4s", "Motorola-Nexus-6", "Sony-NEX-7", "iPhone-6", "HTC-1-M7"]

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def __init__(self,
                 root,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super(FCDIDaset, self).__init__(root=root,
                                        transform=transform,
                                        target_transform=target_transform)
        self.train = train

        if self.train:
            self.root = os.path.join(root, 'train')
        else:
            self.root = os.path.join(root, 'test')

        if not os.path.exists(self.root):
            raise FileNotFoundError(self.root + ' dir is Not Found!')

        self.img_pathes = []
        self.targets = []

        if not os.path.exists(os.path.join(self.root, 'list.txt')):
            raise FileNotFoundError(os.path.join(
                self.root, 'list.txt') + ' is Not Found!')

        with open(os.path.join(self.root, 'list.txt'), 'r') as f:
            while f.readable():
                line = f.readline().strip()
                if line == '':
                    break
                path = line[:-3]
                label = int(line[-2:])
                self.img_pathes.append(path)
                self.targets.append(label)

    def __getitem__(self, index):
        img_path, target = self.img_pathes[index], int(self.targets[index])

        img = Image.open(os.path.join(self.root, img_path))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)


# if __name__ == "__main__":
# from torch.utils.data.dataloader import DataLoader
# from torchvision.transforms import transforms
#     root = '../../data/patches/phaseA'

#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     dataset = FCDIDaset(root=root,
#                         train=True,
#                         transform=transform)
#     print(dataset.class_to_idx)
#     dl = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

#     for X, Y in dl:
#         print(X.shape)
#         print(Y)
