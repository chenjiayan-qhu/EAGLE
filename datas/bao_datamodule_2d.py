import glob
import os
import sys
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

from monai import transforms
from monai.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

""" 
Train loader: [N, 1, H, W]
Val loader: [N, 1, H, W]
Train set, Val set, Test set: [1, H, W]
"""


class BaoTrainDataset(Dataset):
    """
    - The input data is a list of dictionaries:
    - Example:
        [{                             {                             {
            'image': 'image1.png',      'image': 'image2.png',        'image': 'image3.png',
            'label': 'label1.png',      'label': 'label2.png',        'label': 'label3.png',
        },                             },                             }]    
   
    """
    def __init__(
            self,
            data,
            transform=None,
    ) -> None:
        
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        data_dict = self.data[index]  # data_dict = {'img': '.png', 'label': '.png'}
        data = self.transform(data_dict)       # Apply transform
        return data  
    

class BaoValDataset(Dataset):
    """
    - Data set for validation.
    - The input data is a list of dictionaries
    - Example:
        [{                             {                             {
            'image': 'image1.png',      'image': 'image2.png',        'image': 'image3.png',
            'label': 'label1.png',      'label': 'label2.png',        'label': 'label3.png',
        },                             },                             }]       
    """
    def __init__(
            self,
            data,
            transform=None,
    ) -> None:
        
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_dict = self.data[index]
        data = self.transform(data_dict)       # Apply transform
        return data  # ([1, 224, 224])


class BaoTestDataset(Dataset):

    def __init__(
        self,
        data,
        transform=None,
    ) -> None:
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_dict = self.data[index]
        data = self.transform(data_dict)
        return data


class BaoDataModule2d(LightningDataModule):
    def __init__(
            self,
            train_dir: str,
            val_dir: str,
            test_dir: str,
            batch_size: int = 4,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.train_files = sorted(glob.glob(os.path.join(train_dir,"image/*")))     # List of path
        self.train_labels = sorted(glob.glob(os.path.join(train_dir,"mask/*")))     # List of path
        self.train_data = []
        for img_path, seg_path in zip(self.train_files, self.train_labels):
            self.train_data.append({
                "image": img_path,
                "label": seg_path,
            })

        self.val_files = sorted(glob.glob(os.path.join(val_dir,"image/*")))     # List of path
        self.val_labels = sorted(glob.glob(os.path.join(val_dir,"mask/*")))     # List of path
        self.val_data = []
        for img_path, seg_path in zip(self.val_files, self.val_labels):
            self.val_data.append({
                "image": img_path,
                "label": seg_path,
            })

        self.test_files = sorted(glob.glob(os.path.join(test_dir,"image/*")))     # List of path
        self.test_labels = sorted(glob.glob(os.path.join(test_dir,"mask/*")))     # List of path
        self.test_data = []
        for img_path, seg_path in zip(self.test_files, self.test_labels):
            self.test_data.append({
                "image": img_path,
                "label": seg_path,
            })

        self.data_train: BaoTrainDataset = None
        self.data_val: BaoValDataset = None
        self.data_test: BaoTestDataset = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None) -> None:
        train_transforms = transforms.Compose([ 
            # loadding image -> tensor
            transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
            # scale intensity 0,1
            transforms.ScaleIntensityd(keys=["image","label"]),
            # ### resize
            transforms.Resized(keys=["image", "label"], spatial_size=(512, 512), mode=("area", "nearest")),
            #####
            # random rotate
            transforms.RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, 
                                    mode=("bilinear", "nearest"), prob=0.2),
            # random zoom
            transforms.RandZoomd(keys=["image", "label"], min_zoom=0.8, max_zoom=1.2, 
                                    mode=("bilinear", "nearest"), align_corners=(True,None),    
                                    prob=0.16),
            # random smoothed
            transforms.RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), prob=0.15),
            transforms.RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
            transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            transforms.RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            transforms.RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            transforms.RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            transforms.ToTensord(keys=["image", "label"]),    
        ])
        val_transforms = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
            transforms.ScaleIntensityd(keys=["image","label"]),
            #### resize
            transforms.Resized(keys=["image", "label"], spatial_size=(512, 512), mode=("area", "nearest")),
            #####
            transforms.ToTensord(keys=["image", "label"]),
        ])


        self.data_train = BaoTrainDataset(data=self.train_data, transform=train_transforms)
        self.data_val = BaoValDataset(data=self.val_data, transform=val_transforms)
        self.data_test = BaoTestDataset(data=self.test_data, transform=val_transforms)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
             
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test
        
        
if __name__ == '__main__':
    
    train_dir = '../EAGLE/data/train'
    val_dir = '../EAGLE/data/val'
    test_dir ='../EAGLE/data/test'
    
    dm = BaoDataModule2d(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=8,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()
    train_loader, _ = dm.make_loader
    for batch in train_loader:
        img = batch['image']
        mask = batch['label']
        print(img.shape, mask.shape)
        print(torch.unique(mask))
        break
    