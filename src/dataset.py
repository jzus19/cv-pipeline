import os 
import re
from os.path import join as osp
import sys
from pathlib import Path
from glob import glob
import joblib
import pandas as pd
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

def get_transformations():
    data_transforms = {
    "train": A.Compose([
        A.Resize(500, 380),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=60, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    
    "valid": A.Compose([
        A.Resize(500, 380),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
    }
    return data_transforms


def get_dataloaders(config):
    train_path = osp(config.data, "train")
    val_path = osp(config.data, "val")
    
    train_files = glob(train_path+"**/**/*.JPEG", recursive=True)
    val_files = glob(val_path+"**/**/*.JPEG", recursive=True)    
    files = train_files + val_files

    classes = []
    labels = []
    for i, file in enumerate(files):
        clas = re.split('train|val', file)[1][1:10]
        classes.append(clas)
    le = LabelEncoder()
    if config.checkpoint == "":        
        labels = le.fit_transform(classes)
        np.save('classes.npy', le.classes_)
    
    else:
        le.classes_ = np.load('classes.npy')
        labels = le.transform(classes)

    train_classes, val_classes = classes[:len(train_files)], classes[len(train_files):]
    train_labels, val_labels = labels[:len(train_files)], labels[len(train_files):]

    train_data = list(zip(train_files, train_labels))
    val_data = list(zip(val_files, val_labels))
    
    transformations = get_transformations()

    train_ds = DogDataset(train_data, transformations["train"])
    val_ds = DogDataset(val_data, transformations["valid"])
        
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, 
                            num_workers=8, drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size*2, shuffle=True, num_workers=8)
    
    return train_dl, val_dl

class DogDataset(Dataset):
    def __init__(self, data, transforms=None):
        super(DogDataset, self).__init__()
        self.data = data
        self.transforms = transforms

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path, label = item

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        }