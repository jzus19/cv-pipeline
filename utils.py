import os 
import re
from os.path import join as osp
import sys
from pathlib import Path
from glob import glob
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

def get_dataloaders(data_path):
    train_path = osp(data_path, "train")
    val_path = osp(data_path, "val")
    
    train_files = glob(train_path+"**/**/*.JPEG", recursive=True)
    val_files = glob(val_path+"**/**/*.JPEG", recursive=True)
    
    files = train_files + val_files
    le = LabelEncoder()
    classes = []
    labels = []
    for i, file in enumerate(files):
        clas = re.split('train|val', file)[1][1:10]
        classes.append(clas)
    labels = le.fit_transform(classes)

    with open("le.pkl", "wb") as fp:
        joblib.dump(le, fp)
    
    train_classes, val_classes = classes[:len(train_files)], classes[len(train_files):]
    train_labels, val_labels = labels[:len(train_files)], labels[len(train_files):]
    
    train_data = list(zip(train_files, train_labels))
    val_data = list(zip(val_files, val_labels))
    
    return 

class DogDataset(Dataset):
    def __init__(self, ):
        super(DogDataset, self).__init__()