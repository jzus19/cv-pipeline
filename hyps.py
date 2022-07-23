import os 
import sys
import torch.nn as nn
from torch.nn.modules.loss import L1Loss
import torch.optim as optim
from torch.optim import lr_scheduler

def fetch_scheduler(optimizer, scheduler="CosineAnnealingLR"):
    if scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, 
                                                   eta_min=1e-6)
    elif scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  T_0=10,
                                                             T_mult=1, eta_min=1e-6)
    elif scheduler == None:
        return None
        
    return scheduler

def fetch_optimizer(optimizer, model):
    if optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-4, 
                                weight_decay=1e-6)
    return optimizer

def fetch_loss():
    loss = nn.CrossEntropyLoss()
    return loss

def get_hyps(optimizer, scheduler, model):
    optimizer = fetch_optimizer(optimizer, model)
    scheduler = fetch_scheduler(optimizer, scheduler)
    loss = fetch_loss()
    return optimizer, scheduler, loss