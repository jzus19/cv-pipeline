import os 
import sys
import torch.nn as nn
from torch.nn.modules.loss import L1Loss
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.functional as F
import torch

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
    #loss = fpLoss()
    return loss

def get_hyps(optimizer, scheduler, model):
    optimizer = fetch_optimizer(optimizer, model)
    scheduler = fetch_scheduler(optimizer, scheduler)
    loss = fetch_loss()
    return optimizer, scheduler, loss

eps = 1e-10
class fpLoss(nn.Module):
    def __init__(self, ):
        super(fpLoss, self).__init__()

    def cross_entropy(self, logits, onehot_labels, ls=False):
        if ls:
            onehot_labels = 0.9 * onehot_labels + 0.1 / logits.size(-1)
            onehot_labels = onehot_labels.double()
        return (-1.0 * torch.mean(torch.sum(onehot_labels * F.log_softmax(logits, -1), -1), 0))


    def neg_entropy(self, logits):
        probs = F.softmax(logits, -1)
        return torch.mean(torch.sum(probs * F.log_softmax(logits, -1), -1), 0)

    def forward(self, targets, outputs, bi_outputs,):
        # Loss_cls
        difference = F.softmax(outputs, -1) - F.softmax(bi_outputs, -1)  
        onehot_labels = F.one_hot(targets, outputs.size(-1))
        loss_cls = self.cross_entropy(outputs + difference, onehot_labels, True)
        # tiny-imagenet 上， alpha=1, beta=1, ls=True, best test acc: 0.5870
        # tiny-imagenet 上， alpha=1, beta=1, ls=False, best test acc: 0.5840
        # 所以ls不是主要原因

        # R_attention
        # multi_warm_lb = bi_outputs > 0.0
        multi_warm_lb = F.softmax(bi_outputs/2, -1) > 1.0/bi_outputs.size(-1)
        multi_warm_lb = torch.clamp(multi_warm_lb.double() + onehot_labels, 0, 1)
        multi_warm_lb = multi_warm_lb/torch.sum(multi_warm_lb, -1, True)
        R_attention = self.cross_entropy(outputs, multi_warm_lb.detach(), False)# 与FRSKD比较时，使用了detach()

        # R_entropy
        R_negtropy = self.neg_entropy(outputs)

        fp_loss = loss_cls + R_attention + R_negtropy

        # test for CE + neg_entropy
        # loss_cls = self.cross_entropy(outputs, onehot_labels)
        # fp_loss = loss_cls + R_negtropy # 已经试验证明 CE + negtive_entropy的CUB200精度（59.10%）低于loss_cls + negtive_entropy的精度(60.72%)
        return fp_loss

