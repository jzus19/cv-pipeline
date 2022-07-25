import os
import sys
import gc
import time
import copy
from tqdm import tqdm
from collections import defaultdict
from colorama import Fore, Back, Style
from enum import Enum
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import torch 
import numpy as np

def train_one_epoch(model, optimizer, scheduler, dataloader, criterion, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)
        
        batch_size = images.size(0)        
        outputs = model(images)

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        loss = criterion(outputs, labels.long().cuda())            
        loss.backward()
    
        if (step + 1) % 1 == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        del loss
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'], Accuracy=top1, Accuracy5=top5)
        torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss

def valid_one_epoch(model, optimizer, dataloader, criterion, device, epoch):
    model.eval()
    with torch.no_grad():
        dataset_size = 0
        running_loss = 0.0
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:        
            images = data['image'].to(device, dtype=torch.float)
            labels = data['label'].to(device, dtype=torch.long)
            
            batch_size = images.size(0)

            outputs = model(images)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            loss = criterion(outputs, labels)        
            running_loss += (loss.item() * batch_size)
            del loss
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size
            
            bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                            LR=optimizer.param_groups[0]['lr'], Accuracy=top1, Accuracy5=top5) 

            torch.cuda.empty_cache()

        gc.collect()
    
    return epoch_loss

def run_training(model, optimizer, scheduler, train_loader, valid_loader, criterion, device, num_epochs):
     
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           train_loader, criterion, 
                                           device=device, epoch=epoch)
        print("Train epoch loss:", train_epoch_loss) 
        val_epoch_loss = valid_one_epoch(model, optimizer, valid_loader,
                                         criterion, device=device, epoch=epoch)
        print("Val epoch loss:", val_epoch_loss) 
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
                
        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "Loss{:.4f}_epoch{:.0f}.pt".format(best_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res