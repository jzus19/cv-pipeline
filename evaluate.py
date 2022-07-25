import os
from os.path import join as osp
from pathlib import Path
import sys
import logging 
import argparse
import torch
import torch.nn as nn
import numpy
import pickle
from glob import glob
from model import get_model
from hyps import get_hyps
from utils import run_training
from dataset import get_dataloaders
from tqdm import tqdm
from utils import AverageMeter, accuracy
import gc

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def evaluate(opt):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logging.info("Device name:", device)
    torch.backends.cudnn.benchmark = True
    _, val_dl = get_dataloaders(opt)
    
    model = get_model()
    model.to(device)
    
    if opt.checkpoint:
        model.load_state_dict(torch.load(opt.checkpoint))

    model.eval()
    with torch.no_grad():
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        bar = tqdm(enumerate(val_dl), total=len(val_dl))
        for step, data in bar:        
            images = data['image'].to(device, dtype=torch.float)
            labels = data['label'].to(device, dtype=torch.long)
            outputs = model(images)
            
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            bar.set_postfix(Accuracy=top1, Accuracy5=top5) 

            torch.cuda.empty_cache()

        gc.collect()

    print(top1, top5)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT /'data/imagewoof2/', help='dataset path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint')
    parser.add_argument('--debug', default="True", action=argparse.BooleanOptionalAction)
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
    evaluate(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)