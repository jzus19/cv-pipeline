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

from src.dataset import get_dataloaders
from src.model import get_model
from src.hyps import get_hyps
from src.utils import run_training

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def train(opt):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logging.info("Device name:", device)
    torch.backends.cudnn.benchmark = True

    train_dl, val_dl = get_dataloaders(opt)

    model = get_model()
    model.to(device)
    
    if opt.checkpoint:
        model.load_state_dict(torch.load(opt.checkpoint))

    optimizer, scheduler, criterion = get_hyps(opt.optimizer, opt.scheduler, model)
    
    model, history = run_training(model, optimizer, scheduler, train_dl, 
                                    val_dl, criterion, device, opt.epochs)

    with open("history_loss.pkl", "wb") as pkl_handle:
	    pickle.dump(history, pkl_handle)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/imagewoof2/', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--optimizer', type=str, default="Adam", help='optimizer')
    parser.add_argument('--scheduler', type=str, default="CosineAnnealingLR", help='scheduler')
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint')
    parser.add_argument('--debug', default="True", action=argparse.BooleanOptionalAction)
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
    train(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)