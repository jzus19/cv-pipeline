import os 
import sys
import torch.nn as nn
from torch.nn.modules.loss import L1Loss
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.functional as F
from torch.optim.optimizer import Optimizer, required
import torch
from fastai.imports import *
from fastai.torch_imports import *
from fastai.torch_core import *
from fastai.layers import *

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
    #elif optimizer == "Ranger":
        #optimizer = ranger(model.parameters())
    
    return optimizer

def fetch_loss():
    loss = LabelSmoothingCrossEntropy()
    #loss = fpLoss()
    return loss

def get_hyps(optimizer, scheduler, model):
    optimizer = fetch_optimizer(optimizer, model)
    scheduler = fetch_scheduler(optimizer, scheduler)
    loss = fetch_loss()
    return optimizer, scheduler, loss

class LabelSmoothingCrossEntropy(Module):
    y_int = True # y interpolation
    def __init__(self, 
        eps:float=0.1, # The weight for the interpolation formula
        weight=None, # Manual rescaling weight given to each class passed to `F.nll_loss`
        reduction:str='mean' # PyTorch reduction to apply to the output
    ): 
        store_attr()

    def forward(self, output, target):
        "Apply `F.log_softmax` on output then blend the loss/num_classes(`c`) with the `F.nll_loss`"
        c = output.size()[1]
        log_preds = F.log_softmax(output, dim=1)
        if self.reduction=='sum': loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=1) #We divide by that size at the return line so sum and not mean
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target.long(), weight=self.weight, reduction=self.reduction)

    def activation(self, out): 
        "`F.log_softmax`'s fused activation function applied to model output"
        return F.softmax(out, dim=-1)
    
    def decodes(self, out):
        "Converts model output to target format"
        return out.argmax(dim=-1)

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

# @delegates(RAdam)
# def ranger(p, lr=1e-3, mom=0.95, wd=0.01, eps=1e-6, **kwargs):
#     "Convenience method for `Lookahead` with `RAdam`"
#     return Lookahead(RAdam(p, lr=lr, mom=mom, wd=wd, eps=eps, **kwargs))

class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95,0.999), eps=1e-5, weight_decay=0):
        #parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        #parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        #N_sma_threshold of 5 seems better in testing than 4.
        #In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        #prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params,defaults)

        #adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        #now we can get to work...
        #removed as we now use step from RAdam...no need for duplicate step counting
        #for group in self.param_groups:
        #    group["step_counter"] = 0
            #print("group step counter init")

        #look ahead params
        self.alpha = alpha
        self.k = k 

        #radam buffer for state
        self.radam_buffer = [[None,None,None] for ind in range(10)]

        #self.first_run_check=0

        #lookahead weights
        #9/2/19 - lookahead param tensors have been moved to state storage.  
        #This should resolve issues with load/save where weights were left in GPU memory from first load, slowing down future runs.

        #self.slow_weights = [[p.clone().detach() for p in group['params']]
        #                     for group in self.param_groups]

        #don't use grad for lookahead weights
        #for w in it.chain(*self.slow_weights):
        #    w.requires_grad = False

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)


    def step(self, closure=None):
        loss = None
        #note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.  
        #Uncomment if you need to use the actual closure...

        #if closure is not None:
            #loss = closure()

        #Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  #get state dict for this param

                if len(state) == 0:   #if first time to run...init dictionary with our desired entries
                    #if self.first_run_check==0:
                        #self.first_run_check=1
                        #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    #look ahead weight storage now in state dict 
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                #begin computations 
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                #compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                #compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1


                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

                #integrated look ahead...
                #we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer'] #get access to slow param tensor
                    slow_p.add_(self.alpha, p.data - slow_p)  #(fast weights - slow weights) * alpha
                    p.data.copy_(slow_p)  #copy interpolated weights to RAdam param tensor

        return loss