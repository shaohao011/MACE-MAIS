import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_steps:
            # Linear warmup
            lr_scale = float(epoch) / float(max(1, self.warmup_steps))
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
            lr_scale = 0.5 * (1. + torch.cos(torch.pi * progress))
        
        return [base_lr * lr_scale for base_lr in self.base_lrs]