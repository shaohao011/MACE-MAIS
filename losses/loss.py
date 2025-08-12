import torch
import torch.nn as nn
import torch.nn.functional as F
from sksurv.metrics import concordance_index_censored


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean',smooth=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',label_smoothing=self.smooth)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
        
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),t=t,
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)

# # TODO: document better and clean up
def nll_loss(h, y, c,t, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # print("S.shape", S.shape, S)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss 

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss 


class MiccaiSurvLoss(nn.Module):
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c,weight_iter=1.0):

        return nll_loss_miccai(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),t=t,
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction,weight_iter=weight_iter)

# # TODO: document better and clean up
def nll_loss_miccai(h, y, c,t, alpha=0.0, eps=1e-7, reduction='mean',weight_iter=1.0):
    y = y.type(torch.int64)
    c = c.type(torch.int64)
    hazards = torch.sigmoid(h)
    # process hazards according to miccai
    
    
    S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    neg_l = censored_loss + uncensored_loss
    # censored_p:
    for i in range(h.shape[0]):
        if int(c[i]) == 1:  # 
            new_hazard = torch.ones_like(hazards[i])  # 
            new_hazard[:y[i] + 1] = 0.0
            hazards = hazards * new_hazard
            hazards[i] = torch.nn.functional.softmax(hazards[i], dim=0)  #
            
    S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    censored_p_loss = - c * torch.log(s_this)
    
    loss = uncensored_loss + 0.6*(censored_loss+weight_iter*censored_p_loss)

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


def ranking_loss(risk, event_time,c, alpha=0.1, delta=0.1):
    n = risk.size(0)
    loss = torch.tensor(0.0,device=risk.device)
    for i in range(n):
        for j in range(i + 1, n):
            if c[i]==0.0:
                if event_time[i] < event_time[j]:
                    loss += max(0, risk[j] - risk[i] + delta)
    return loss

