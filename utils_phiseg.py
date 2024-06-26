import torch.nn.functional as F
import numpy as np
"""Taken from wkentaro repo"""


# Cross Entropy Loss function that computes softmax of label probs within function
def crossEntropy2D(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target.long(), weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

def diceOverlap(true, pred):
    el_dice = np.count_nonzero(true * pred) * 2.0
    el_dice /= (np.count_nonzero(true) + np.count_nonzero(pred))
    return el_dice