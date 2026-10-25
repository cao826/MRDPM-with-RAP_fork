import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def mae_loss(output, target):
    return F.l1_loss(output, target)

def smooth_l1_loss(output, target):
    return F.smooth_l1_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)


# bran_ext_plan.md §3: Zeng et al.'s extreme loss (gan_bpm models/losses.py:l_ext) on MRDPM's
# [0, 1] label. gan_bpm's label is B = 2·gt − 1, so with ext weight w,
#     L1 + w · mean(W · (pred − gt)²),   W = (2·gt − 1)²,
# is gan_bpm's 100·L_pix + λ_ext·L_ext divided by 200 when w = λ_ext / 50: Zeng's λ_ext = 50 is w = 1.
EXT_SIDES = ('both', 'low', 'high')


def ext_weight(gt, side='both'):
    """W = (2·gt − 1)²: 1 at both ends of the label's range, 0 at its middle (rCBF 1, Tmax 6 s).

    side 'low' keeps the lower half only (rCBF's core end), 'high' the upper half (Tmax's lesion
    end); the two halves add up to 'both'."""
    b = 2. * gt - 1.
    if side == 'low':
        b = b.clamp(max=0.)
    elif side == 'high':
        b = b.clamp(min=0.)
    elif side != 'both':
        raise ValueError('ext side must be one of {}, not {!r}'.format(EXT_SIDES, side))
    return b * b


def ext_term(output, target, weight, valid):
    """mean(valid · weight · (output − target)²) over every element, normalized as the L1 and MSE
    terms beside it in Network.forward are."""
    return (valid * weight * (output - target) ** 2).mean()

    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

