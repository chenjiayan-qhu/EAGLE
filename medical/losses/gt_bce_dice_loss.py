from ast import alias
import torch.nn as nn
import torch 
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()
    
    def forward(self, pred, target):
        size = pred.size(0)                 # batch size
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        loss = self.bceloss(pred_, target_)
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class GT_BceDiceLoss(nn.Module):
    """
    
    Input: 
        gt_pre: (gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1)
        out: pred of model: B, C, H, W
        target: label: B, C, H, W
    Args:
        wb: weight of bceloss, default 1
        wd: weight of diceloss, default 1
    """
    def __init__(self, wb=1, wd=1):
        super().__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = ( self.bcedice(gt_pre5, target) * 0.1 + 
                    self.bcedice(gt_pre4, target) * 0.2 + 
                    self.bcedice(gt_pre3, target) * 0.3 + 
                    self.bcedice(gt_pre2, target) * 0.4 + 
                    self.bcedice(gt_pre1, target) * 0.5 )
        return bcediceloss + gt_loss


class GT_BceDiceLoss4(nn.Module):
    """
  
    Input: 
        gt_pre: (gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1)
        out: pred of model: B, C, H, W
        target: label: B, C, H, W
    Args:
        wb: weight of bceloss, default 1
        wd: weight of diceloss, default 1
    """
    def __init__(self, wb=1, wd=1):
        super().__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = (  
                    self.bcedice(gt_pre4, target) * 0.1 + 
                    self.bcedice(gt_pre3, target) * 0.2 + 
                    self.bcedice(gt_pre2, target) * 0.3 + 
                    self.bcedice(gt_pre1, target) * 0.5 )
        return bcediceloss + gt_loss