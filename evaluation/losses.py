"""
Loss functions for segmentation
"""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, H, W)
            target: Ground truth (B, 1, H, W)
        """
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class IoULoss(nn.Module):
    """IoU (Jaccard) Loss"""
    
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class CombinedLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BBoxLoss(nn.Module):
    """Bounding Box Regression Loss"""
    
    def __init__(self):
        super(BBoxLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred_bbox, target_bbox):
        """
        Args:
            pred_bbox: Predicted bbox (B, 4)
            target_bbox: Target bbox (B, 4)
        """
        return self.mse(pred_bbox, target_bbox)
