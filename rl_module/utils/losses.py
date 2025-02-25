import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
import numpy as np

class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        if mask is not None:
            pred = pred * mask
            target = target * mask
            
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Calculate focal loss
        alpha = self.alpha
        gamma = self.gamma
        
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = alpha * (1-pt)**gamma * bce
        
        if mask is not None:
            focal_loss = focal_loss * mask
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class BoundaryLoss(nn.Module):
    """Boundary-aware loss for better edge prediction"""
    def __init__(self, theta: float = 1.0):
        super().__init__()
        self.theta = theta
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Calculate gradients
        target_grad = self._compute_gradient(target)
        pred_grad = self._compute_gradient(pred)
        
        # Calculate boundary loss
        boundary_loss = F.mse_loss(pred_grad, target_grad, reduction='none')
        
        if mask is not None:
            boundary_loss = boundary_loss * mask
            
        return self.theta * boundary_loss.mean()
        
    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        D = x.shape[2:]
        grad = []
        
        for d in range(len(D)):
            slice_pos = [slice(None)] * len(x.shape)
            slice_pos[d+2] = slice(1, None)
            slice_neg = [slice(None)] * len(x.shape)
            slice_neg[d+2] = slice(None, -1)
            
            grad.append(x[slice_pos] - x[slice_neg])
            
        return torch.stack(grad, dim=1)

class CombinedLoss(nn.Module):
    """Combined loss function with multiple components"""
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        boundary_weight: float = 0.5,
        deep_supervision_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        if deep_supervision_weights is None:
            self.deep_supervision_weights = [1.0, 0.8, 0.6, 0.4]
        else:
            self.deep_supervision_weights = deep_supervision_weights
            
    def forward(
        self,
        pred: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if isinstance(pred, list):  # Deep supervision mode
            total_loss = 0
            for p, w in zip(pred, self.deep_supervision_weights):
                loss = self._compute_single_loss(p, target, mask)
                total_loss += w * loss
            return total_loss / len(pred)
        else:
            return self._compute_single_loss(pred, target, mask)
            
    def _compute_single_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        dice = self.dice_loss(pred, target, mask)
        focal = self.focal_loss(pred, target, mask)
        boundary = self.boundary_loss(pred, target, mask)
        
        total_loss = (
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.boundary_weight * boundary
        )
        
        return total_loss 