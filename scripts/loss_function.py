import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.5, gamma=0.2, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Weight for BCE loss
        self.beta = beta    # Weight for Dice loss
        self.gamma = gamma  # Weight for Focal loss
        self.focal_gamma = focal_gamma  # Focusing parameter for Focal loss
        self.eps = 1e-6     # Small epsilon to avoid division by zero
        
    def forward(self, predictions, targets):
        # Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='mean')
        
        # Dice Loss
        # Flatten predictions and targets
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions_flat * targets_flat).sum()
        union = predictions_flat.sum() + targets_flat.sum()
        
        # Calculate Dice score and loss
        dice_score = (2. * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice_score
        
        # Focal Loss - fixed implementation
        # Calculate probability of the correct class
        pt = targets * predictions + (1 - targets) * (1 - predictions)
        # Apply the focusing parameter
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Compute focal loss manually without using weight parameter
        focal_loss = -((focal_weight * targets * torch.log(predictions + self.eps)) + 
                      (focal_weight * (1 - targets) * torch.log(1 - predictions + self.eps))).mean()
        
        # Combine all losses with their respective weights
        combined_loss = (self.alpha * bce_loss) + (self.beta * dice_loss) + (self.gamma * focal_loss)
        
        return combined_loss