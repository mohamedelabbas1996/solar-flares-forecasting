import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic


class FocalBCELoss(nn.Module):
    """
    Combination of BCE and Focal Loss
    """

    def __init__(self, alpha=1, gamma=2, weight=None, reduction="mean"):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        # Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy_with_logits(x, target)

        # Compute the focal loss term
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma + bce_loss

        # Combine both losses
        loss = focal_loss

        # if self.reduction == "mean":
        #     loss = torch.mean(loss)
        # elif self.reduction == "sum":
        #     loss = torch.sum(loss)

        return loss


if __name__ == "__main__":
    criterion = FocalBCELoss()
    logits = torch.rand(10, 1)
    target = torch.randint(low=0, high=1, size=(10, 1), dtype=torch.float32)
    ic(logits.dtype)
    ic(target.dtype)
    loss = criterion(logits, target)
    ic(loss)
