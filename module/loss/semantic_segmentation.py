import torch
import torch.nn as nn


class SegmentationLosses(nn.Module):
    def __init__(self, weight=None,
                 ignore_index=255, mode='CE', gamma=2, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.mode = mode
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='mean',
                                      ignore_index=ignore_index)

    def forward(self, logit, target):
        """Choices: ['CE' or 'FL']"""
        if self.mode == 'CE':
            return self.CrossEntropyLoss(logit, target)
        elif self.mode == 'FL':
            return self.FocalLoss(logit, target)
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        mask = target >= 0
        return self.ce(logit, target.long())

    def FocalLoss(self, logit, target):

        logpt = -self.ce(logit, target.long())
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt

        return loss

