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

    def forward(self, preds, target):
        """Choices: ['CE' or 'FL']"""
        H1, W1 = preds.size()[2:]
        H2, W2 = target.size()[1:]
        assert H1 == H2 and W1 == W2

        C = preds.size(1)
        preds = preds.transpose(1, 2).transpose(2, 3).contiguous().view(-1, C)
        target = target.view(-1)

        if self.mode == 'CE':
            return self.CrossEntropyLoss(preds, target)
        elif self.mode == 'FL':
            return self.FocalLoss(preds, target)
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, preds, target):
        return self.ce(preds, target)

    def FocalLoss(self, preds, target):
        logpt = -self.ce(preds, target.long())
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt

        return loss

