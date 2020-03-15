import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SegmentationLosses(nn.Module):
    def __init__(self, num_classes=8, mode='CE', weights=None,
                 ignore_index=255, gamma=2, alpha=0.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.mode = mode
        self.weights = weights
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, preds, target):
        """"""
        H1, W1 = preds.size()[2:]
        H2, W2 = target.size()[1:]
        assert H1 == H2 and W1 == W2

        if self.mode == 'CE':
            return self.CrossEntropyLoss(preds, target)
        elif self.mode == 'FL':
            return self.FocalLoss(preds, target)
        elif self.mode == 'Dice':
            return self.GeneralizedSoftDiceLoss(preds, target)
        elif self.mode == 'Dice2':
            return self.BatchSoftDeviceLoss(preds, target)
        elif self.mode == 'CE || Dice':
            loss = self.CrossEntropyLoss(preds, target) + \
                   self.GeneralizedSoftDiceLoss(preds, target)
            return loss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, preds, target):
        """

        :param preds: Tensor of shape [N, C, H, W]
        :param target: Tensor of shape [N, H, W]
        :return:
        """
        device = target.device
        # if self.weights is not None:
        #     weight = self.weights.to(device)
        # else:
        #     arr = target.data.cpu().numpy().reshape(-1)
        #     weight = np.bincount(arr)
        #     weight = weight.astype(np.float)
        #     # weight = weight.sum() / weight
        #     weight = weight / weight.sum()
        #     median = np.median(weight)
        #     for i in range(weight.shape[0]):
        #         if int(weight[i]) == 0:
        #             continue
        #         weight[i] = median / weight[i]
        #     weight = torch.from_numpy(weight).to(device).float()

        return F.cross_entropy(preds, target, weight=self.weights.to(device), ignore_index=self.ignore_index)

    def FocalLoss(self, preds, target):
        """
        FL = alpha * (1 - pt) ** beta * log(pt)
        :param preds: Tensor of shape [N, C, H, W]
        :param target: Tensor of shape [N, H, W]
        :return:
        """
        logits = -F.cross_entropy(preds, target.long(),
                                  ignore_index=self.ignore_index)
        pt = torch.exp(logits)
        if self.alpha is not None:
            logits *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logits

        return loss

    def GeneralizedSoftDiceLoss(self, preds, target):
        """
        Paper:
            https://arxiv.org/pdf/1606.04797.pdf
        :param preds: Tensor of shape [N, C, H, W]
        :param target: Tensor of shape [N, H, W]
        :return:
        """
        # overcome ignored label
        ignore = target.data.cpu() == self.ignore_index
        label = target.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(preds).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()

        # compute loss
        probs = torch.sigmoid(preds)
        numer = torch.sum((probs * lb_one_hot), dim=(2, 3))
        denom = torch.sum(probs.pow(1) + lb_one_hot.pow(1), dim=(2, 3))
        if not self.weights is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer, dim=1)
        denom = torch.sum(denom, dim=1)
        smooth = 1
        loss = 1 - (2 * numer + smooth) / (denom + smooth)

        if self.reduction == 'mean':
            loss = loss.mean()
        return loss

    def BatchSoftDeviceLoss(self, preds, target):
        """

        :param preds:
        :param target:
        :return:
        """
        # overcome ignored label
        ignore = target.data.cpu() == self.ignore_index
        target = target.clone()
        target[ignore] = 0
        lb_one_hot = torch.zeros_like(preds).scatter_(1, target.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()

        # compute loss
        probs = torch.sigmoid(preds)
        numer = torch.sum((probs * lb_one_hot), dim=(2, 3))
        denom = torch.sum(probs.pow(1) + lb_one_hot.pow(1), dim=(2, 3))
        if not self.weights is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer)
        denom = torch.sum(denom)
        smooth = 1
        loss = 1 - (2 * numer + smooth) / (denom + smooth)

        return loss


if __name__ == '__main__':
    criteria = SegmentationLosses(mode='CE')
    #  logits = torch.randn(16, 19, 14, 14)
    im = torch.randn(16, 3, 14, 14)
    label = torch.randint(0, 19, (16, 14, 14)).long()
    net = torch.nn.Conv2d(3, 19, 3, 1, 1)
    print(label.dtype)
    label[2, 3, 3] = 255
    print(label.dtype)

    logits = net(im)
    loss = criteria(logits, label)
    loss.backward()
    print(loss)

