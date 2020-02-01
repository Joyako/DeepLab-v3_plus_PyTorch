import numpy as np


class Metric(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def confusion_matrix(self, preds, target):
        assert preds.shape == target.shape
        # rows -> predict
        # cols -> target
        preds = preds.astype(np.uint8)
        target = target.astype(np.uint8)
        m = np.zeros((self.num_classes, self.num_classes), dtype=np.float32)
        N, H, W = target.shape
        for n in range(N):
            for i in range(H):
                for j in range(W):
                    p = preds[n][i][j]
                    q = target[n][i][j]
                    m[q][p] += 1.

        return m

    def recall(self, preds, target, m=None):
        if m is None:
            m = self.confusion_matrix(preds, target)
        sum = m.sum(axis=0)

        return m.diagonal() / sum

    def precision(self, preds, target, m=None):
        if m is None:
            m = self.confusion_matrix(preds, target)
        sum = m.sum(axis=1)

        return m.diagonal() / sum

    def mIoU(self, preds, target, m=None):
        if m is None:
            m = self.confusion_matrix(preds, target)

        miou = 0.
        for i in range(self.num_classes):
            tp = m[i, i]
            fn = m[i, 1:].sum()
            fp = m[1:, i].sum()
            miou += 1. * tp / (tp + fp + fn)

        return  miou / self.num_classes

