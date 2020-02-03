import numpy as np


class Metric(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def confusion_matrix(self, preds, target):
        assert preds.shape == target.shape
        # rows -> predict
        # cols -> target
        preds = preds.astype(np.int)
        target = target.astype(np.int)
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

        tp = np.diagonal(m)
        # fn = fn + tp
        fn = np.sum(m, 0)
        # fp = fp + tp
        fp = np.sum(m, 1)
        miou = tp / (fn + fp - tp)
        miou = np.nanmean(miou)

        return miou

