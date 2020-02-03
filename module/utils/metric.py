import numpy as np


class Metric(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def confusion_matrix(self, preds, target):
        assert preds.shape == target.shape

        mask = (target >= 0) & (target < self.num_classes)
        label = self.num_classes * target[mask].astype('int') + preds[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)

        confusion_matrix = count.reshape(self.num_classes, self.num_classes)

        return confusion_matrix

    def recall(self, preds, target, m=None):
        # rec = tp / (tp + fn)
        assert preds.shape == target.shape
        if m is None:
            m = self.confusion_matrix(preds, target)
        sum = m.sum(axis=1)

        return m.diagonal() / sum

    def precision(self, preds, target, m=None):
        # pre = tp / (tp + fp )
        assert preds.shape == target.shape
        if m is None:
            m = self.confusion_matrix(preds, target)
        sum = m.sum(axis=0)

        return m.diagonal() / sum

    def add(self, preds, target, m=0.):
        m += self.confusion_matrix(preds, target)

        return m

    def mIoU(self, preds=None, target=None, m=None):
        # IoU = tp / (tp + fp + fn)
        if m is None:
            assert (preds is not None and target is not None)
            m = self.confusion_matrix(preds, target)

        tp = np.diagonal(m)
        # fn = fn + tp
        fn = np.sum(m, 1)
        # fp = fp + tp
        fp = np.sum(m, 0)
        eps = 1e-5
        miou = tp / (fn + fp - tp + eps)
        miou = np.nanmean(miou)

        return miou


if __name__ == '__main__':
    # predict
    p = np.array([[2, 0, 1, 0],
                  [2, 1, 1, 0],
                  [0, 1, 1, 0],
                  [1, 1, 0, 1]])

    # target
    t = np.array([[2, 1, 1, 2],
                  [2, 1, 1, 0],
                  [0, 1, 0, 0],
                  [1, 1, 0, 0]])

    metric = Metric(3)
    m = metric.confusion_matrix(p, t)
    print(m)
    print('recall : ', metric.recall(p, t, m))
    print('precision : ', metric.precision(p, t, m))
    print('mIoU : ', metric.mIoU(p, t, m))
