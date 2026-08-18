import torch

class SimpleMetric():
    """ Base class for any metric that keeps a numerator and denominator,
    designed as a simple drop-in replacement for torchmetrics.
    """
    def __init__(self, v=None):
        if v is not None:
            self.v = v
        else:
            self.v = torch.zeros(2)

    def reset(self):
        self.v = torch.zeros_like(self.v)

    
    def to(self, device):
        self.v = self.v.to(device)
        return self
    

    def compute(self):
        return self.v[0] / self.v[1]

class MSEMetric(SimpleMetric):
    """ Mean square error metric.
    """
    def update(self, pred, y):
        self.v[0] += ((pred - y)**2).sum()
        self.v[1] += y.numel()


class BinaryAccuracyMetric(SimpleMetric):
    def update(self, pred, y):
        self.v[0] += (pred == y).sum()
        self.v[1] += y.numel()



class BinaryRecallMetric(SimpleMetric):
    def update(self, pred, y):
        pred = pred.bool()
        y = y.bool()

        tp = (pred & y).sum()
        fn = ((~pred) & y).sum()

        self.v[0] += tp
        self.v[1] += tp + fn



class BinaryPrecisionMetric(SimpleMetric):
    def update(self, pred, y):
        pred = pred.bool()
        y = y.bool()

        tp = (pred & y).sum()
        fp = (pred & (~y)).sum()

        self.v[0] += tp
        self.v[1] += tp + fp


class BinarySpecificityMetric(SimpleMetric):
    def update(self, pred, y):
        pred = pred.bool()
        y = y.bool()

        tn = ((~pred) & (~y)).sum()
        fp = (pred & (~y)).sum()

        self.v[0] += tn
        self.v[1] += tn + fp



class BinaryNPVMetric(SimpleMetric):
    def update(self, pred, y):
        pred = pred.bool()
        y = y.bool()

        tn = ((~pred) & (~y)).sum()
        fn = ((~pred) & y).sum()

        self.v[0] += tn
        self.v[1] += tn + fn

        
class BinaryBalancedAccuracyMetric(SimpleMetric):
    def update(self, pred, y):
        pred = pred.bool()
        y = y.bool()

        tp = (pred & y).sum()
        fn = ((~pred) & y).sum()
        tn = ((~pred) & (~y)).sum()
        fp = (pred & (~y)).sum()

        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        self.v[0] += (recall + specificity) / 2
        self.v[1] += 1


class MulticlassAccuracyMetric(SimpleMetric):
    def update(self, pred, y):
        self.v[0] += (pred == y).sum()
        self.v[1] += y.numel()

class MulticlassF1Metric(SimpleMetric):
    def __init__(self, num_classes):
        super().__init__(torch.zeros(3, num_classes))  
        # rows: TP, FP, FN

    def reset(self):
        self.v.zero_()

    def update(self, pred, y):
        for c in range(self.v.shape[1]):
            p = pred == c
            t = y == c

            self.v[0, c] += (p & t).sum()          # TP
            self.v[1, c] += (p & (~t)).sum()       # FP
            self.v[2, c] += ((~p) & t).sum()       # FN

    def compute(self):
        tp, fp, fn = self.v
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        return f1.mean()

class MulticlassPrecisionMetric(SimpleMetric):
    def __init__(self, num_classes):
        super().__init__(torch.zeros(2, num_classes))  # TP, FP

    def reset(self):
        self.v.zero_()

    def update(self, pred, y):
        num_classes = self.v.shape[1]

        for c in range(num_classes):
            p = pred == c
            t = y == c

            self.v[0, c] += (p & t).sum()        # TP
            self.v[1, c] += (p & (~t)).sum()     # FP

    def compute(self):
        tp, fp = self.v
        precision = tp / (tp + fp + 1e-8)
        return precision.mean()  

    
# class CategoricalAccuracyMetric(SimpleMetric):      # OBS! Copilot har cookat ihop denna
#     """
#     Accuracy for single-label (categorical) classification.
#     Assumes:
#     - pred: (B, num_classes) logits
#     - y: (B,) integer class labels
#     """
#     def update(self, pred, y):
#         # Convert logits to predicted class indices
#         pred_classes = pred.argmax(dim=1)

#         self.v[0] += (pred_classes == y).sum()
#         self.v[1] += y.numel()
