import jiwer
import Levenshtein as Lev
import torch
from torchmetrics import Metric
from statistics import mean

def cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2) / len(s1)

class WER(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, target, preds):
        self.correct += jiwer.wer(target, preds)
        self.total += 1

    def compute(self):
        return (self.correct / self.total).item()

    def clear(self):
        self.correct = torch.tensor(0.0)
        self.total = torch.tensor(0.0)

class CER(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, target, preds):
        self.correct += sum([cer(rf, pd) for (rf, pd) in zip(target, preds)])
        self.total += len(target)

    def compute(self):
        return (self.correct / self.total).item()

    def clear(self):
        self.correct = torch.tensor(0.0)
        self.total = torch.tensor(0.0)