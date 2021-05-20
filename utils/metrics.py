import jiwer
import Levenshtein as Lev
import torch
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

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def __call__(self, target, preds):
        pass

    def compute(self):
        return self.correct / self.total

class WER(AverageMeter):
    def __call__(self, target, preds):
        self.correct += jiwer.wer(target, preds)
        self.total += 1

class CER(AverageMeter):
    def __call__(self, target, preds):
        self.correct += sum([cer(rf, pd) for (rf, pd) in zip(target, preds)])
        self.total += len(target)