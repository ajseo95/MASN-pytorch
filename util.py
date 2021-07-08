# --------------------------------------------------------
# This code is modified from Jumpin2's repository.
# https://github.com/Jumpin2/HGA
# --------------------------------------------------------
""" Common utilities. """

# Logging
# =======

import logging
import os, os.path
from colorlog import ColoredFormatter
import torch

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    #    datefmt='%H:%M:%S.%f',
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white,bold',
        'INFOV': 'cyan,bold',
        'WARNING': 'yellow',
        'ERROR': 'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%')
ch.setFormatter(formatter)

log = logging.getLogger('videocap')
log.setLevel(logging.DEBUG)
log.handlers = []  # No duplicated handlers
log.propagate = False  # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')


def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)


logging.Logger.infov = _infov


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class StrToBytes:

    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def get_accuracy(logits, targets):
    correct = torch.sum(logits.eq(targets)).float()
    return correct * 100.0 / targets.size(0)


class nvidia_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data = [d.cuda(non_blocking=True) for d in self.next_data]
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_data is None:
            raise StopIteration
        next_data = self.next_data
        self.preload()
        return next_data

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self