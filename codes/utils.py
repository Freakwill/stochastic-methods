#!/usr/bin/env python


import numpy as np
from scipy.special import logit, expit


def normalize(a, *args, **kwargs):
    return a / np.sum(a, *args, **kwargs)


def scaling_logit(x, lb=0, ub=255):
    x_ = (x-lb + 0.5) / (ub-lb+0.75)
    return logit(x)


def scaling_logit(x, lb=0, ub=255):
    x_ = expit(x)
    return x_ * (ub-lb+0.75) + lb - 0.5