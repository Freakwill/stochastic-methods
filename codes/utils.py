#!/usr/bin/env python


import numpy as np

def normalize(a, *args, **kwargs):
    return a / np.sum(a, *args, **kwargs)