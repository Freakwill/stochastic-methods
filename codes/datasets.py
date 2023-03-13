#!/usr/bin/env python


import pathlib

import numpy as np
from PIL import Image

from typing import List


def get_data(image, m, n, w=None, h=None, channel=0):
    # set of 2d (3d) arrays
    data = np.asarray(image, dtype=np.float64)
    M, N = data.shape[:2]
    h = h or M // m
    w = w or N // n
    if channel is None:
        return np.array([data[i*h:(i+1)*h, j*w:(j+1)*w, :] for i in range(m) for j in range(n)])
    else:
        return np.array([data[i*h:(i+1)*h, j*w:(j+1)*w, channel] for i in range(m) for j in range(n)])

def load_hanzi(channel=0, ravel=False):
    im = Image.open('../data/hanzi.jpeg')
    im = im.crop((7,7,im.size[0]-7, im.size[1]-7))
    m, n = 14, 20
    width, height = im.size[0] // n, im.size[1] // m
    data = get_data(im, m, n, height, width, channel)
    if ravel:
        return np.array([d.ravel() for d in data if np.mean(d<100)>0.01])
    return np.array([d for d in data if np.mean(d<100)>0.01])


def load_corpus(path) -> List[str]:
    if isinstance(path, str):
        path = pathlib.Path(path)
    return [f.read_text() for f in path.iterdir() if f.suffix == '.txt']

