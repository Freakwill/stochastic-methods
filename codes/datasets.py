#!/usr/bin/env python


import pathlib

import numpy as np
from PIL import Image

from typing import List


def get_data(image, m, n, w=None, h=None, channel=0) -> np.ndarray:
    # set of 2d (3d) arrays
    data = np.asarray(image, dtype=np.float64)
    M, N = data.shape[:2]
    h = h or M // m
    w = w or N // n
    if data.ndim == 3:
        if channel is None:
            return np.array([data[i*w:(i+1)*w, j*h:(j+1)*h, :] for i in range(m) for j in range(n)])
        else:
            return np.array([data[i*w:(i+1)*w, j*h:(j+1)*h, channel] for i in range(m) for j in range(n)])
    else:
        return np.array([data[i*w:(i+1)*w, j*h:(j+1)*h] for i in range(m) for j in range(n)])

# def load_hanzi(channel=0, ravel=False):
#     im = Image.open('hanzi.jpeg')
#     im = im.crop((7,7,im.size[0]-7, im.size[1]-7))
#     m, n = 14, 20
#     width, height = im.size[0] // n, im.size[1] // m
#     data = get_data(im, m, n, height, width, channel)
#     if ravel:
#         return np.array([d.ravel() for d in data if np.mean(d<100)>0.01])
#     return np.array([d for d in data if np.mean(d<100)>0.01])

def load_images(path, bm_size, channel=None, ravel=False):
    """Get small images from a large image
    The large image is regarded as a block matrix.
    
    Args:
        path (TYPE): The path of the image
        bm_size (TYPE): the size of the block matrix of the images
        channel (None, optional): the channel of the image
        ravel (bool, optional): flatten the arrays
    
    Returns:
        arrays: the arrays of images
    """

    im = Image.open(path)
    m, n = bm_size
    width, height = im.size[0] // n, im.size[1] // m
    data = get_data(im, m, n, height, width, channel)
    if ravel:
        return np.array([d.ravel() for d in data])
    return data


def load_corpus(path) -> List[str]:
    if isinstance(path, str):
        path = pathlib.Path(path)
    return [f.read_text() for f in path.iterdir() if f.suffix == '.txt']


def fashion_images(with_labels=False, *args, **kwargs):
    if with_labels:
        images = load_images('data/fashion.png', bm_size=(30,30), *args, **kwargs)
        labels = np.repeat(np.arange(10),90)
        return images, labels
    else:
        return load_images('data/fashion.png', bm_size=(30,30), *args, **kwargs)


def cartoon_images(ravel=False):
    return load_images('data/cartoon.jpg', bm_size=(10,10), ravel=ravel)


def load_pictures(path, exts={'.jpg', 'jpeg', '.png'}, size=None, asarray=False, channel=None):
    if isinstance(path, str):
        path = pathlib.Path(path)
    labels = []
    arrays = []
    for folder in path.iterdir():
        if folder.is_dir():
            array_ = [Image.open(pic) for pic in folder.iterdir() if pic.suffix in exts]
            labels.extend((folder.stem,) * len(array_))
            arrays.extend(array_)
    if size:
        arrays = [a.resize(size) for a in arrays]
    if asarray:
        arrays = np.asarray([np.asarray(a, np.float64) for a in arrays])
        if channel is not None:
            arrays = [a[:,:,channel] for a in arrays]
    
    return arrays, np.asarray(labels)



