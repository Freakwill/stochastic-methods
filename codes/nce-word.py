#!/usr/bin/env python

"""
CD for RBM

version = 1.0

*Reference*
Geoffrey E. Hinton. Training Products of Experts by Minimizing Contrastive Divergence.
Miguel A. Carreira-Perpinnaan, Geoffrey E. Hinton. On Contrastive Divergence Learning.
"""

import numpy as np
import numpy.linalg as LA
import random
from scipy.stats import rv_discrete, bernoulli
from scipy.special import expit, softmax
from sklearn.base import TransformerMixin
from utils import *


def r(x, nu=1):
    return 1/(1 + nu * np.exp(-x))


def similarity(w, c, wv):
    return np.sum(word_vec(w, wv) * context_vec(c, wv), axis=-1)

def word_vec(w, wv):
    return wv[w]

def context_vec(c, wv):
    if c.ndim==2:
        return np.mean(np.array([[word_vec(w, wv) for w in ci] for ci in c]), axis=1)
    return np.mean([word_vec(w, wv) for w in c], axis=0)


def logpm(w, c, wv, a=0):
    return similarity(w, c, wv) + a


def logpn(w, c, alpha=3/4):
    global p_w, p_c, context
    if c.ndim == 2:
        return np.log(p_w[w]) + np.array([np.log(p_c[np.all(context==ci, axis=1)][0]) for ci in c])*alpha
    return np.log(p_w[w]) + np.log(p_c[np.all(context==c, axis=1)][0])*alpha


def G(w, c, wv, a=0):
    return logpm(w, c, wv, a) - logpn(w, c)


def h(w, c, wv, a=0, nu=1):
    # for NCE
    return r(G(w, c, wv, a), nu)

# def h(w, c, wv, a=0, nu=None):
#     # for Negative sampling
#     return expit(similarity(w, c, wv))


def DJ(w, c, wv, a, nu, hwc):
    # diff G wrt wv (score function) weighted by hwc
    hwc = 1-h(w, c, wv, a, nu)
    D = np.zeros_like(wv)
    v = np.unique(w)
    D[v] = [np.dot(hwc[w==vi], context_vec(c, wv)[w==vi]) for vi in v]
    v = np.unique(c.ravel())
    c_v = np.mean([[cj==vi for cj in c] for vi in v], axis=-1)
    D[v] += np.dot(c_v * hwc, wv[w])
    return D / len(w)


def _fit(W, C, Wn, Cn, nu=2):
    global vocab, n_vocab
    Dwv = 0
    Da = 0
    n_batches = 20
    n_samples = W.shape[0]
    batch_size = n_samples // n_batches
    noise_batch_size = batch_size * nu
    dim_embedding = 10
    wv = np.random.randn(*(n_vocab, dim_embedding))
    a = 0
    max_iter = 300
    learning_rate = 0.005
    for _ in range(max_iter):
        for k in range(n_batches):
            w, c = W[k*batch_size:(k+1)*batch_size], C[k*batch_size:(k+1)*batch_size]
            wn, cn = Wn[k*noise_batch_size:(k+1)*noise_batch_size], Cn[k*noise_batch_size:(k+1)*noise_batch_size]
            Dwv = DJ(w, c, wv, a, nu, 1-h(w, c, wv, a, nu)) + nu * DJ(wn, cn, wv, a, nu, h(w, c, wv, a, nu))
            w, c = W[k*batch_size:(k+1)*batch_size], C[k*batch_size:(k+1)*batch_size]
            wn, cn = Wn[k*noise_batch_size:(k+1)*noise_batch_size], Cn[k*noise_batch_size:(k+1)*noise_batch_size]
            Da = 1 - np.mean(h(w, c, wv, a, nu)) + nu * np.mean(h(wn, cn, wv, a, nu))
            wv += learning_rate * Dwv
            a += learning_rate * Da
    return wv, a


def fit(d, min_count=None, smoothing_coef=1):
    # d: corpus, List[str]
    global vocab, n_vocab, p_w, p_c, context
    nu = 2
    alpha = 3/4
    vocab = np.unique(d).tolist()
    n_vocab = len(vocab)
    size_window = 2
    w_c = np.array([(vocab.index(dk), *(vocab.index(d[k+j]) for j in range(-size_window, size_window+1) if j!=0))
        for k, dk in enumerate(d[size_window:-size_window])])
    w, c = w_c[:, 0], w_c[:, 1:]
    count_w = np.array([np.sum(w==i) for i in range(n_vocab)])
    p_w = normalize(count_w+ smoothing_coef)
    context = np.unique(c, axis=0)
    n_context = len(context)
    count_c = np.array([np.sum(np.all(context==c, axis=1)) for c in context])
    p_c = normalize((count_c + smoothing_coef * n_context) ** alpha)
    n_samples = w.shape[0]
    wn = rv_discrete(values=(np.arange(n_vocab), p_w)).rvs(size=n_samples*nu)
    cn = context[rv_discrete(values=(np.arange(n_context), p_c)).rvs(size=n_samples*nu)]
    wv, _ = _fit(w, c, wn, cn, nu=2)

    from sklearn.manifold import TSNE
    
    z = TSNE(n_components=2,init="pca").fit_transform(wv)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_w = 50
    z = z[np.argsort(count_w)[-n_w:]] # coordinates of first n_w words
    ax.scatter(z[:,0], z[:,1])
    for w, (x, y) in zip(vocab, z):
        ax.text(x, y, w)
    ax.set_title("word vectors")
    plt.show()



import pathlib
import re

d = pathlib.Path('/Users/William/Folders/mycorpus/Archive/hegel.txt').read_text()
d = [w.strip("\n“”.1234567890:()").lower() for w in re.split(r'\W', d) if w.strip("\n“”.1234567890:()") not in {'', ' ', '\n'}]

fit(d)
