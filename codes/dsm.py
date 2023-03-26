#!/usr/bin/env python

"""
DSM: SM-DAE

*Reference*
Pascal Vincent. A Connection Between Score Matching and Denoising Autoencoders, 2010.
"""

import numpy as np
import random
from scipy.stats import norm
from scipy.special import expit, logit, softmax


def DJ(x, y, W, c):
    """diff J wrt W and c
    Dwst = pt ds + (Wds - sumsk(Wdk pk)) ps xt; Wdk=sumj(Wkjdj)
    """
    p = softmax(np.dot(W, x.T), axis=0)
    y_ = np.dot(p.T, W) + c
    d = y_ - y
    Wd = np.dot(W, d.T)
    return (np.dot(p, d) + np.dot((Wd - np.sum(Wd * p, axis=0)) * p, x)) / x.shape[0], np.mean(d, axis=0)


# def DJ0(x, y, W, c):
#     """diff J wrt W and c
#     Dwst = pt ds + (Wkjdj ps) xt - (Wkjdj pk ps) xt

#     Deprecated!
#     """
#     p = softmax(np.dot(W, x))
#     y_ = np.dot(p, W) + c
#     d = y_ - y
#     Wd = np.dot(W, d)
#     return (np.outer(p, d) + np.outer((Wd - np.sum(Wd * p)) * p, x))/x.shape[0], np.mean(d)


def _fit(X, Y, max_iter=500, init_W=None, init_c=0, learning_rate=0.001, n_batches=10):
    """SDM: SM-DAE
    """

    n_samples, n_features = X.shape

    batch_size, rem = divmod(n_samples, n_batches)

    W = init_W
    c = init_c
    for _ in range(max_iter):
        for k in range(n_batches):
            # Deprecated
            # DW=Dc =0
            # for x, t in zip(X[k*batch_size: (k+1)*batch_size], Y[k*batch_size: (k+1)*batch_size]):
            #     _DW, _Dc = DJ0(x, t, W, c)
            #     DW += _DW
            #     Dc += _Dc
            # if rem >0:
            #     for x, t in zip(X[-rem:], Y[-rem:]):
            #         _DW, _Dc = DJ0(x, t, W, c)
            #         DW += _DW
            #         Dc += _Dc

            x, t = X[k*batch_size: (k+1)*batch_size], Y[k*batch_size: (k+1)*batch_size]
            DW, Dc = DJ(x, t, W, c)
            if rem >0:
                x, t = X[-rem:], Y[-rem:]
                _DW, _Dc = DJ(x, t, W, c)
                DW += _DW
                Dc += _Dc
            W -= learning_rate * DW
            c -= learning_rate * Dc
        if np.all(np.abs(DW)<1e-10) and np.all(np.abs(Dc)<1e-10):
            break
    return W, c


from sklearn.base import TransformerMixin
class DSM(TransformerMixin):
    # Score Matching for Denoising
    def __init__(self, dim_latent, sigma=1, max_iter=500, learning_rate=0.001, add_constant=False):
        self.dim_latent = dim_latent
        self.max_iter = max_iter
        self.sigma = sigma
        self.add_constant = add_constant
        self.learning_rate = learning_rate
        if add_constant: raise ValueError('add_constant=True is deprecated!')

    def init(self, X):
        
        self.weights_ = np.random.randn(self.dim_latent, self.n_features_)
        self.bias_ = 0

    def fit(self, X, n_y_per_x=5):
        _, self.n_features_ = X.shape

        n_y_per_x = 5 # number of y instances for each x
        X = np.repeat(X, n_y_per_x, axis=0)
        if self.add_constant:
            X = np.insert(X, self.n_features_, 1, axis=1)
        Xn = X + np.random.randn(*X.shape) * self.sigma

        self.init(X)
        self.weights_, self.bias_ = _fit(Xn, X, self.max_iter, self.weights_, self.bias_, self.learning_rate)

    def transform(self, X):
        # x -> W^T softmax(Wx)
        W = self.weights_
        if self.add_constant:
            X = np.insert(X, self.n_features_, 1, axis=1)
        Y = np.dot(softmax(np.dot(X, W.T), axis=1), W) + self.bias_
        if self.add_constant:
            Y = np.delete(Y, -1, axis=1)
        return Y

    denoise = transform


from sklearn import datasets
digists = datasets.load_digits()
X, y = digists.data, digists.target
X = X[y<3]
y = y[y<3]

# preprocess
X = logit((X + 0.5)/17)

# training
sigma = 4
dsm = DSM(dim_latent=25, sigma=sigma, max_iter=1000)
dsm.fit(X, n_y_per_x=20)

# denoising
y_sel = (0, 1, 2, 0, 1, 2)
X_sel = np.array([random.choice(X[y==k]) for k in y_sel])
Xn = X_sel + np.random.rand(*X_sel.shape) * sigma
Xr = dsm.denoise(Xn)

size = 8, 8

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots(len(y_sel), 3, sharex=True, sharey=True)
for k in range(len(y_sel)):
    for j in range(3):
        ax[k, j].set_yticks([])
        ax[k, j].set_xticks([])

# postprocess
X = expit(X)*17-0.5
Xn = expit(Xn)*17-0.5
Xr = expit(Xr)*17-0.5

for axk, y, x, xn, xr in zip(ax, y_sel, X_sel, Xn, Xr):
    axk[0].imshow(x.reshape(size))
    axk[1].imshow(xn.reshape(size))
    axk[2].imshow(xr.reshape(size))
    axk[0].set_ylabel(str(y))

ax[0, 0].set_title('Original Digits')
ax[0, 1].set_title('Noised Digits')
ax[0, 2].set_title('Denoised Digits')

fig.suptitle("demo of DSM/SM-DAE")
plt.show()

