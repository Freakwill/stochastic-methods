#!/usr/bin/env python

"""
DSM: SM-DAE

*Reference*
Pascal Vincent. A Connection Between Score Matching and Denoising Autoencoders
"""

import numpy as np
from scipy.stats import norm
from scipy.special import expit, logit, softmax


def F(x, W):
    # x -> W^T softmax(Wx)
    return np.dot(softmax(np.dot(W, x)), W)

def DJ(x, y, W):
    # diff J
    p = softmax(np.dot(W, x))
    y_ = np.dot(p, W)
    d = y_ - y
    return W * np.outer(p, x * d) + np.outer(p, (1- x * y_)*d)


def _fit(X, Y, max_iter=500, init_W=None):
    """SDM: SM-DAE
    """

    eta = 0.01
    n_samples, n_features = X.shape

    n_batches = 10
    batch_size, rem = divmod(n_samples, n_batches)


    W = init_W

    for _ in range(max_iter):
        for k in range(n_batches):
            D = 0
            for x, t in zip(X[k*batch_size: (k+1)*batch_size], Y[k*batch_size: (k+1)*batch_size]):
                D += DJ(x, t, W)
            if rem >0:
                for x, t in zip(X[-rem:], Y[-rem:]):
                    D += DJ(x, t, W)
            W -= eta * D
        if np.all(np.abs(D)<1e-10):
            break
    return W


from sklearn.base import TransformerMixin
class DSM(TransformerMixin):
    # Score Matching for Denoising
    def __init__(self, dim_latent, sigma=1, max_iter=500):
        self.dim_latent = dim_latent
        self.max_iter = max_iter
        self.sigma = sigma

    def fit(self, X, n_y_per_x=5):
        n_y_per_x = 5 # number of y instances for each x
        X = np.repeat(X, n_y_per_x, axis=0)
        Y = X + np.random.randn(*X.shape) * self.sigma
        W = np.zeros((self.dim_latent, X.shape[1]))
        self.weights_ = _fit(X, Y, self.max_iter, W)

    def transform(self, X):
        return np.asarray([F(x, self.weights_) for x in X])

    denoise = transform


from sklearn import datasets
digists = datasets.load_digits()
X, y = digists.data, digists.target
X=X[y<4]
y=y[y<4]

# preprocess
X = logit((X + 0.5)/17)

sigma = 5
dsm = DSM(dim_latent=15, sigma=sigma, max_iter=30)
dsm.fit(X, n_y_per_x=10)


# reconstruction

size = 8, 8

import matplotlib.pyplot as plt
fig = plt.figure()
y_sel = (0, 1, 2, 3)
X_sel = np.array([X[y==k][10] for k in y_sel])
Xn = X_sel + np.random.rand(*X_sel.shape) * sigma
Xr = dsm.denoise(Xn)
# postprocess
Xr = expit(Xr)*17-0.5
ax = fig.subplots(len(y_sel), 3, sharex=True, sharey=True)
for axk, x, xn, xr in zip(ax, X_sel, Xn, Xr):
    axk[0].imshow(x.reshape(size))
    axk[1].imshow(xn.reshape(size))
    axk[2].imshow(xr.reshape(size))

ax[0, 0].set_title('Original Digits')
ax[0, 1].set_title('Noised Digits')
ax[0, 2].set_title('Denoised Digits')
for k in range(len(y_sel)):
    for j in range(3):
        ax[k,j].set_axis_off()
fig.suptitle("demo of DSM/SM-DAE")
plt.show()

