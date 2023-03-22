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


def _fit(X, Y, max_iter=500):
    """SDM: SM-DAE
    """

    eta = 0.01
    n_samples, n_features = X.shape

    dim_latent = 5

    n_batches = 10
    batch_size, rem = divmod(n_samples, n_batches)


    W = np.zeros((dim_latent, n_features))

    for _ in range(max_iter):
        for k in range(n_batches):
            D = 0
            for x, t in zip(X[k*batch_size: (k+1)*batch_size], Y[k*batch_size: (k+1)*batch_size]):
                D += DJ(x, t, W)
            if rem >0:
                for x, t in zip(X[-rem:], Y[-rem:]):
                    D += DJ(x, t, W)
            W -= eta * D
        if np.all(np.abs(D)<0.00001):
            break
    return W


from sklearn import datasets
digists = datasets.load_digits()
X_train, y_train = digists.data, digists.target

X = X_train[(y_train==2) | (y_train==0)]
# preprocess
X = logit((X + 0.5)/17)

sigma = 5

n_y_per_x = 5 # number of y instances for each x
X = np.repeat(X, n_y_per_x, axis=0)
Y = X + np.random.randn(*X.shape) * sigma

W = _fit(X, Y, max_iter=150)

# reconstruction
x = X[10]
xn = x + np.random.rand(x.size) * sigma
xr = F(xn, W)

# postprocess
xr = expit(xr)*17-0.5

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots(1, 2)
size = 8, 8
ax[0].imshow(xn.reshape(size))
ax[0].set_title('A noised digit')
ax[1].imshow(xr.reshape(size))
ax[1].set_title('A denoised digit')
for _ in ax: _.set_axis_off()
fig.suptitle("demo of DSM/SM-DAE")
plt.show()

