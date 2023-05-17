#!/usr/bin/env python

"""
CD for FVBM

Does not work well!

*Reference*
Geoffrey E. Hinton. Training Products of Experts by Minimizing Contrastive Divergence.
Miguel A. Carreira-Perpinnaan, Geoffrey E. Hinton. On Contrastive Divergence Learning.
"""

import numpy as np
import numpy.linalg as LA
import random
from scipy.stats import rv_discrete, multivariate_normal, bernoulli
from scipy.special import expit
from sklearn.model_selection import train_test_split


def _energy(x, W):
    # energy function of VBM
    # W: symmetric, diag W = 0 (Optional)
    return 0.5 * np.dot(np.dot(x, W), x)

def grad_energy(x, W):
    DE = np.outer(x, x)
    for i in range(len(x)):
        DE[i, i] = 0
    return DE


def random_walk(start, energy, proposed=None, mc_iter=1):
    """Random Walk (a special case of H-M algo.)

    proposed: x -> distr, proposed/instrumental probability density
    mc_iter: int, iterations of MCMC
    """
    for _ in range(mc_iter):
        alternative = proposed(start)
        d = energy(alternative) - energy(start)
        if d >= 0:
            start = alternative
        else:
            accept_proba = np.exp(d)
            if random.random() < accept_proba:
                start = alternative
    return start


def proposed(x, p=0.1):
    m = bernoulli(p).rvs(n_features)
    x = (1-x) * m + x * (1-m)
    x[-1] = 1
    return x

def gibbs(x, W, mc_iter=10):
    # Gibbs sampling for VBM
    x = x.copy()
    for _ in range(mc_iter):
        for k in range(len(x)):
            x[k] = random.random() < expit(np.dot(W[k], x))
        x[-1] = 1
    return x


from sklearn import datasets

digists = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digists.data, digists.target, test_size=0.05)

X_train = X_train[y_train==0]
X = (X_train>7).astype(np.int_)

X = np.insert(X, -1, 1, axis=1)

n_samples, n_features = X.shape


def _fit(X, max_iter=500, mc_iter=1, persistent=False):
    """CD k(==1) algo.

    mc_iter: iterations of mcmc
    persistent: for persistent CD
    """
    tol = 1e-6
    eta = 0.02
    n_samples, n_features = X.shape

    W = np.zeros((n_features,n_features))

    for _ in range(max_iter):
        positive = np.mean([grad_energy(x, W) for x in X], axis=0)
        # X1 = [random_walk(x, energy=lambda _: _energy(_, W), proposed=proposed, mc_iter=mc_iter) for x in X]
        X1 = [gibbs(x, W, mc_iter=mc_iter) for x in X]
        negative = np.mean([grad_energy(x, W) for x in X1], axis=0)
        DW = positive - negative
        if persistent:
            X = X1
        if LA.norm(DW) < tol: break
        eta *= 0.96
        W += eta * DW

        for i in range(n_features):
            W[i,i] = 0
    return W

W = _fit(X, max_iter=250, mc_iter=2, persistent=False)

# x = np.random.randint(2, size=(n_features,))
x = proposed(X[3], p=0.12)
# xx = random_walk(x, energy=lambda _: _energy(_, W), proposed=proposed, mc_iter=100)
xx = gibbs(x, W, mc_iter=20)

def to_matrix(x):
    return np.delete(x, -1).reshape((8, 8))

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots(1, 3)
ax[0].imshow(to_matrix(X[3]))
ax[0].set_title('A real digit')
ax[1].imshow(to_matrix(x))
ax[1].set_title('A noised digit')
ax[2].imshow(to_matrix(xx))
ax[2].set_title('A generated digit')
fig.suptitle("Digit Generator (Test of CD-VBM)")
plt.show()

