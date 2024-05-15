#!/usr/bin/env python

"""
Score Matching Algo for ICA

Ref.
Estimation of Non-Normalized Statistical Models
by Score Matching
"""

import numpy as np

from scipy.optimize import minimize

import numpy.linalg as LA


sqrt3 = np.sqrt(3)


def g(s):
    return - np.pi/3 * np.tanh(np.pi/(2*sqrt3) * s)


def dg(s):
    return - np.pi ** 2 /(6*sqrt3) * (1 - np.tanh(np.pi/(2*sqrt3) * s)**2)


def J(W, X):
    # sample version of ISM for ICA
    W2 = np.sum(W**2, axis=0)
    Z = X @ W
    Y = np.dot(dg(Z), W2[:,None]) + np.sum(np.dot(g(Z), W.T)**2, axis=1) * 0.5
    return Y.mean()


def minimize_matrix(objective, X0, *args, **kwargs):
    shape = X0.shape
    x0 = np.ravel(X0)
    def _objective(x):
        X = np.reshape(x, shape)
        return objective(X) + 0.01*np.sum(x ** 2) + LA.norm(X.T @ X - np.eye(shape[1]))
    x = minimize(_objective, x0, *args, **kwargs)['x']
    return x.reshape(shape)


t = np.linspace(0, 20, 1000)
s = np.row_stack((np.floor(t/3)*2-4, 5*np.sin(t*2), np.random.randn(2, 1000)/4))
s = s.T

N, q = s.shape
p = 5

V = np.random.randn(q, p)
Q, R = LA.qr(V.T)

X = s @ Q.T

W0 = np.random.randn(p, q)

def objective(W):
    return J(W, X)

W = minimize_matrix(objective, W0, method='BFGS', options={'maxiter': 1000})
Z = X @ W

import matplotlib.pyplot as plt

fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

for k in range(q):
    ax0.plot(t, s[:,k], alpha=0.75)

for k in range(p):
    ax1.plot(t, X[:,k], alpha=0.75)

for k in range(q):
    ax2.plot(t, Z[:,k], alpha=0.75)

plt.show()
