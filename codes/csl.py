#!/usr/bin/env python

"""
Contrastive supervised learning (but dose not work)
"""

import numpy as np

from sklearn import datasets
from scipy.optimize import *
from sklearn.model_selection import train_test_split

from scipy.special import softmax


def loss(x):
    return np.log(1+np.exp(-x))


data = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1)

N, p = X_train.shape
classes = np.unique(y_train)

def _fit(X, y):
    # l(k,y) = l({yk - yk', k' != k})

    def f(W):
        return np.sum([np.sum(loss(np.prod([X[y==k] @ (W[:,k] - W[:,kk]) for kk in classes if kk !=k], axis=0))) for k in classes]) + 0.1*np.sum(np.abs(W))

    from utils import minimize_matrix
    W0 = np.random.random(size=(p, len(classes)))
    return minimize_matrix(f, W0, method='BFGS', options={'maxiter':1000})

def _predict(X, W):
    return np.argmax(X @ W, axis=1)

def _predict_proba(X, W):
    return softmax(X @ W, axis=1)

W = _fit(X_train, y_train)
print((W>0.001) * W)
y_ = _predict(X_test, W)
print(np.mean(y_test == y_), y_test, y_)

print(_predict_proba(X_test, W))
