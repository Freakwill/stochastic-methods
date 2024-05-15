#!/usr/bin/env python

"""
NCE for multivariate normal distr.

*Reference*
M U. Gutmann, A Hyvarinen. NCE of Unnormalized Statistical Models, with Applications to NaturalImage Statistics
"""

import numpy as np
import numpy.linalg as LA
from scipy.stats import multivariate_normal
from scipy.special import expit


# configuration
Td = 1000
nu = 5
log_nu = np.log(nu)
Tn = Td * nu

# parameters/distribution and sampling
Sigma = np.array([[1.0, 0.5], [0.5, 2]])
Lambda =LA.inv(Sigma)
c = 0.5 * np.log(LA.det(Lambda)) - np.log(2*np.pi)
pd = multivariate_normal([0, 0], Sigma)  # data distr.

Xd = pd.rvs(size=Td)

# MLE
cov = np.cov(Xd, rowvar=0)
per = LA.inv(cov)

pn = multivariate_normal([0, 0], cov) # noise distr.
Xn = pn.rvs(size=Tn)


def G(x, L, c):
    # log-likelihood ratio
    Lx = np.dot(L, x)
    return -0.5* np.dot(x, Lx) + c - pn.logpdf(x)

C = (2*np.ones((2,2))-0.5*np.eye(2))

def g(x, L, c):
    # derivation of G wrt L
    return - np.outer(x, x) * C


def DJ(X, Y, L, c):
    # derivation of J wrt L, c
    
    H = np.array([1 - expit(G(x, L, c) - log_nu) for x in X])
    K = np.array([expit(G(y, L, c) - log_nu) for y in Y])

    DJ_L1 = np.mean([h * g(x, L, c) for h, x in zip(H, X)], axis=0)
    DJ_L2 = np.mean([k * g(y, L, c) for k, y in zip(K, Y)], axis=0)

    DJ_L = DJ_L1 - DJ_L2 * nu

    return DJ_L, np.mean(H) - np.mean(K) * nu


n_iter = 500

def fit(Xd, Xn):
    # do NCE with Xd, Xn
    Lambda_ = np.eye(2)
    c_ = 0
    errors = {'Lambda':[],
    'c':[]
    }
    alpha = 0.01
    for k in range(n_iter):
        DJ_L, DJ_c = DJ(Xd, Xn, Lambda_, c)
        Lambda_ += alpha * DJ_L
        c_ += alpha * DJ_c
        alpha *= 0.99
        errors['Lambda'].append(LA.norm(Lambda_-Lambda, 'fro'))
        errors['c'].append(abs(c_-c))
    return errors

errors = fit(Xd, Xn)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(errors['Lambda'])
e = LA.norm(Lambda-per)
ax.plot([1, n_iter], [e, e], '--')
ax.set_title('estimates of Lambda')
ax.set_xlabel('iteration')
ax.legend(('NCE', 'MLE'))
ax = fig.add_subplot(122)
ax.plot(errors['c'])
ax.set_title('estimates of c')
ax.set_xlabel('iteration')
fig.suptitle('NCE test')
plt.show()
