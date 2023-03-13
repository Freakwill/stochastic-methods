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
pn = multivariate_normal([0, 0], [[1, 0], [0, 1]]) # noise distr.

Xd = pd.rvs(size=Td)
Xn = pn.rvs(size=Tn)


# MLE
cov = np.cov(Xd, rowvar=0)
per = LA.inv(cov)

def G(x, L, c):
    # log-likelihood ratio
    Lx = np.dot(L, x)
    return -0.5* np.dot(x, Lx) + c - pn.logpdf(x)

C = (np.ones((2,2))-0.5*np.eye(2))
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


def fit(Xd, Xn):
    # do NCE with Xd, Xn
    Lambda_ = np.eye(2)
    c_ = 0
    errors = {'Lambda':[],
    'c':[]
    }
    for k in range(1000):
        DJ_L, DJ_c = DJ(Xd, Xn, L, c)
        Lambda_ += 0.1 * 0.99**k * DJ_L
        c_ += 0.1 * 0.99**k * DJ_c
        errors['Lambda'].append(LA.norm(Lambda_-Lambda, 'fro'))
        errors['c'].append(abs(c_-c))
    return errors

errors = fit(Xd, Xn)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(errors['Lambda'])
e = LA.norm(Lambda-per)
ax.plot([1, 1000], [e, e], '--')
ax.set_title('estimates of Lambda')
ax.set_xlabel('iteration')
ax.legend(('NCE', 'MLE'))
ax = fig.add_subplot(122)
ax.plot(errors['c'])
ax.set_title('estimates of c')
ax.set_xlabel('iteration')
fig.suptitle('NCE test')
plt.show()
