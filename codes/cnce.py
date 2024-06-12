#!/usr/bin/env python

"""
CNCE for multivariate normal distr.
"""

import numpy as np
import numpy.linalg as LA
from scipy.stats import multivariate_normal
from scipy.special import expit
from scipy.optimize import minimize


# configuration
Td = 1000
nu = 10
log_nu = np.log(nu)
Tn = Td * nu

# parameters/distribution and sampling
Sigma = np.array([[1.0, 0.5], [0.5, 2]])
Lambda = LA.inv(Sigma)
pd = multivariate_normal([0, 0], Sigma)  # data distr.


def neg_log_phi(X, theta):
    L = np.array([[theta[0], 0],[theta[1], theta[2]]])
    # x @ LL.T @ x == ||Lx||_2^2
    return np.array([np.sum(np.dot(L, x)**2) for x in X])/2

# dataset
Xd = pd.rvs(size=Td)

# noise dataset
Xn = np.empty((Tn, 2))
for k, x in enumerate(Xd):
    pn = multivariate_normal(x) # noise distr. p(y|x) = N(x, 1)
    Xn[np.arange(nu)* Td + k] = pn.rvs(size=nu)

Xd = np.tile(Xd, (nu, 1))


def fit(Xd, Xn):
    # do NCE with Xd, Xn

    def loss(theta):
        G = neg_log_phi(Xd, theta) - neg_log_phi(Xn, theta)
        return np.mean(np.log(1+np.exp(G)))

    theta0 = np.array([1, 0, 1])
    theta = minimize(loss, theta0).x
    return np.array([[theta[0], 0],[theta[1], theta[2]]])


Lambda_ = fit(Xd, Xn)


def prob(X):
    from sklearn.neural_network import MLPClassifier

    h = MLPClassifier(hidden_layer_sizes=(8,))
    Xc = np.block([[Xd, Xn],[Xn, Xd]])
    Nd, _ = Xd.shape
    Nn, _ = Xn.shape
    y = np.concatenate((np.ones(Nd), np.zeros(Nn)))
    h.fit(Xc, y)

    N = X.shape[0]
    # Xn_ = np.empty((N, 2))
    # for k, x in enumerate(X):
    #     pn = multivariate_normal(x) # noise distr.
    #     Xn_[k] = pn.rvs()
    X_ = np.tile(X.mean(axis=0), (N, 1))
    hx = h.predict_proba(np.hstack((X, X_)))[:,0]
    return hx / (1-hx)

 
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

x1 = np.linspace(-5, 5, 100)  
x2 = np.linspace(-5, 5, 100)  
X2, X1 = np.meshgrid(x2, x1)  
X = np.column_stack((X1.ravel(), X2.ravel()))

PX = pd.pdf(X)
cp = ax1.contourf(X1, X2, PX.reshape((100,100)), 20, cmap='viridis')
ax1.set_title('model distribution')

Simga_ = LA.inv(Lambda_ @ Lambda_.T)
print(Simga_)
pd_ = multivariate_normal([0, 0], Simga_)
PX = pd_.pdf(X)
cp = ax2.contourf(X1, X2, PX.reshape((100,100)), 20, cmap='viridis')
ax2.set_title('estimated distribution by SGD-NCE')

PX = prob(X)
cp = ax3.contourf(X1, X2, PX.reshape((100,100)), 20, cmap='viridis')
ax3.set_title('by Classifier')

plt.show()
