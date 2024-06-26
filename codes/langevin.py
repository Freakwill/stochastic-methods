#!/usr/bin/env python3

"""
Langevin Dynamics is a stochastic simulation method often used in molecular dynamics simulations
 to model the thermal motion of particles in a system.
"""

import random

import numpy as np
import numpy.linalg as LA

from scipy.stats import multivariate_normal


class LangevinMC:

    def __init__(self, sigma=1, n_steps=1000, init_state=None):
        self.sigma = sigma
        self.n_steps = n_steps
        self.init_state = init_state

    def _grad_log_prob(self, x):
        raise NotImplemented

    def _log_prob(self, x):
        raise NotImplemented

    def _prob(self, x):
        raise NotImplemented

    def _like_rate(self, x, y):
        return self._prob(y) / self._prob(x)

    def sample(self, init_state=None, size=None, sigma=None, burn_in=1):

        size = size or self.n_steps
        sigma = sigma or self.sigma
        if init_state is None:
            state = self.init_state
        else:
            state = init_state

        assert size > 0 and sigma >0

        if state is None:
            state = np.zeros(n_features)

        states = np.empty((size, n_features))

        # Langevin Dynamics simulation
        i= -burn_in
        sigma_square = self.sigma **2
        while True:
            # Compute stochastic force (random force from the heat bath)
            force = self.sigma * np.random.normal(size=n_features)
            x = state
            gradx = self._grad_log_prob(x)
            v = sigma_square/2 * gradx + force
            y = x + v

            grady = self._grad_log_prob(y)
            m = (gradx + grady) / 2

            rho = self._like_rate(x, y) * np.exp(np.dot(x - y + sigma_square/4 * (gradx - grady), m))
            # Store data
            if random.random() < rho:
                state = y
                if i>=0:
                    states[i]=state
                i += 1
                if i == size:
                    break
        return states


class AnnealingLangevinMC(LangevinMC):

    def sample(self, init_state=None, size=1000, n_epoches=100, sigam_init=1):

        size = size or self.n_steps
        state = self.init_state
        if init_state is None:
            state = self.init_state
        else:
            state = init_state

        states = np.empty((size, n_features))

        mc_size = size // n_epoches

        for k in range(n_epoches):
            sigma = sigam_init * 0.9**k
            states[k*mc_size:(k+1)*mc_size] = super().sample(init_state=state, size=mc_size, sigma=sigma)
            state = states[(k+1)*mc_size-1]

        return states


if __name__ == '__main__':
    
    # mean = np.array([0, 0])  
    # cov = np.array([[1, 0.5], [0.5, 2]])
    # precise = LA.inv(cov)

    # class MyLangevinMC(LangevinMC):

    #     def _log_prob(self, x):
    #         return multivariate_normal(mean, cov).logpdf(x)

    #     def _grad_log_prob(self, x):
    #         return np.dot(precise, mean-x)

    mean1 = np.array([-1, -2])  
    cov1 = np.array([[1, 0.5], [0.5, 2]])
    precise1 = LA.inv(cov1)

    mean2 = np.array([3, 3])  
    cov2 = np.array([[1, 0], [0, 0.5]])
    precise2 = LA.inv(cov2)

    pi = 0.8

    class MyLangevinMC2(LangevinMC):

        def _prob(self, x):
            return multivariate_normal(mean1, cov1).pdf(x) * pi + multivariate_normal(mean2, cov2).pdf(x) * (1-pi)

        def _log_prob(self, x):
            return np.log(self._prob(x))

        def _grad_log_prob(self, x):
            p1 = multivariate_normal(mean1, cov1).pdf(x) * pi
            p2 = multivariate_normal(mean2, cov2).pdf(x) * (1-pi)
            return (np.dot(precise1, mean1-x) * p1 + np.dot(precise2, mean2-x) * p2) / (p1 + p2+0.0001)

    class MyAnnealingLangevinMC(AnnealingLangevinMC, MyLangevinMC2):
        pass

    n_features=2
    ld = MyLangevinMC2(init_state=[0, 0])
    states = ld.sample(size=2000)

    # ld = MyAnnealingLangevinMC(init_state=[0, 0])
    # states = ld.sample(size=2000)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(*states.T, alpha=0.6)
    x = np.linspace(-8, 8, 100)  
    y = np.linspace(-8, 8, 100)  
    X, Y = np.meshgrid(x, y)

    Z = ld._log_prob(np.column_stack([np.ravel(X), np.ravel(Y)])).reshape((100, 100))
    ax.contour(X, Y, Z, 30, cmap='viridis', alpha=0.8)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    plt.show()
