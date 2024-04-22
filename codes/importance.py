#!/usr/bin/env python

"""
Importance Sampling
"""


import numpy as np
from scipy.stats import norm, rv_discrete


def target_pdf(x, lambda_val=1.0):
    return lambda_val * np.exp(-lambda_val * np.abs(x+1)) * 0.5 + lambda_val * np.exp(-lambda_val * np.abs(x-1)) * 0.5


def _proposal(mu=0, sigma=0.2):
    return norm(mu, sigma)


def importance_simulation(n_samples=1000, lambda_val=1.0, mu=0, sigma=1):

    x_ = _proposal(mu, sigma).rvs(size=n_samples)
    weights = target_pdf(x_) / _proposal(mu, sigma).pdf(x_)
    weights /= weights.sum()
    custom_rv = rv_discrete(name='custom_dist', values=(np.arange(n_samples), weights))  

    return np.array([x_[k] for k in custom_rv.rvs(size=n_samples)]) 


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    samples = importance_simulation(n_samples=5000)
    plt.hist(samples, bins=20, density=True, alpha=0.6, color='g')

    x_values = np.linspace(-4, 4, 300)
    plt.plot(x_values, target_pdf(x_values), 'r', alpha=0.5, label='Target Distribution')

    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Importance Simulation')
    # plt.savefig('../lectures/accept-reject.png')
    plt.show()
    