#!/usr/bin/env python

"""
Accept-Reject Algo.
"""


import numpy as np
import matplotlib.pyplot as plt


def target_pdf(x, lambda_val=1.0):
    return lambda_val * np.exp(-lambda_val * np.abs(x))


def proposal_pdf(x, mu=0, sigma=1):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def max_constant_c(lambda_val=1.0, mu=0, sigma=1):
    x_values = np.linspace(-1, 1, 100)
    target_values = target_pdf(x_values, lambda_val)
    proposal_values = proposal_pdf(x_values, mu, sigma)
    return np.max(target_values / proposal_values)


def accept_reject_simulation(n_samples, lambda_val=1.0, mu=0, sigma=1):
    c = max_constant_c(lambda_val, mu, sigma)
    samples = []
    while len(samples) < n_samples:
        # sampling from proposal pdf
        x = np.random.normal(mu, sigma)
        u = np.random.uniform(0, 1)
        
        if u * c * proposal_pdf(x, mu, sigma) < target_pdf(x, lambda_val) and -1<x<1:
            samples.append(x)
    return np.array(samples)

if __name__ == '__main__':
    
    n_samples = 500
    samples = accept_reject_simulation(n_samples)
    plt.hist(samples, bins=15, density=True, alpha=0.6, color='g')

    x_values = np.linspace(-1, 1, 200)
    plt.plot(x_values, target_pdf(x_values), 'r', alpha=0.5, label='Target Distribution')

    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Accept-Reject Random Simulation')
    plt.legend()
    plt.savefig('../lectures/accept-reject.png')
    plt.show()
    