#!/usr/bin/env python

"""
Importance sampling
"""

import numpy as np
import scipy.stats

def importance_sampling(func, proposal_dist, p=None, sample_size=1000, bias=False):
    """Generate samples from the proposal distribution
    
    Args:
        func (function): the integrated function
        proposal_dist (scipy distribution): the proposal distribution for IS
        p (None, optional): a pdf
        sample_size (int, optional): size of sample
        bias (bool, optional): bias or unbias
    
    Returns:
        TYPE: Description
    """

    samples = proposal_dist.rvs(size=sample_size)

    weights = 1 / proposal_dist.pdf(samples) 
    if p is not None:
        weights *= p(samples)
        
    if bias:
        weights /= np.sum(weights)
        return np.sum(weights * func(samples))
    else:
        return np.mean(weights * func(samples))


if __name__ == '__main__':

    import scipy.integrate as spi
    
    def f(x):
        return np.sin(x*10) * (0<x) * (x<np.pi)

    # Define the proposal distribution (e.g., Gaussian)
    proposal_dist = scipy.stats.norm(loc=0, scale=1)

    # Perform importance sampling
    import time
    time1 = time.perf_counter()
    estimated_integral = importance_sampling(f, proposal_dist, sample_size=2000)
    time2 = time.perf_counter()
    integral, error = spi.quad(f, 0, np.pi)
    time3 = time.perf_counter()

    print(f"Importance Sampling: {estimated_integral}, Time: {time2-time1}")
    print(f"Numerical Integration: {integral}, Time: {time3-time2}")

