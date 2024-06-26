#!/usr/bin/env nim

# Importance Sampling by Nimble

import random
import sequtils

import alea

type
  Func = proc (x: float): float  # float -> float
  ProposalDist = proc (): float
  Pdf = proc (x: float): float

proc sum(arr: seq[float]): float =
  var result = 0.0
  for item in arr:
    result += item
  return result

proc mean(arr: seq[float]): float =
  result = arr.sum() / float(arr.len)


proc importance_sampling(function: Func, proposal_dist: ProposalDist, p: Pdf = nil, sample_size: int = 1000, bias: bool = false): float =
  
  #[
  Importance sampling to integrate the function `function` using `proposal_dist`.
  If `bias` is `true`, the proposal distribution density `p` is required.
  ]#

  var
    samples: seq[float] = @[]
    weights: seq[float] = @[]
    w: float

  for _ in 0..<sample_size:
    let sample = proposal_dist()
    # samples.add(sample)
    if not bias:
      assert p != nil
      w = function(sample) / p(sample)
    else:
      w = function(sample)

  weights.add(w)

  if not bias:
    return weights.mean
  else:
    return weights.sum / sample_size.float

# Example usage:

# Define the function to integrate.
proc myFunc(x: float): float =
  return x * x  # Example: x^2

# Define the proposal distribution (uniform distribution in this case).
proc proposalDist(): float =
  return rand(1.0)

# Define the probability density function of the proposal distribution.
proc proposalPdf(x: float): float =
  return 1.0  # Uniform distribution on [0, 1] has constant density 1.

# Perform importance sampling.
let result = importance_sampling(myFunc, proposalDist, proposalPdf, sample_size = 10000)
echo "Estimated integral (bias corrected): ", result

# Perform importance sampling without bias correction (using uniform distribution).
let result_bias = importance_sampling(myFunc, proposalDist, sample_size = 10000, bias = true)

echo "Estimated integral (biased): ", result_bias

