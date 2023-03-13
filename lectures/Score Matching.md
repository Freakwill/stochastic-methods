# Score Matching
[TOC]

Notations:
- $p$: true distr.
- $p(x|\theta)$: model/hypothisis distr.
- $\psi(x)=\frac{\partial\log p(x)}{\partial x}=\frac{\nabla_x p(x)}{p(x)}$: *score function* of $p(x)$
- $E(x)$: energy function, $\psi=\nabla E$

score function does not depend on partition function, i.e. $\psi(x)=\frac{\nabla f(x)}{f(x)}$ where $p(x)=\frac{f(x)}{Z}$

## Assumption
1. $\mathcal{X}=\R^n$ (open subset of $\R^n$)
2. $p$ is smooth enough

## Definition
### *Explicit score matching(ESM)*

$$
J_{ESM}(\theta) := \frac{1}{2}E\|\psi(x|\theta)-\frac{\partial \log p(x)}{\partial x}\|^2
$$

*Remark.* ESM=the diff of the score of the model distr. and the that of true distr. 

*Fact.* $J(\theta) = 0 \iff  p(x|\theta)=p(x)$ where $p(x)>0$


### *Implicit score matching(ISM)*

$$
J_{ISM}(\theta) := \mathbb{E} ( \frac{1}{2}\|\psi(x|\theta)\|^2-\sum_j\frac{\partial \psi_j(x|\theta)}{\partial x_j})\\
= \mathbb{E}(\frac{1}{2}\|\nabla E\|^2-\Delta E)
$$

*Lemma.* $\mathbb{E}(\psi(x|\theta)\cdot\frac{\partial \log p(x)}{\partial x})=\int_x\psi(x|\theta)\cdot\frac{\partial  p(x)}{\partial x}=-\int p(x)\sum_j\frac{\partial \psi_j(x|\theta)}{\partial x_j}$ ($=\Delta E(x|\theta)$ where $p(x)=\frac{e^E}{Z}$)

*Proof.* $\nabla ^* =-\mathrm{div}$ on the sp. of smooth functions; $\nabla ^* \nabla =\Delta$.

*Fact.*  $J_{ESM} = J_{ISM} + const.$; $\hat{\theta}_{ESM}=\hat{\theta}_{ISM}$


## SM Algo
task:
$$
\min J_{ISM}(\theta)\approx \sum_i ( \frac{1}{2}\|\psi(x_i|\theta)\|^2+\sum_j\frac{\partial \psi^{(j)}(x_i|\theta)}{\partial x^{(j)}})\\
\approx \sum_i ( \frac{1}{2}\|\nabla E(x_i|\theta)\|^2+\Delta E (x_i|\theta))
$$
by GD


## SM-DAE


## Aplications

### MV Gaussian
energy function: $E(x|\mu,\Sigma)=-(x-\mu)^T\Sigma(x-\mu)$

$$
J_{ISM} = -\mathrm{tr}\Sigma + \frac{1}{2N}\sum_i (x_i-\mu)^T\Sigma^2(x_i-\mu)
$$

Result: SME = MLE

### ICA



---
*References*

1. Aapo Hyvarinen. Estimation of Non-Normalized Statistical Models by Score Matching.
2. Pascal Vincent. A Connection Between Score Matching
and Denoising Autoencoders, 2010.
