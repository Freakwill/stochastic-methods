# Score Matching advanced

[TOC]

## Concrete SM

*Def. Concrete SM*
$$
c_p(x,\mathcal{N}):=\{\frac{p(x_k)-p(x)}{p(x)}=\frac{p(x_k)}{p(x)}-1, x_k\in \mathcal{N}\} : \mathcal{X}\to\R^{|\mathcal{N}|}
$$
where $\mathcal{N}$ is the neighbor system of $x$.

*Def. Concrete SM(with kernel)*

$$
c_p(x,K):=\{\frac{\int K(x_k,x)p(x) \mathrm{d}x}{p(x)}\} : \mathcal{X}\to\R^{N}
$$
where $N$ is the size of the sample.

$K^+c(x) = \sum_k c_k K(x,x_k):\R^{N}\to \mathcal{F}$ as the adjoint of $K$

## Concrete Score Matching Objective

$$
L_{CSM}:=E_x\|c_p(x) -c_q(x)\|^2\\
=E_x(c_q^2+2c_q)\cdot 1-2\sum_x\sum_{x_k:\mathcal{N}(x)}p(x_k)c_q(x)_k\\
\sim \sum_kq(x_k)^2E_x(\frac{1}{q(x)^2})-2\sum_x\sum_{x_k:\mathcal{N}(x)}p(x_k)c_q(x)_k
$$


*Homework*
define CSM objective for Concrete SM(with kernel) and design the corresponding algorithm.