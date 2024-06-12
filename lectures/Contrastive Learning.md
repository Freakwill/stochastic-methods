# Contrastive  Learning
[TOC]

## conditional log-likelihood
genrative classification model: $p(x,y), y=1,\cdots, K, x\in\mathcal{X}$

CLL:
$$
l(p)= N E_{x,y} \ln p(y|x)\approx\sum_i \ln p(y_i|x_i)\\
=\sum_k\sum_{y_i=k} \ln p(k|x_i)\\
\approx\sum_kN_k E_{x\sim p(x|k)}\ln p(k|x)\\
=E_{x_1,\cdots, x_K}\sum_kN_k\ln p(k|x_k)\\
$$

For special $x_k^*\sim p(x|k)$
$$
l(p)\propto E_{x}\ln h(x), ~~  p(k|x_k^*) \propto h(x_1^*)
$$

approx to expectation risk of the loss on sample sp $\mathcal{X}^K$:
$$
L(p;\{x_1,\cdots, x_K\})=\sum_kN_k\ln p(k|x_k)
$$

<!-- *Example* NCE: $p(y=1|x)=h(x)$; CNCE: $p(y=1|x,x')=h(x,x')$; NEG: $p(y=1|x)=p(x)$ -->

Contrastive loss ($p_n\sim 1$):
$$
L\propto - E_{x_1,\cdots,x_K}\ln\frac{e^{f(x_1)}}{\sum_k{e^{f(x_k)}}}
$$
(CLL of $p(x_1,x_2,\cdots,x_K|k)=p(x_k)\prod_{l\neq k }p_n(x_l|x_k)$)

Contrastive loss ($p_n(x)$ independent form):
$$
L\propto - E_{x_1,\cdots,x_K}\ln\frac{1}{1+\sum_{k\neq 1}{e^{f(x_k)/f(x_1)}}}
$$
(CLL of $p(x_1,x_2,\cdots,x_K|k)=p(x_k)\prod_{l\neq k }p_n(x_l|x_k)$)

equiv. to the classifier for dataset:
```
...
X^K, Y
X1=X,...,Xk,...,xK, 1  #
...
X1,...,Xk=X,...,XK, k  # 
...
```

<!-- $$
l(p)= E_{x,y} f(x,y)\approx \frac{1}{N}\sum_i f(x_i,y_i)\\
=\frac{1}{N}\sum_k\sum_{y_i=k} f_k(x_i)\\
\approx\sum_k\pi_k E_{x\sim p(x|k)}f_k(x)\\
=E_{x_1,\cdots, x_K}\sum_k\pi_kf_k(x_k)
$$ -->

## InfoNCE

### CPC

CPC: $z_t=f(x_t), c_t=g(z_{\leq t})$

**InfoNCE loss**： NCE loss of $p(x_{t+k},c_t)$ with proposal/noise distr. $p(x_{t+k})$
$$
L:=-E_X\log\frac{f(x_{t+k},c_t)}{\sum_jf(x_j,c_t)}, f(x_{t+k},c_t)=\frac{p(x_{t+k}|c_t)}{p(x_{t+k})}\\
X=\{x_1,\cdots,x_K\},x_j\sim p(x_{t+k}|c_t), x_{-j}\sim p(x_{t+k})
$$

$\iff$ $K$-classifier: $P(j|X,c_t)\sim f(x_j,c_t)$

sample:

$\mathcal{X}^K$ | $\{1,\cdots, K\}$
---|---
xj, x_{-j} | j


*Fact* $I(x_{t+k},c_t)\geq \log K-L$; minimizing the **InfoNCE loss** $L$   maximizes a l.b. on MI.

### Lvm
model: $P(x,z)$

loss:
$$
L:=-E_X\log\frac{f(x_{k},z)}{\sum_jf(x_j,z)}, f(x,z):=\frac{p(x|z)}{p(x)}\\
X=\{x_1,\cdots,x_K\},x_j\sim p(x_{k}|z), x_{-j}\sim p(x_{k})
$$