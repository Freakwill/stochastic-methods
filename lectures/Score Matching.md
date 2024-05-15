# Score Matching
[TOC]

Notations:
- $p$: true distr.
- $p(x|\theta)$: model/hypothisis distr.
- $\psi(x)=\frac{\partial\log p(x)}{\partial x}=\frac{\nabla_x p(x)}{p(x)}$: *score function* of $p(x)$
- $E(x)$: energy function, $p\sim e^E$

score function does not depend on partition function, i.e. $\psi(x)=\frac{\nabla f(x)}{f(x)}=\nabla E$ where $p(x)=\frac{f(x)}{Z}$

## Assumption
1. $\mathcal{X}=\R^n$ (open subset of $\R^n$)
2. $p$ is smooth enough

## Concetps

*Lemma.* $E_p\langle f, \nabla \log p\rangle=\int f\nabla p$ ($=-E_p \mathrm{div} f$, if $f, p$ are smooth enough)

### *Explicit score matching(ESM)*

*Definition*
$$
J_{ESM}(\theta) := \frac{1}{2}\mathbb{E}_{x\sim p}\|\psi(x|\theta)-\frac{\partial \log p(x)}{\partial x}\|^2
$$

*Remark.* ESM is the **Fisher div.** denoted by $D_F(p,p_\theta)$.

*Fact.* $J(\theta) = 0 \iff  p(x|\theta)=p(x)$ where $p(x)>0$


### *Implicit score matching(ISM)*

$$
J_{ISM}(\theta) := \mathbb{E} ( \frac{1}{2}\|\psi(x|\theta)\|^2+\sum_j\frac{\partial \psi_j(x|\theta)}{\partial x_j})\\
= \mathbb{E}(\frac{1}{2}\|\nabla E\|^2+\Delta E)
$$

*Lemma.* $\mathbb{E}(\psi(x|\theta)\cdot\frac{\partial \log p(x)}{\partial x})=\int_x\psi(x|\theta)\cdot\frac{\partial  p(x)}{\partial x}=-\int p(x)\sum_j\frac{\partial \psi_j(x|\theta)}{\partial x_j}$ ($=\Delta E(x|\theta)$ where $p(x)=\frac{e^E}{Z}$)

*Proof.* $\nabla ^* =-\mathrm{div}$ on the sp. of smooth functions; $\nabla ^* \nabla =\Delta$.

*Fact.*  $J_{ESM} = J_{ISM} + const.$; $\hat{\theta}_{ESM}=\hat{\theta}_{ISM}$


## SM Algo

task/ERM:
$$
\min J_{ISM}(\theta)\approx \sum_i ( \frac{1}{2}\|\psi(x_i|\theta)\|^2+\sum_j\frac{\partial \psi^{(j)}(x_i|\theta)}{\partial x^{(j)}})\\
\approx \sum_i ( \frac{1}{2}\|\nabla E(x_i|\theta)\|^2+\Delta E (x_i|\theta)).
$$

by GD,

$\partial_\theta J = \sum_i(\nabla  E\cdot \nabla \partial_\theta E +\Delta\partial_\theta E)$


## SM-DAE

*Fact.* Given joint distr. $p(x,z)$ ==>

1. 
  $$
  E_{x\sim p(x)}\langle\psi, \nabla \log p(x)\rangle =
  E_{(x,z)\sim p(x,z)}\langle\psi, \nabla \log p(x|z)\rangle
  $$

2. 
  $$
  \frac{1}{2}E_{(x,z)\sim p(x,z)}\|\psi(x;\theta)-\nabla \log p(x|z)\|^2\sim J_{ESM}
  $$
  that is, if we define $J_{CSM}(z):=\frac{1}{2}E_{x\sim p(x|z)}\|\psi(x;\theta)-\nabla \log p(x|z)\|^2$ (conditional SM), then $J_{ESM}\sim E_z J_{CSM}(z)$


**Denoising model**:
- $p(z)$: distr. of data
- $p(x|z)$: $x$ is the noised version of data $z$

**Denoising  score matching (DSM)** :
$$
J_{DSM}(p(x)):=\frac{1}{2}E\|\psi(x;\theta)-\nabla \log p(x|z)\|^2\\
\sim J_{ESM}(p(x))
$$

Gaussian DSM: $p(x|z)\sim N(z,\sigma^2)$ ==>
 $\nabla \log p(x|z)=\frac{z-x}{\sigma^2}$
$$
J_{DSM}(p(x))\sim \frac{1}{\sigma^4}\mathbb{E}\|\nabla E(x;\theta)-(z-x)\|^2
$$
where $\frac{E}{\sigma^2}$ is the energy of model.


## Aplications

### MV Gaussian
energy function: $E(x|\mu,\Sigma)=-\frac{1}{2}(x-\mu)^T\Sigma(x-\mu)$

$$
J_{ISM} = -\mathrm{tr}\Sigma + \frac{1}{2N}\sum_i (x_i-\mu)^T\Sigma^2(x_i-\mu)
$$

Result: SME = MLE

### ICA

Energy of Model: $E(x)=\sum_k G(W_kx)$

==> 

$$
G = g(XW), G'=g'(XW)\\
J(W) = \frac{1}{N}1_N(G' \{W_k.^2\} + (GW^T).^21_p/2)
$$

### Laten Var. Models

Model: $\frac{e^{E(x;z)}}{Z(\theta)}$ ($x$ is continous)

Let $F(x)=\log \sum_z e^{E(x,z)}$, as the energy of $p(x)$

*Fact.* $\nabla F = \mathbb{E}_{P(z|x)}\nabla E$

Give out $J_{ESM},J_{ISM}$

$\Delta F=?$ as your homework. 📢

### Mixed Models

Model: $\sum_k p_k\frac{e^{E(x;\theta_k,\theta)}}{Z(\theta)}$

what is the SM of the model?

### DAE

structure:
`x(Image + noise) -> NN -> Image`

$$
E(W,b,c) = \frac{c\cdot x-\frac{1}{2}\|x\|^2+\sum_j\mathrm{softplus}(W_j\cdot x+b_j)}{\sigma^2}
$$
==>
$$
J_{DSM}=\frac{1}{\sigma^4}E\|W^T\mathrm{softmax}(Wx+b)+c-z\|^2\\
\approx \frac{1}{\sigma^4}\sum_{i}\sum_{j}\|W^T\mathrm{softmax}(Wx_i^{j}+b)+c-z_i\|^2, x_{i}^{j}\sim p(x|z_i)
$$

that is, train NN $x\mapsto W^T\mathrm{softmax}(Wx+b)+c$ by data $\{x_{ij},z_i\}$


For the code see `codes/dsm.py`

## Relation with other algo/models

## SM ~ ML
**Theorem** Under regularization conditions.
$$
\frac{d}{dt}D_{KL}(p_t\|q_t) =-\frac{1}{2}D_F(p_t\|q_t)\\
\frac{d}{dt}D_{KL}(p_t\|q_t)|_{t=0} =-\frac{1}{2}D_F(p\|q)
$$
where $p_t$ is the convolution of $p$ and $N(0,t)$. Specially,
$$
\frac{d}{dt}H(p_t) =-\frac{1}{2}J(p_t)
$$

*Interpretation* While ML aims to minimize the KL divergence
directly, SM seeks to eliminate its derivative inthe scale space at $t = 0$. SM looks for stability, where the optimal parameter $θ$ leads to least changes in the KL divergence between the two models when a smallamount of noise ispresent in the training data, while ML pursues extremity of the KL divergence. It is known that MLE is sensitive to noisy training data, which may give rise to many false extreme values, yet SM may be more robust to small perturbation in training data.

### SM ~ CD ~ PL

*Fact*
$$
j(\theta) = \nabla_\theta J_{SM}
$$
where $j(\theta)\approx \nabla_\theta J_{CD}(θ', θ, x_t),\theta'\approx\theta$ under LD that $\sigma^2\to 0$. Hence SM == deterministic CD

*Fact*
$$
\nabla_\theta J_{PL} \approx E_x \nabla \log p(x|\theta) -E_iE_{x_i|x_{\bar{i}}}\nabla \log p(x|\theta)
$$


## Generalized SM(GSM)

### General form
Generalized Fisher div.

$$
D_{F}(p\|q)=\int p(\frac{Lp}{p}-\frac{Lq}{q})^2\\
\sim \int p((\frac{Lq}{q})^2- 2L^*(\frac{Lq}{q}))
$$

Generalized Fisher info.$H_{F}(p)=\int p(\frac{Lp}{p})^2$
and its cross version $H_{F}(p,q)=\int p(\frac{Lp}{q})^2$

*Completeness*: $\frac{Lp}{p}=\frac{Lq}{q}\iff p=q$

Assume that $L:F\to F^d$ is linear and complete, where $F$ is a family of PDFs.

*Example 1* $L=\nabla$ in standard SM

*Example 2* marginalization operator $Mp:=\{\int_{x_j}p(x),j=1,\cdots,d\}$, we have
$$
GSM_M:=E\sum_j\int (1-\frac{1}{q(x_j|x_{\hat{j}})})^2 dx_j 
$$

### extension of SM
#### discrete SM --- Ratio Matching
$L(p)(x) = \{p(x_{-j}),j=1,\cdots,d\}$ where $\mathcal{X}=\{-1,1\}^{d}$

#### non-negative SM

#### piecewise-continuous data
$\psi(x)\Rightarrow \psi(x)\phi(x)$?

---
*Exercise*

1. Can SM be applied to train Boltzmann machine?
2. SM for the model with enery $E(x)=x^TWx + \phi(x)$ on $\R^n$, constrained BMs， where $\phi$ is a reasonable function.
3. compute $\nabla J_{DSM}$, prompt: $\frac{\partial p_k}{\partial W}=p_k(\delta_{k} -p) \circ x$ where $p_k=\mathrm{softmax}(Wx)_k$
4. construct your own $L$ for GSM

*References*

1. Aapo Hyvarinen. Estimation of Non-Normalized Statistical Models by Score Matching.
2. Pascal Vincent. A Connection Between Score Matching and Denoising Autoencoders, 2010.
3. Siwei Lyu. Interpretation and Generalization of Score Matching, 2009.
4. Aapo Hyvärinen, Some extensions of score matching, 2007.

