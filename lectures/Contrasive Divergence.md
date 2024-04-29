# Contrastive  Divergence(CD) Algo.
[TOC]

## Abstract
an approximate Maximum-Likelihood learning algorithm  proposed by G. Hinton(2002) for energy-based models or Markov networks


CD is a general MCMC gradient ascent learning algorithm particularly well suited to learning Product of Experts (PoE)
and energy-based (Gibbs distributions, etc.) model parameters. 


Notations:

- $\langle\cdot\rangle_{p}$: expection under distr. $p$, i.e.$\mathbb{E}_p$
- $\langle\cdot\rangle_{X}$: mean on sample $X$

## Model
### Distribution/partition function
model distr.:
$$
p(x|\theta)=\frac{f(x;\theta)}{Z(\theta)}
$$
where $Z(\cdot)$ is the *partition function*.

Problem: $Z$ dose not have a closed form/explict computation/ is intractable.

*Example* RBM; joint distr. of Markov Net; PoE model

We have
*Fact.*
$$
\frac{\partial \ln p(x|\theta)}{\partial\theta}=  \frac{\partial \ln f(x;\theta)}{\partial\theta}-\int p(x|\theta) \frac{\partial \ln f(x;\theta)}{\partial\theta}\\
=  \frac{\partial \ln f(x;\theta)}{\partial\theta}-E_{x\sim p(x|\theta)} \frac{\partial \ln f(x;\theta)}{\partial\theta}
$$

by the following fact

*Fact.* $\partial_z \ln p(z) = \partial_z \ln \int p(x,z) dx = \int p(x|z)\partial_z \ln p(x,z)dx$

### Approximation of Maximum-Likelihood

log-likelihood: $l(\theta):=\sum_i\ln p(x_i;\theta)=\sum_i \ln f(x_i;\theta)-N\ln Z(\theta)$

Mean log-likelihood: 
$$
\bar{l}(\theta):=\frac{1}{N}\sum_i\ln p(x_i;\theta)=\ln Z(\theta)-\frac{1}{N}\sum_i \ln f(x_i;\theta)
$$

$$
\frac{\partial \bar{l}(\theta)}{\partial\theta}= \frac{1}{N}\sum_i \frac{\partial \ln f(x_i;\theta)}{\partial\theta}-\int p(x;\theta) \frac{\partial \ln f(x;\theta)}{\partial\theta}\\
\approx\langle \frac{\partial \ln f(x;\theta)}{\partial\theta} \rangle_{X^{0}}-\langle \frac{\partial \ln f(x;\theta)}{\partial\theta} \rangle_{X^{\infty}}\\
\approx \langle \frac{\partial \ln f(x;\theta)}{\partial\theta} \rangle_{X^{0}}-\langle \frac{\partial \ln f(x;\theta)}{\partial\theta} \rangle_{X^{k}}  ~~(\text{CD-k update})
$$
where $X^{k}$ (fantasy samples) represents the training data transformed using $k$ cycles of MCMC, such that $X^{0} ≡ X, X^{\infty} \sim p(x|\theta)$.

*k-CD algo.*
$\Delta \theta \leftarrow \eta\langle \frac{\partial \ln f(x;\theta)}{\partial\theta} \rangle_{X^{0}}-\langle \frac{\partial \ln f(x;\theta)}{\partial\theta} \rangle_{X^{k}}$

*Remark.* Let $H(\theta)=H(p^0,p_\theta)$,
$$
\frac{\partial H(\theta)}{\partial\theta}=E_{x\sim p^0}\frac{\partial \ln f(x;\theta)}{\partial\theta}-E_{x\sim p_\theta}\frac{\partial \ln f(x;\theta)}{\partial\theta}\\
\approx E_{x\sim p^0}\frac{\partial \ln f(x;\theta)}{\partial\theta}-E_{x\sim p^k}\frac{\partial \ln f(x;\theta)}{\partial\theta}, p^k = p^0K^k
$$
where $K$ is the transite kernel of MCMC.

If $f(x;\theta)$ has form $e^{E(x;\theta)}$, then
$$
\frac{\partial \bar{l}(\theta)}{\partial\theta}
\approx\langle \frac{\partial E(x;\theta)}{\partial\theta} \rangle_{X^{0}}-\langle \frac{\partial E(x;\theta)}{\partial\theta} \rangle_{X^{\infty}}\\
\approx \langle \frac{\partial E(x;\theta)}{\partial\theta} \rangle_{X^{0}} - \langle \frac{\partial E(x;\theta)}{\partial\theta} \rangle_{X^{k}}
$$


*Remark.* Samples cannot be drawn directly from $p(x; \Theta)$ for the partition function, but we can use many cycles of  MCMC sampling to transform our training data (drawn from the target distribution) into data drawn from the proposed distribution Draw samples from $p$ by MCMC. The transformation only involves calculating likelihood ratio $\frac{f(x';\theta)}{f(x;\theta)}$ (without partition funciton).

#### Proposed distr. of MCMC
$q(x|y)p(y)=q(y|x)p(x)$
==> $\frac{q(y|x)}{q(x|y)}=e^{E(y)-E(x)}$

a possible proposed distr.: $q(y|x)\sim r(x,y)e^{\frac{1}{2}(E(y)-E(x))}$, where $r(x,y)=r(y,x)$.

### Contrastive Divergence
*Definition* **CD objective**:
$$
CD_k:=D_{KL}(p_d\|p)-D_{KL}(p^k\|p); \\ 
CD:=CD_1
$$

- $p_d$: data/target distr.(population)
- $p^k$: fabulation distr. ($k$-step of MCMC from $p_d$)
- $p=p^\infty$: model distr.

*Fact.* Assume $\frac{D_{KL}(p^k\|p)}{\partial p^k}$ is small,
$$
\frac{\partial CD_k}{\partial\theta} = \text{CD-update} + \frac{\partial D_{KL}(p^k\|p)}{\partial p^k}\frac{\partial p^k}{\partial\theta}\\
 \approx \text{CD-update}
$$

*Hinton CD algo.*
Input $X^0$
Return $\theta$

0. initialize $\theta$
1. sample $X^1$ by MCMC from $X^0$ (with the stationary distr. $p(x|\theta)$)
2. compute
$$
\Delta\theta = \eta (\langle \frac{\partial \ln f(x;\theta)}{\partial\theta} \rangle_{X^{0}}- \langle \frac{\partial \ln f(x;\theta)}{\partial\theta} \rangle_{X^{1}})
$$
and update $\theta$
1. repeat 1-2 until it converges

*Hinton k-CD algo.*
sample $X^k$ by MCMC from $X^0$ (after $k$ steps)

## Discrete cases
$G_{\theta,x}:=\frac{\partial E(x;\theta)}{\partial\theta}:|\Theta|\times |\mathcal{X}|$

$\frac{\partial \bar{l}(\theta)}{\partial\theta} = Gp_d-Gp_\theta\approx Gp_d - Gp^k, p_{.}:\Delta^{|\mathcal{X}|-1}$

## Convergence

### fixed points of iteration (discrete case)
$Gp_0=Gp^\infty=Gp^k$ where $p^k=T^kp_0$

### updating form
update after sampling $x\sim p^0, x^k\sim p^k$:

$$
\Delta\theta \propto \ \frac{\partial E(x^{0};\theta)}{\partial\theta} - \ \frac{\partial E(x^k;\theta)}{\partial\theta}
$$

Expected update:
$$
E\Delta\theta \propto \langle \frac{\partial E(x;\theta)}{\partial\theta} \rangle_{p^{0}(x)}- \langle \frac{\partial E(x;\theta)}{\partial\theta} \rangle_{\int K(x^k|x)p^0(x)}
$$
where $K(\cdot|\cdot)$ is the kenerl of MCMC.

### Bias
CD-bias expansion:
$$
\frac{\partial \log p(x^0;\theta)}{\partial\theta}=\frac{\partial E(x^0;\theta)}{\partial\theta}+\mathbb{E}(\frac{\partial E(x^k;\theta)}{\partial\theta}|x^0)+\mathbb{E}(\frac{\partial \log P(x^k|\theta)}{\partial\theta}|x^0)
$$

where $\mathbb{E}(\frac{\partial \log P(x^k|\theta)}{\partial\theta}|x^0)$ is the bias.

Bias to ML (expection):
$$
\langle \frac{\partial E(x;\theta)}{\partial\theta} \rangle_{p(x|\theta)}- \langle \frac{\partial E(x;\theta)}{\partial\theta} \rangle_{\int K(x^k|x)p^0(x)}
$$

*Remark.* in discrete form: $G(p_\theta - K^kp_0)$

*Conclusion.* The speed of convergence of CD depends on the expression of $E$ and that of MCMC

## Application

Useful formula
$$
\frac{\partial x^TWz}{\partial W} = x\circ z=\{x_iz_j\}
$$

### FVBM

model distr.: $\frac{1}{Z(W)}e^{x^TWx}$
where $W$ is symmetric, diag W =0

$\Delta w_{ij}=\eta(\langle x_ix_j\rangle_{x^0}-\langle x_ix_j\rangle_{x^k})$

matrix form: $\Delta W=\eta(\langle x\circ x\rangle_{x^0}-\langle x\circ x\rangle_{x^k})$

*Algo.*
input $X$
1. initialize $W$
2. $X^1$ by MCMC/Gibbs sampling from $X^0$ (How do you implement it by Gibbs sampling?)
3. update $W$
4. repeat 2-3

*General Setting*
$x^{(j)}=0,1$

$p(x^{(j)}|x^{(-j)})\sim e^{w\cdot x^{(j)}}$.


### RBM

joint distr.: $\frac{1}{Z(W)}e^{x^TWz}$

conditional distr.: $P(z|x)\sim \mathrm{softmax}\{x^TWz,z\in\mathcal{Z}\}$

$$
\frac{\partial\bar{l}}{\partial\theta}=\langle x\circ z\rangle_{p(z|x),X^0}-\langle x\circ z\rangle_{p(X,Z)}\\
\approx\langle x\circ z\rangle_{p(z|x),X^0}-\langle x\circ z\rangle_{p(z|x),X^1}
$$


*Algo.(CD-RBM)*

- Gibbs sampling: $X^0\to Z$ by $P(z|x)$ $\to$  $X^1$ by $P(x|z)$
- update $W$ by $\langle x\circ E(z|x) \rangle_{X^0} -\langle x\circ E(z|x) \rangle_{X^1}$

*Remark.* if $z|x\sim B(p(z=1|x))$, then $E(z|x)=p(z=1|x)=\mathrm{expit}(x^TW)$


*Remark.* The joint distr. of RBM has form $\frac{1}{Z(W,\alpha,\beta)}e^{x^TWz+x^T\alpha+\beta^T z}$.


### ICA
$$
P(x|w)=|det(W)|\prod_j p_j(\sum_iw_{ji}x_i)
$$

The algo. is left to readers

### Laten value models

*Fact.*  $F(x)=\log \sum_z e^{E(x,z)} \Longrightarrow \frac{\partial F}{\partial\theta} = \mathbb{E}_{z\sim P(z|x)} \frac{\partial E(x,z)}{\partial\theta}$,

Model:
$$
p(x,z|\theta)=\frac{ e^{E(x,z|\theta)}}{Z(\theta)}\\
p(x|\theta)=\frac{\sum_z e^{E(x,z|\theta)}}{Z(\theta)}
$$

We have
$$
\frac{\partial \log p(x|\theta)}{\partial\theta}=\langle\frac{\partial E}{\partial\theta}\rangle_{p(z|x)}-\langle\frac{\partial E}{\partial\theta}\rangle_{p(x,z)}\\
$$

==>
$$
\frac{\partial\bar{l}}{\partial\theta}=\langle\frac{\partial E}{\partial\theta}\rangle_{p(z|x), X^0}-\langle\frac{\partial E}{\partial\theta}\rangle_{p(x,z)}\\
\approx\langle\frac{\partial E}{\partial\theta}\rangle_{p(z|x), X^0}-\langle\frac{\partial E}{\partial\theta}\rangle_{p(z|x), X^1}~\text{if P(z|x) is easy to compute}\\
\approx\langle\frac{\partial E}{\partial\theta}\rangle_{Z\sim p(z|x), X^0}-\langle\frac{\partial E}{\partial\theta}\rangle_{Z\sim p(z|x), X^1}~\text{else}
$$


*Algo.*
Input $X^0$

- Gibbs sampling: $X^0\to z$ by $P(z|x)$ $\to$  $X^1$ by $P(x|z)\to z^1$
- update $W$ by $\langle \frac{\partial E}{\partial\theta} \rangle_{X^0,z} -\langle \frac{\partial E}{\partial\theta} \rangle_{X^1,z^1}$
(or by $\langle \int dz\frac{\partial E}{\partial\theta} p(z|x) \rangle_{X^0} -\langle \int dz\frac{\partial E}{\partial\theta} \rangle_{X^1}$)


### Mixed models
distr: $p(x)=\sum_k \pi_kp_k(x|\theta)=\frac{\sum_k\pi_ke^{E_k(x,\theta)}}{Z(\theta)}$

*Remark* GMM has not such form in general.

## Variant CDs

### Persistent CD(Tieleman, 2008)

input $X^0$
1. initialize $W$
2. $X^1$ by MCMC from $X^0$ (with $W$)
3. update $W$; $X^0\leftarrow X^1$
4. repeat 2-3

### Weighted CD

IS for CD,
$$
\frac{\partial p(x|\theta)}{\partial\theta}
=  \frac{\partial E(x;\theta)}{\partial\theta}-\mathbb{E}_{x\sim p(x|\theta)} \frac{\partial E(x;\theta)}{\partial\theta}\\
= \frac{\partial E(x;\theta)}{\partial\theta}-\mathbb{E}_{x\sim q} \frac{p(x|\theta)}{q(x)} \frac{\partial E(x;\theta)}{\partial\theta}\\
\approx  \frac{\partial E(x;\theta)}{\partial\theta}-\sum_{x\sim q} w_x\frac{\partial E(x;\theta)}{\partial\theta}
$$



## CD for transfer learning

for readers

---

*Exercise:* 😅
1. Talk about what type of distr./model the CD dose (or dose not) serve for.
2. CD algo for weighted likelihood $\ln \sum_kw_kp(x_k)$
3. CD algo for marginal likelihood $\int p(x,z)dz$ or $\sum_z p(x,z)$
4. load a dataset of images from the web, then use CD to learn it and generate a new image.
5. Given a laten variable model $P(x,z)=\frac{e^{E(x,z)}}{Z}$. Let $F(x)=\log \sum_z e^{E(x,z)}$. Verify $D F = E_{z\sim P(z|x)}D E(x,z)$, $D$ is 1-order diff. operator such as $\partial_{x_i}$ or $\nabla_x$. Compute $D^2F=?$
6. For multivariate Gaussian $N(\mu,\Sigma)$, give out CD1, under Gibbs sampling.

*References*

1. Geoffrey E. Hinton. Training Products of Experts by Minimizing Contrastive Divergence.
2. Miguel A. Carreira-Perpinan, Geoffrey E. Hinton. On Contrastive Divergence Learning.
3. https://medium.com/datatype/restricted-boltzmann-machine-a-complete-analysis-part-3-contrastive-divergence-algorithm-3d06bbebb10c
4. https://hackmd.io/@A8e-o-EGSGq0GvfIVrSFYw/SJW561DQK
5. Enrique Romero Merino, Ferran Mazzanti Castrillejo
, Jordi Delgado Pin, David Buchaca Prat. Weighted Contrastive Divergence.