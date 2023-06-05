# Diffusion Models

## Markovian Hierachical VAE

rvs:
- data: $x_{0}$,
- representation: $x_{T}$
  
$$
(p(x_0,x_1,\cdots,x_T),q(x_1,\cdots,x_{T}|x_0))
$$
where $x_1,\cdots,x_T$ is unobservable, and 
- generative model/backward trajectory: 
  $$p(x_0,x_1,\cdots,x_T)=p(x_T)\prod_tp(x_{t-1}|x_{t})$$
- forward trajectory(Markov process): 
  $$q(x_1,\cdots,x_{T}|x_0))=\prod_tq(x_{t}|x_{t-1})$$
  
---

$$
ELBO:=\int q(x_{T}|x_{0}) \log \frac{p(x_{T})}{q(x_{T}|x_{0})}\mathrm{d}x_{T}\\
+\sum_{t=2}^T \int q(x_{t-1},x_{t}|x_{0})\log \frac{p(x_{t-1}|x_{t})}{q(x_{t-1}|x_{t}, x_{0})}\mathrm{d}x_{t-1}x_{t}\\+\int q(x_{1}|x_{0})\log p(x_{1}|x_{0})\mathrm{d}x_{1}
$$

### Loss

$$
Loss:=-ELBO= D_{KL} (q(x_{T}|x_{0})\| p(x_{T}))\\
+\sum_{t=2}^T \int q(x_{t}|x_{0})\mathrm{d}x_{t}D_{KL}(q(x_{t-1}|x_{t}, x_{0})\|p(x_{t-1}|x_{t}))\\-\int q(x_{1}|x_{0})\log p(x_{1}|x_{0})\mathrm{d}x_{1}
$$
- prior matching term
- denoising matching term
- reconstruction term

## Diffusion Models
basic assumption
- tractable distr: $p(x_{T})$
- forward trajectory(Markov process): $q(x_{t}|x_{t-1})$ is fixed (has no unlearned parameter)
  
**Definition(Diffusion Model)**
- tractable distr: $p(x_{T})\sim N(0,1)$
- generative model/backward trajectory: $p(x_{t-1}|x_{t})\sim N(\mu(t),\Sigma(t))$
- forward trajectory(Gaussian diffusion): $q(x_{t}|x_{t-1})\sim N(x_{t-1}\sqrt{1-\beta_t},\beta_t)$, 

Parameters:
- $\beta_t=1-\alpha_t$ or $\bar{\alpha}_t:=\prod_t\alpha_t$: noise schedule, where $\alpha_t$ is small
- $\sqrt{\bar{\alpha}_t}$: signal rate


![](https://picx.zhimg.com/80/v2-29563658cff776aee4fb49d84eec2741_1440w.webp)



*Fact.*
- $q(x_{t}|x_{0})\sim N(x_{0}\sqrt{\bar{\alpha}_t},1-\bar{\alpha}_t)$
- $q(x_{t-1}|x_{t},x_{0})\sim N(\mu_q(x_t ,x_0),\sigma^2(t))$ where 
  $$\mu_q(x_t,x_0):=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_t-\sqrt{\bar\alpha_{t-1}}(1-\alpha_{t})x_0}{1-\bar\alpha_t}\\
  =\frac{1}{\sqrt{\alpha_t}}x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}\sqrt{\alpha_t}}\epsilon_0
  $$
  and $\sigma^2(t):=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_{t}$.
  
Design I: $p(x_{t-1}|x_{t})\sim N(\mu(t),\Sigma(t))$:
$$
\mu(t)=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_t-\beta_{t}\sqrt{\bar\alpha_{t-1}}\hat{x}(x_t,t)}{1-\bar\alpha_t}\\
\Sigma(t)=\sigma^2(t)
$$

Design II: $p(x_{t-1}|x_{t})\sim N(\mu(t),\Sigma(t))$:
$$
\mu(t)=\frac{1}{\sqrt{\alpha_t}}x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}\sqrt{\alpha_t}}\hat{\epsilon}(x_t,t)\\
\Sigma(t)=\sigma^2(t)
$$


*Fact.*
Under the design I:
$$
D_{KL} (q(x_{t−1}|x_t , x_0) \| p_θ (x_{t−1} |x_t))=\frac{1}{2\sigma_t^2}\frac{(1-\bar{\alpha}_{t-1})\beta_t^2}{(1-\bar{\alpha}_{t})^2}\|\hat{x}(x_t,t)-x_0\|^2\\
=\frac{1}{2}(\frac{1}{1-\bar{\alpha}_{t-1}}-\frac{1}{1-\bar{\alpha}_{t}})\|\hat{x}(x_t,t)-x_0\|^2
$$

Under the design II:
$$
D_{KL} (q(x_{t−1}|x_t , x_0) \| p_θ (x_{t−1} |x_t))=\frac{1}{2\sigma_t^2}\frac{\beta_t^2}{(1-\bar{\alpha}_{t})\alpha_t^2}\|\hat{\epsilon}(x_t,t)-\epsilon_0\|^2
$$


### Algorithm


Loss:
$$
L=\sum_t L_t\\
L_t\approx \sum_{\epsilon\sim N(0,1)}\|\epsilon-\hat{\epsilon}(x_{t},t)\|^2,(0\leq t<T)
$$
where $x_{t}:=\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$.

train NN $\hat\epsilon$ by data $\{(\hat{\epsilon}(x_{t}(x_{0,i},\epsilon_{il}),t),\epsilon_{il}),\epsilon_{il}\sim N(0,1),l=1,\cdots, L\}$ with size of $NL$ for each $t$。


---
*Exercise*

1. Given a latent variable model $p(x,z)$ with variational distr. $q(z|x)$. $q(x)$ represents data distr. and let $q(x,z)=q(z|x)q(x)$. 
  $$\int q(x)L_x=\int q(x,z)\log\frac{p(x,z)}{q(z|x)}\sim D_{KL}(q(x,z)\|p(x,z))
  $$
  where $L_x$ is LEBO.


*References*

1. Jonathan Ho, Ajay Jain, Pieter Abbeel. Denoising Diffusion Probabilistic Models, 2020.
2. Calvin Luo, Understanding  Diﬀusion  Models:   A  Uniﬁed  Perspective, 2022