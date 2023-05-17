# Diffusion Models

## Models

**Definition(Diffusion Model)**
- data distr.: $q(x^{(0)})$
- model distr: $p(x^{(0)})$
- tractable distr: $\pi(x^{(T)})=p(x^{(T)})$
- generative model/backward trajectory: $p(x^{(t-1)}|x^{(t)})$
- forward trajectory(Markov process): $q(x^{(t)}|x^{(t-1)})$(e.g. Gaussian diffusion: $q(x^{(t)}|x^{(t-1)})\sim N(x^{(t-1)}\sqrt{1-\beta_t},\beta_t)$, in the case, $q(x^{(t)}|x^{(0)})\sim N(x^{(0)}\sqrt{\bar{\alpha}_t},1-\bar{\alpha}_t),\bar{\alpha}_t:=\prod_t\alpha_t,\alpha_t=1-\beta_t$)

Parameters:
- $\beta_t$ or $\bar{\alpha}_t$: noise schedule
- $\sqrt{\bar{\alpha}_t}$: signal rate

rvs:
- data: $x^{(0)}$,
- representation: $x^{(T)}$

objective: 
$$l(p):=H(q,p)=\int q \log p\\
l(p)\geq \int q(x^{(0\cdots T)}) \log (p(x^{(T)})\prod_t\frac{p(x^{(t-1)}|x^{(t)})}{q(x^{(t)}|x^{(t-1)})})\mathrm{d}x^{(0)}\cdots x^{(T)}
$$
by *variational ineq.*,
$$
\log p(x^{(0)}) = \log \int q(x^{(1\cdots T)}|x^{(0)}) p(x^{(T)})\prod_{t=1}^T\frac{p(x^{(t-1)}|x^{(t)})}{q(x^{(t)}|x^{(t-1)})}\mathrm{d}x^{(1)}\cdots x^{(T)}\\
\geq \int q(x^{(1\cdots T)}|x^{(0)}) \log (p(x^{(T)})\prod_t\frac{p(x^{(t-1)}|x^{(t)})}{q(x^{(t)}|x^{(t-1)})})\mathrm{d}x^{(1)}\cdots x^{(T)}
$$

Lower bound as objective for diffusion model:
$$
L:=\int q(x^{(T)}|x^{(0)}) \log \frac{p(x^{(T)})}{q(x^{(T)}|x^{(0)})}\mathrm{d}x^{(T)}\\
+\sum_t \int q(x^{(t-1)},x^{(t)}|x^{(0)})\log \frac{p(x^{(t-1)}|x^{(t)})}{q(x^{(t-1)}|x^{(t)}, x^{(0)})}\mathrm{d}x^{(t-1)}x^{(t)}\\+\int q(x^{(1)}|x^{(0)})\log p(x^{(1)}|x^{(0)})
$$

$$-l=
\bigg( D(q(x^{(T)}|x^{(0)})\| p(x^{(T)}))\\
+\sum_t D(q(x^{(t-1)},x^{(t)}|x^{(0)})\| p(x^{(t-1)},x^{(t)}))\\+D(q(x^{(1)}|x^{(0)})\| p(x^{(1)}|x^{(0)}))\bigg)
$$

Note $p(x^{(T)})=\pi(x^{(T)})$ and
by design
$$
H(q(x^{(t)}),\pi(x^{(t)}))=H(x^{(T)})\\
p(x^{(1)}|x^{(0)})=q(x^{(1)}|x^{(0)})
$$
We get
$$
L=-\sum_{t=2}^T\sum_{x^{(0)}}\int \mathrm{d}x^{(t)}q(x^{(t)}|x^{(0)})D_{KL}(p(x^{(t-1)}|x^{(t)},x^{(0)})\|q(x^{(t)}|x^{(t-1)}))\\
+H_q(X^{(T)}|X^{(0)})-H_q(X^{(1)}|X^{(0)})-H(X^{(T)})
$$

![](https://picx.zhimg.com/80/v2-29563658cff776aee4fb49d84eec2741_1440w.webp)

## Algorithm

Design:
$\sigma_t^2=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_t$

Consider Gaussian diffusion:

$$
L=\sum_t L_t\\
L_t\sim -\frac{1}{\sigma^2_t}\sum_{x^{(0)}}\int dx^{(t)}q(x^{(t)}|x^{(0)})\|\tilde{\mu}(x^{(t)},x^{(0)})-\mu_\theta(x^{(t)},t)\|\\
\approx \|\epsilon-\epsilon_{\theta,t}(x^{(t)}(x^{(0)},\epsilon))\|,(0\leq t<T)
$$
where $x^{(t)}(x^{(0)},\epsilon):=\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$.

train NN $\epsilon_{\theta,t}$ by data $\{(x^{(t)}(x^{(0)},\epsilon_l),\epsilon_l),\epsilon\sim N(0,1)\}$


---
*Exercise*

1. Given a latent variable model $p(x,z)$ with variational distr. $q(z|x)$. $q(x)$ represents data distr. and let $q(x,z)=q(z|x)q(x)$. 
  $$\int q(x)L_x=\int q(x,z)\log\frac{p(x,z)}{q(z|x)}\sim D_{KL}(q(x,z)\|p(x,z))
  $$
  where $L_x$ is LEBO.


*References*

1. Jonathan Ho, Ajay Jain, Pieter Abbeel. Denoising Diffusion Probabilistic Models, 2020.
2. Calvin Luo, Understanding  Diﬀusion  Models:   A  Uniﬁed  Perspective, 2022