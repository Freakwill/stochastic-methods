# Stochastic methods

## simulation
$\{X_t\} = A(p)$ where $A$ is a algo, $p$ is the target PDF, return $X_t\sim p$ (asymptotically)

## accept-reject algo.

**instumental proba. density** $g$ for $f$: $f\leq Mg, M\geq 1$; easy-to-sample-from

*Algo*
Input target PDF $f$, instumental PDF $g$
Return a sample $\{x_t\} \sim f$
1. $x_t\sim g, u_t\sim U[0,1]$
2. accept $x_t$ if $u_t\leq f(x_t)/Mg(x_t)$, otherwise reject it.                                                                                                                                                                                                                                               
3. loop t=1,2...

## Importance sampling

### compute expectation
$$
Eh(X) = E h(Y)\frac{p_X(Y)}{p_Y(Y)}\\
\approx \frac{1}{N}\sum_{y_i\sim p_Y} h(y_i)\frac{p_X(y_i)}{p_Y(y_i)}~~\text{(unbiased form)}
\\
\approx \frac{\sum_{y_i\sim p_Y} h(y_i)p_X(y_i)/p_Y(y_i)}{\sum_{y_i\sim p_Y}p_X(y_i)/p_Y(y_i)} ~~\text{(biased form)}
\\
\approx \frac{N-1}{N}\sum_{y_i\sim p_Y}  \frac{h(y_i)p_X(y_i)/p_Y(y_i)}{\sum_{y_j\sim p_Y,j\neq i}p_X(y_j)/p_Y(y_j)} ~~\text{(biased form as left one out)}
$$
where $p_X$ is intractable but $p_Y$ is "easy-to-sample-from" and $p_X\ll p_Y$.

esp. when $Y\sim U(\mathcal{X})$, $Eh\approx \frac{\sum_{y_i\sim p_Y} h(y_i)p_X(y_i)}{\sum_{y_i\sim p_Y}p_X(y_i)} ~~\text{(biased form)}$

Introduce $k(z|y)$, we have
$$
Eh(X) = E_{x,y\sim p(x)q(y)}h(X)=Eh(Z)\frac{p(Z,Y)}{q(Z,Y)}\\
= E_{y\sim q}E_{z\sim k(z|y)}h(z)\frac{p(z)}{k(z|y)}
$$

### integral
$$
\int h(x) = E \frac{h(X)}{p(X)}\\
\approx \frac{1}{N}\sum_{x_i\sim p} \frac{h(x_i)}{p(x_i)}
$$


## Optimisation

### Simulated Annealing


*Algo*

Input obj. function $h$;
output max of $h$

0. init. $\theta_0$ and temperature $T$
1. $\Delta \theta\sim g(x|\theta_i)$ ($g$ is sym. around $\theta_i$)
2. let $\Delta h = h(\theta_{i}+\Delta\theta)-h(\theta_i)$
$$
\theta_{i+1}\sim \begin{cases}
\theta_i+\Delta\theta, & p=\min(e^{\Delta h/T},1)\\
\theta_i, & 1-p
\end{cases}
$$
update $T$ (decrease to 0)

*Remark* MCMC for $p(x)\sim e^{h/T}$, given $T$

### Genetic Algorithm

---

*References*
C. P. Robert and G. Casella. Monte Carlo Statistical Methods. Springer.