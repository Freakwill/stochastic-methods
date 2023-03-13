# Stochastic methods

## stochastic simulation

## accept-reject algo

instumental PDF $p$, $f\leq Mg, M\geq 1$

*Algo*
Return $x_t \sim f$
1. $x_t\sim g, u_t\sim U[0,1]$
2. accept $x_t$ if $u_t\leq f(x_t)/Mg(x_t)$, otherwise reject it.                                                                                                                                                                                                                                               
3. loop t=1,2...

## importance sampling

- compute expectation
    $$
    Eh(X) = E h(Y)\frac{p_X(Y)}{p_Y(Y)}\\
    \approx \frac{1}{N}\sum_{y_i\sim p_Y} h(y_i)\frac{p_X(y_i)}{p_Y(y_i)}
    $$
    where $p_X$ is intractible to compute.
- integral
    $$
    \int h(x) = E \frac{h(X)}{p(X)}\\
    approx \frac{1}{N}\sum_{x_i\sim p} \frac{h(x_i)}{p(x_i)}
    $$