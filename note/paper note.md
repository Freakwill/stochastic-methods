# A CD for Combining Variational Inference and MCMC

instantatneous ELBO: $f(z)=\log p(x,z) -\log q(z)$

*Definition*
$$
L_{VCD}(q):=D_{KL}(q\|p(z|x))-D_{KL}(q^{(t)}\|p(z|x)) + D_{KL}(q^{(t)}\|q)\\
=-E_q(f)+E_{q^{(t)}}(f)
$$
where $q^{(t)}(z)\to p(z|x)$: MCMC, $q^{(0)}(z)=q(z)$

Properties:
1. non-negative
2. when $q(z)=p(z|x)$, it reaches the min.


VI based on VCD:
$$
\min_q L_{VCD}(q)
$$

*Fact*
$\nabla L_{VCD}=E_q\nabla q - E_{q^{(t)}}\nabla q+ E_{q(z_0)}(E_{Q^{(t)}(z|z_0)}f(z) \nabla q(z_0))$


*Corollary*
$\nabla L_{VCD}\approx \nabla q(z^0) - \nabla q(z^t)+ f(z^t) \nabla q(z_0), z^0\sim q$


# Improved Contrastive Divergence Training of Energy-Based Models

*Definition*
$$
CD_k(\theta,\phi):=D_{KL}(p_d\|p)-D_{KL}(p^k(\phi)\|p);
$$