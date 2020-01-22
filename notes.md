# Notes

## Negative binomial NMF

The likelihood of negative binomial distribution is given by the following equation according to [Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution).

$$\mathcal{L}(x \mid r, p) = \frac{(x + r - 1)!}{x! (r-1)!} (1-p)^r p^x$$

Thus, its log-likelihood becomes:

$$l(x | r, p) = \log \Gamma(x + r) - \log \Gamma(x - 1) - \log \Gamma(r) + r \log(1 - p) + x \log (p)$$

An alternative parametrisation of the model uses the mean parameter $\mu = pr / (1 - p)$ and $r$, in which case $p = \mu / (r + \mu)$. Under this formulation, the log-likelihood becomes the following.

$$l(x | r, \mu) = \log \Gamma(x + r) - \log \Gamma(x - 1) - \log \Gamma(r) + r \log(\frac{r}{r + \mu}) + x \log (\frac{\mu}{r + \mu})$$

In NB-NMF, we would like to formulate the problem as:

$$\mathbf{X} \sim \mathcal{NB}(\textbf{WH}, r)$$

where $\mathbf{X} \in \mathcal{R}^{M \times N}$, $\mathbf{W} \in \mathcal{R}^{M \times K}$, $\mathbf{H} \in \mathcal{R}^{K \times N}$ and $\mathbb{E}[\mathbf{X}] = \mathbf{WH}$.

### Multiplicative updates for NB-NMF

In conventional gradient descent, the updates in $\mathbf{W}$ (and equivalently $\mathbf{H}$) take the form of

$$\mathbf{W} \leftarrow \mathbf{W} + \eta_\mathbf{W} \circ \nabla_\mathbf{W} l(\mathbf{W}; \mathbf{X} \mid \mathbf{H}, r)$$

To find the gradient, let us differentiate $l(\mathbf{W}; \mathbf{X} \mid \mathbf{H}, r) = \sum_{m, n} l(x_{m, n} \mid [\mathbf{WH}]_{m, n}, r)$ with respect to $w_{ij}$. We will use the shorthand $[\mathbf{WH}]_{mn} = \mu_{mn}$.

$$
\frac{\delta}{\delta w_{i, j}} \sum_{m, n} l(x_{mn} \mid [\mathbf{WH}]_{mn}, r)  \\
= \frac{\delta}{\delta w_{i, j}} \sum_{m, n} [ \log \Gamma(x_{mn} + r) - \log \Gamma(x_{mn} - 1) - \log \Gamma(r) + r \log(\frac{r}{r + \mu_{mn}}) + x_{mn} \log (\frac{\mu_{mn}}{r + \mu_{mn}}) ]  \\
= \frac{\delta}{\delta w_{i, j}} \sum_{m, n} [ -r \log(r + \mu_{mn}) + x_{mn} \log \mu_{mn} - x_{mn} \log (r + \mu_{mn}) ]  \\
= \frac{\delta}{\delta w_{i, j}} \sum_{m, n} [ x_{mn} \log \mu_{mn} - (x_{mn} + r) \log (r + \mu_{mn}) ]  \\
= \frac{\delta}{\delta w_{i, j}} \sum_{n} [ x_{in} \log \mu_{in} - (x_{in} + r) \log (r + \mu_{in})]  \\
= \sum_n \frac{x_{in}}{\mu_{in}} h_{jn} - \sum_n \frac{(x_{in} + r) }{r + \mu_{in}} h_{jn}  \\
= [\frac{\mathbf{X}}{\mathbf{WH}} \mathbf{H}^\intercal]_{ij} - [ \frac{\mathbf{X} + r}{\mathbf{WH} + r} \mathbf{H}^\intercal ]_{ij}
$$

Thus, we get the following gradient.

$$
\nabla_{\mathbf{W}} l(\mathbf{W} ; \mathbf{X} \mid \mathbf{H}, r) 
= \frac{\mathbf{X}}{\mathbf{WH}} \mathbf{H}^\intercal - \frac{\mathbf{X} + r}{\mathbf{WH} + r} \mathbf{H}^\intercal
$$

Analogous to regular NMF multiplicative updates, we make use of the gradient to set step size $\eta_{\mathbf{W}}$ to $\mathbf{W} / \frac{\mathbf{X} + r}{\mathbf{WH} + r} \mathbf{H}^\intercal$. This yields the following updates for $\mathbf{W}$ (and analogously for $\mathbf{H}$).

$$
\mathbf{W} \leftarrow \mathbf{W} + \eta_\mathbf{W} \circ \nabla_\mathbf{W} l(\mathbf{W}; \mathbf{X} \mid \mathbf{H}, r) = \mathbf{W} \circ \frac{\frac{\mathbf{X}}{\mathbf{WH}} \mathbf{H}^\intercal}{\frac{\mathbf{X} + r}{\mathbf{WH} + r} \mathbf{H}^\intercal}
$$

$$
\mathbf{H} \leftarrow \mathbf{H} \circ \frac{\mathbf{W}^\intercal \frac{\mathbf{X}}{\mathbf{WH}}}{\mathbf{W}^\intercal \frac{\mathbf{X} + r}{\mathbf{WH} + r}}
$$

### Fitting rate parameter in NB-NMF 

Now we wish to find the derivative(s) of the negative binomial log-likelihood function with respect to r.

$$
\frac{\delta}{\delta r} \sum_{m,n} l(r \mid x_{mn} ; \mu_{mn})  \\
= \frac{\delta}{\delta r} \sum_{mn} \left[ \log \Gamma(x + r) - \log \Gamma(x - 1) - \log \Gamma(r) + r \log(\frac{r}{r + \mu_{mn}}) + x_{mn} \log (\frac{\mu_{mn}}{r + \mu_{mn}}) \right] \\
= \frac{\delta}{\delta r} \sum_{mn} \left[ \log \Gamma(x + r) - \log \Gamma(r) + r \log(r) - r \log(r + \mu_{mn}) - x_{mn} \log(r + \mu_{mn}) \right] \\
= \sum_{m,n} \left[ \psi(x_{mn} + r) - \psi(r) + \log(r) + 1 - \log(r + \mu_{mn}) - \frac{r}{r + \mu_{mn}} - \frac{x_{mn}}{r + \mu_{mn}} \right]  \\
= MN(- \psi(r) + \log(r) + 1) + \sum_{m,n} [ \psi(x_{mn} + r) - \log(r + \mu_{mn}) - \frac{r + x_{mn}}{r + \mu_{mn}}]  \\
= MN(- \psi(r) + \log(r) + 1) + \left| \psi(\mathbf{X} + r) - \log(\mathbf{WH} + r) - \frac{\mathbf{X} + r}{\mathbf{WH} + r} \right|_{\sum_{m, n}}
$$

We can use this to get the second derivative, too.

$$
\frac{\delta^2}{\delta^2 r} = \sum_{m,n} l(r \mid x_{mn} ; \mu_{mn})  \\
= MN(-\psi^2 (r) + \frac{1}{r}) + \sum_{m, n} \psi^2 (x_{mn} + r) - \sum_{m, n} \frac{2}{\mu_{mn} + r} + \sum_{m, n} \frac{x_{mn} + r}{(\mu_{mn} + r)^2}
$$

With the first and the second derivate, Newton's method can be used during iterations.

## The full negative binomial model updates

The model is

$$
\mathbf{X} \sim \text{NegBin}(\text{mean} = (\mathbf{WH} + \mathbf{O}) \circ \mathbf{S}, \text{dispersion} = r)
$$

Updates for $\mathbf{W}$:

$$
\mathbf{W} \leftarrow \mathbf{W} \circ
    \frac{\left( \frac{\mathbf{X}}
                      {\mathbf{WH + \mathbf{O}}}
              - \frac{\mathbf{X \circ \mathbf{S}}}
                     {(\mathbf{WH} + \mathbf{O}) \circ \mathbf{S} + r}
              \right) \mathbf{H}^\intercal}
         {\frac{r \mathbf{S}}
               {(\mathbf{WH} + \mathbf{O}) \circ \mathbf{S} + r}
              \mathbf{H}^\intercal}
$$

Equivalently, the updates for $\mathbf{H}$ are:

$$
\mathbf{H} \leftarrow \mathbf{H} \circ
    \frac{\mathbf{W}^\intercal \left( \frac{\mathbf{X}}
                                           {\mathbf{WH + \mathbf{O}}}
              - \frac{\mathbf{X \circ \mathbf{S}}}
                     {(\mathbf{WH} + \mathbf{O}) \circ \mathbf{S} + r}
              \right)}
         {\mathbf{W}^\intercal \frac{r \mathbf{S}}
               {(\mathbf{WH} + \mathbf{O}) \circ \mathbf{S} + r}}
$$

## Comparison of negative binomial formulations

Source | $\Pr(X)$ | Interpretation of $x$ | Interpretation of $p$ | Symbol of $x$ in source
-------|----------|-------------------------|-----------|---
[Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution) | $\frac{\Gamma(x + r)}{x! \Gamma(r)} (1 - p)^r p^x$ | Number of **successes** before $r$ failures | $\Pr(\text{Success})$ | $k$
[Wikipedia alternative formulation](https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations) | $\frac{\Gamma(x + k)}{x! \Gamma(k)} (1 - p)^x p^k$ | Number of **failures** given $k$ successes | $\Pr(\text{Success})$ | $r$
[R](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/NegBinomial.html) | $\frac{\Gamma(x + n)}{x! \Gamma(n)} (1 - p)^x p^n$ | Number of **failures** before $n$ successes. | $\Pr(\text{Success})$ | $x$
[`scipy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html) | $\frac{\Gamma(x + n)}{x! \Gamma(n)} (1 - p)^x p^n$ | Number of **failures** before n successes. | $\Pr(\text{Success})$ | $k$

It was confirmed empirically that the R and scipy formulations are the same.

    # In R
    > dnbinom(1:3, 2, 0.25)
    [1] 0.0937500 0.1054687 0.105468
    
    # In Python
    In [1]: import scipy.stats                                                                                                                                                                                
    scipy.stats.nbinom.
    In [2]: scipy.stats.nbinom.pmf([1, 2, 3], 2, 0.25)                                                                                                                                                        
    Out[2]: array([0.09375   , 0.10546875, 0.10546875])

The only way the formulations differ is that the success and failure probabilities are flipped. Thus, we just need to flip the probability when we use the Scipy function.


## Profile likelihood confidence intervals.

Wilks' theorem states that $2 (l(\hat{\theta}) - l(\theta_0)) \sim \chi^2_{d}$, where $d$ is the difference in free parameters between $\hat{\theta}$ and $\theta$. Thus, asymptotically $1 - \alpha$ of the time, $2 (l(\hat{\theta}) - l(\theta_0)) \le c$, where $\Pr(X <= c) = 1 - \alpha$, when $X \sim \chi^2_d$. Since in profile likelihood, $\hat{\theta}$ is "known" and fixed, we get the following condition for the confidence interval of $\hat{\theta}$.

$$l(\theta_0) \ge l(\hat{\theta}) - c / 2$$