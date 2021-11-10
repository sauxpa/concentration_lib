"""
Functions to calculate marginal (non-uniform) concentration bounds.
These are allowed to depend on prior knowledge such as the exact variance
of the underlying distributions, by opposition to the empirical bounds.
"""

# Author: Patrick Saux <patrick.saux@inria.fr>

import numpy as np
from typing import Union, List
from .utils import return_bound, root_scalar
from .utils import return_interval_root_search_bound2


def gaussian_bound(
    n: Union[List, int],
    delta: float = 0.05,
    sigma: float = None,
    side: str = 'lower',
    mode: str = 'sum',
    **kwargs,
) -> Union[List, float]:
    """Gaussian quantile concentration bound.
    Usually very tight but only valid for Gaussian samples.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    sigma: float
    Standard deviation.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu > mu_hat + U(delta)) < delta
    Lower: P(mu < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.
    """
    from scipy.stats import norm

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == 'both':
        delta /= 2
    return return_bound(
        n, sigma * np.sqrt(n) * norm.ppf(1 - delta), side, 'sum', mode
        )


def chernoff_subgaussian_bound(
    n: Union[List, int],
    delta: float = 0.05,
    R: float = None,
    eta: float = 1.0,
    side: str = 'lower',
    mode: str = 'sum',
    override_delta_division: bool = False,
    **kwargs,
) -> Union[List, float]:
    """Chernoff sub-Gaussian concentration bound.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    R: float
    Sub-Gaussian parameter.

    eta: float
    Exponent in the sub-Chernoff condition (eta=1 == Gaussian).

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.
    """
    assert eta > 0.0

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == 'both' and not override_delta_division:
        delta /= 2

    return return_bound(
        n,
        R * n ** (1 / (1 + eta))
        * ((1 + 1 / eta) * np.log(1 / delta)) ** (eta / (1 + eta)),
        side, 'sum', mode
        )


def hoeffding_bound(
    n: Union[List, int],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    side: str = 'lower',
    mode: str = 'sum',
    **kwargs,
) -> Union[List, float]:
    """Hoeffding concentration bound.
    Same as Chernoff, here for naming convention mostly.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound (-upper_bound by default).

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.
    """
    if lower_bound is None:
        lower_bound = -upper_bound

    R = (upper_bound - lower_bound) / 2

    return chernoff_subgaussian_bound(n, delta, R, 1.0, side, mode)


def bernstein_bound(
    n: Union[List, int],
    delta: float = 0.05,
    sigma: float = None,
    upper_bound: float = None,
    lower_bound: float = None,
    side: str = 'lower',
    mode: str = 'sum',
    **kwargs,
) -> Union[List, float]:
    """Bernstein concentration bound.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    sigma: float
    Standard deviation.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound (-upper_bound by default).

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.
    """
    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == 'both':
        delta /= 2

    if lower_bound is None:
        lower_bound = -upper_bound

    # if no variance is provided, use an upper bound
    if sigma is None:
        sigma = (upper_bound - lower_bound) / 2

    bound_sum = sigma * np.sqrt(
            2 * n * np.log(1 / delta)
            ) + (upper_bound - lower_bound) * np.log(1 / delta) / 3

    return return_bound(
        n, bound_sum, side, 'sum', mode
        )


def bentkus_bound(
    n: Union[List, int],
    delta: float = 0.05,
    sigma: float = None,
    upper_bound: float = None,
    lower_bound: float = None,
    side: str = 'lower',
    mode: str = 'sum',
    tol: float = 1e-8,
    override_delta_division: bool = False,
    **kwargs,
) -> Union[List, float]:
    """Bentkus concentration bound using binomial quantile.

    delta: float
    Confidence level.

    n: int or List
    Sample size.

    sigma: float
    Standard deviation bound.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    tol: float
    Small epsilon to add for an edge case in computing quantiles.
    """
    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == 'both' and not override_delta_division:
        delta /= 2

    if lower_bound is None:
        lower_bound = -upper_bound

    # if no variance is provided, use an upper bound
    if sigma is None:
        sigma = (upper_bound - lower_bound) / 2

    from scipy.stats import binom

    if side == 'both':
        return np.array([
                bentkus_bound(
                    n, delta, sigma, upper_bound, lower_bound,
                    'lower', mode, tol),
                bentkus_bound(
                    n, delta, sigma, upper_bound, lower_bound,
                    'upper', mode, tol)
            ]
        )

    B = upper_bound - lower_bound

    pAB = sigma ** 2 / (sigma ** 2 + B ** 2)
    Z = binom(n, pAB)
    kk = np.linspace(0, n, n + 1)
    p = Z.sf(kk - 1)

    e = np.empty(n + 1)  # Clipped mean
    v = np.empty(n + 1)  # Clipped variance
    e[-1] = Z.pmf(n) * n
    v[-1] = Z.pmf(n) * n ** 2

    for k in range(2, n + 2):
        e[-k] = e[-k + 1] + Z.pmf(n - k + 1) * (n - k + 1)
        v[-k] = v[-k + 1] + Z.pmf(n - k + 1) * (n - k + 1) ** 2

    psi = (v - kk * e) / (e - kk * p)

    def P2(x):
        if x < n * pAB:
            return 1.0
        elif x > n * pAB and x <= v[0] / e[0]:
            return n*pAB*(1-pAB)/((x-n*pAB)**2+n*pAB*(1-pAB))
        elif x >= n:
            return pAB ** n
        else:
            k = np.searchsorted(psi[1:n], x) + 1
            return (v[k]*p[k]-e[k]**2)/(x**2*p[k]-2*x*e[k]+v[k])

    if np.log(delta) <= n * np.log(pAB):
        x = n + tol
    elif delta <= 1 and delta >= P2(v[0]/e[0]):
        x = n*pAB+np.sqrt(((1-delta)*n*pAB*(1-pAB))/delta)
    else:
        k = np.searchsorted([-P2(psi_k) for psi_k in psi[1:n]], -delta) + 1
        x = (e[k]+np.sqrt(e[k]**2-p[k]*(v[k]-(v[k]*p[k]-e[k]**2)/delta)))/p[k]

    bound_sum = (
        (sigma ** 2 + B ** 2) * x - n * sigma ** 2
        ) / B

    return return_bound(
        n, bound_sum, side, 'sum', mode
        )


def bercu_touati_bound(
    samples: List = [],
    delta: float = 0.05,
    sigma: float = None,
    a: float = None,
    x: float = None,
    side: str = 'lower',
    mode: str = 'sum',
    root_search_params: dict = {},
    **kwargs,
) -> Union[List, float]:
    """Bercu-Touati self-normalized concentration bound.
    See Theorem 2.6 in https://hal.sorbonne-universite.fr/hal-02345000/document
    Note that all it requires is the existence of the second-order moment.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    sigma: float
    Standard deviation bound.

    a: float
    Parameter to control E[exp(X - a/2 * X^2)]

    x: float
    (2/3)^{1/3} * x^{-2/3} e^{-x^2/2} = delta

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    root_search_params: dict
    Dictionary of parameters for interal root search.
        grid_scale: float
        Initial diameter of the grid to initialise root search.

        grid_size: int
        Initial size of the of the grid to initialise root search.

        n_try: int
        How many times to double the grid size before giving up.
    """
    assert a > 1 / 8, 'a must be > 1/8.'

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == 'both':
        delta /= 2

    if x is None:
        def f(x):
            return (
                (2 / 3) ** (1 / 3) * x ** (-2 / 3) * np.exp(-x ** 2 / 2)
                - delta
                )

        ret = root_scalar(f, method='brentq', bracket=(1e-6, 1e2))
        if ret.converged:
            x = ret.root
        else:
            raise Exception(
                'Unfeasible confidence level delta={:.0%}'.format(delta)
                )

    grid_scale = root_search_params.get('grid_scale', 100.0)
    grid_size = root_search_params.get('grid_size', 10)
    n_try = root_search_params.get('n_try', 7)

    # Sample mean
    mu_hat = np.mean(samples)

    # Sample size
    n = len(samples)

    c_a = 2 * (1 - 2 * a + 2 * np.sqrt(a * (a + 1))) / (8 * a - 1)

    def f_lower(m):
        return (
            x * np.sqrt(
                1.5 / n * (
                    a * (np.mean((samples - m) ** 2) + c_a * sigma ** 2)
                    + sigma ** 2
                    )
                )
            + m - mu_hat
            )

    def f_upper(m):
        return (
            x * np.sqrt(
                1.5 / n * (
                    a * (np.mean((samples - m) ** 2) + c_a * sigma ** 2)
                    + sigma ** 2
                    )
                )
            + mu_hat - m
            )

    if side == 'lower':
        return return_interval_root_search_bound2(
            n, mu_hat, f_lower, side='lower', mode=mode,
            grid_scale=grid_scale, grid_size=grid_size, n_try=n_try
            )
    elif side == 'upper':
        return return_interval_root_search_bound2(
            n, mu_hat, f_upper, side='upper', mode=mode,
            grid_scale=grid_scale, grid_size=grid_size, n_try=n_try
            )
    else:
        bound_lower = return_interval_root_search_bound2(
            n, mu_hat, f_lower, side='lower', mode=mode,
            grid_scale=grid_scale, grid_size=grid_size, n_try=n_try
            )
        bound_upper = return_interval_root_search_bound2(
            n, mu_hat, f_upper, side='upper', mode=mode,
            grid_scale=grid_scale, grid_size=grid_size, n_try=n_try,
            )
        return bound_lower, bound_upper


def bercu_touati_zero_mean_bound(
    samples: List = [],
    delta: float = 0.05,
    sigma: float = None,
    a: float = None,
    x: float = None,
    side: str = 'lower',
    mode: str = 'sum',
    **kwargs,
) -> Union[List, float]:
    """Bercu-Touati self-normalized concentration bound.
    See Theorem 2.6 in https://hal.sorbonne-universite.fr/hal-02345000/document
    Note that all it requires is the existence of the second-order moment,
    assuming zero mean. Useful to control deviation of random walk.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    sigma: float
    Standard deviation bound.

    a: float
    Parameter to control E[exp(X - a/2 * X^2)]

    x: float
    (2/3)^{1/3} * x^{-2/3} e^{-x^2/2} = delta

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.
    """
    assert a > 1 / 8, 'a must be > 1/8.'

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == 'both':
        delta /= 2

    if x is None:
        def f(x):
            return (
                (2 / 3) ** (1 / 3) * x ** (-2 / 3) * np.exp(-x ** 2 / 2)
                - delta
                )

        ret = root_scalar(f, method='brentq', bracket=(1e-6, 1e2))
        if ret.converged:
            x = ret.root
        else:
            raise Exception(
                'Unfeasible confidence level delta={:.0%}'.format(delta)
                )

    # Sample size
    n = len(samples)

    # Empirical sum of squares
    S2 = np.sum(samples ** 2)

    c_a = 2 * (1 - 2 * a + 2 * np.sqrt(a * (a + 1))) / (8 * a - 1)

    bound_sum = x * np.sqrt(
        3 / 2 * (a * (S2 + c_a * n * sigma ** 2) + n * sigma ** 2)
        )

    return return_bound(
        n, bound_sum, side, 'sum', mode
        )
