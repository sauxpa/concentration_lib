"""
Functions to calculate marginal (non-uniform) empirical concentration bounds.
These are not allowed to depend on prior knowledge such as the exact variance
of the underlying distributions, and instead rely on data-dependent statistics.
"""

# Author: Patrick Saux <patrick.saux@inria.fr>

import numpy as np
from typing import Union, List
from .concentration_bounds import bentkus_bound
from scipy.stats import norm, t as student_t
from .helper_ptlm import b_alpha_l2norm
from .utils import check_side, return_bound
from .utils import return_interval_root_search_bound
from .utils import return_interval_root_search_bound2


def empirical_student_bound(
    samples: List = [],
    delta: float = 0.05,
    side: str = 'lower',
    mode: str = 'sum',
    **kwargs,
) -> Union[List, float]:
    """Empirical Bernstein concentration bound.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.
    """
    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    # For empirical bounds, we need to make the union of three events:
    # one for each side, and one for the upper bound on the variance.
    if side == 'both':
        delta /= 2

    # Sample size
    n = len(samples)

    # Empirical standard deviation
    sigma_hat = np.std(samples)

    t_stat = student_t.ppf(1 - delta, n - 1)
    bound_sum = t_stat * sigma_hat * np.sqrt(n)

    return return_bound(
        n, bound_sum, side, 'sum', mode
        )


def empirical_bernstein_bound(
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    side: str = 'lower',
    mode: str = 'sum',
    **kwargs,
) -> Union[List, float]:
    """Empirical Bernstein concentration bound.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

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
    # For empirical bounds, we need to make the union of three events:
    # one for each side, and one for the upper bound on the variance.
    if side == 'both':
        delta /= 3
    else:
        delta /= 2

    if lower_bound is None:
        lower_bound = -upper_bound

    # Sample size
    n = len(samples)

    # Empirical standard deviation
    sigma_hat = np.std(samples)

    bound_sum = (
        sigma_hat * np.sqrt(2 * n * np.log(1 / delta))
        + 7 * n * (upper_bound - lower_bound)
        * np.log(1 / delta) / (3 * (n - 1))
        )
    return return_bound(
        n, bound_sum, side, 'sum', mode
        )


def empirical_bentkus_bound(
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    side: str = 'lower',
    mode: str = 'sum',
    tol: float = 1e-8,
    **kwargs,
) -> Union[List, float]:
    """Empirical Bentkus concentration bound with Pinelis relaxation
    for the variance bound (same as the version presented in the
    ICML 2021 article on near-optimal confidence sequences).

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

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

    tol: float
    Small epsilon to add for an edge case in computing quantiles.
    """
    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    # For empirical bounds, we need to make the union of three events:
    # one for each side, and one for the upper bound on the variance.
    if side == 'both':
        delta /= 3
    else:
        delta /= 2

    # Sample size
    n = len(samples)

    # Empirical standard deviation
    sigma_hat = np.std(samples)

    if lower_bound is None:
        lower_bound = -upper_bound

    g = 1 / (2 * np.sqrt(2 * n)) * (upper_bound - lower_bound) * norm.ppf(
        1 - 2 * delta / (1 * np.exp(2))
        )
    sigma_bound = g + np.sqrt(g ** 2 + sigma_hat ** 2)
    return bentkus_bound(
        n, delta, sigma_bound, upper_bound, side=side, mode=mode, tol=tol,
        override_delta_division=True,
        )


def empirical_hedged_capital_bound(
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    theta: float = 0.5,
    c: float = 0.5,
    bias_mean: float = 0.5,
    bias_var: float = 0.25,
    side: str = 'lower',
    mode: str = 'sum',
    root_search_params: dict = {},
    safe: bool = False,
    **kwargs,
) -> Union[List, float]:
    """Empirical Hedged Capital bound.
    See https://arxiv.org/pdf/2010.09686.pdf.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    upper_bound: float
    Support upper bound.

    upper_bound: float
    Support lower bound (-upper_bound by default).

    theta: float
    Proportion of capital allocated to positive bets.

    c: float
    Thresholding constant in the predictable mixing weights.

    bias_mean: float
    Bias in the empirical mean estimator. Default (0.5) corresponds to
    the uninformative uniform prior on [0, 1],

    bias_var: float
    Bias in the empirical variance estimator. Default (0.25) corresponds to
    consevative prior of maximal variance (for support in [0, 1]).

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    root_search_params: dict
    Dictionary of parameters for interal root search.
        tol: float
        Small epsilon to add for numberical stability
        (division by zero when root searching).

        grid_size: int
        Initial size of the of the grid to initialise root search.

        n_try: int
        How many times to double the grid size before giving up.

    safe: bool
    If True, don't raise if root_scalar fails (see force_one_sided for a
    different exception management policy).
    """
    if lower_bound is None:
        lower_bound = -upper_bound

    tol = root_search_params.get('tol', 1e-8)
    grid_size = root_search_params.get('grid_size', 10)
    n_try = root_search_params.get('n_try', 7)

    # Sample size
    n = len(samples)

    # Sample mean (true scale)
    mu_hat = np.mean(samples)

    # Transform samples to be between 0 and 1
    samples = (samples - lower_bound) / (upper_bound - lower_bound)

    # Biased sample statistics used in the predictable mixing weights
    mu_hat_biased = (
        bias_mean + np.cumsum(samples)
        ) / np.linspace(2, n + 1, n)
    var_hat_biased = (
        bias_var + np.cumsum((samples - mu_hat_biased) ** 2)
        ) / np.linspace(2, n + 1, n)

    # Predictable mixture weights
    lambda_PM = np.zeros(n)
    lambda_PM[1:] = np.sqrt(
        2 * np.log(1 / delta) / (n * var_hat_biased[:-1])
        )

    # Capital process - 1 / delta
    def K(m):
        lambda_PM_plus = np.minimum(np.abs(lambda_PM), c / m)
        lambda_PM_minus = np.minimum(np.abs(lambda_PM), c / (1 - m))
        K_plus = np.prod(1 + lambda_PM_plus * (samples - m))
        K_minus = np.prod(1 - lambda_PM_minus * (samples - m))
        return np.maximum(
            theta * K_plus,
            (1 - theta) * K_minus
            ) - 1 / delta

    return return_interval_root_search_bound(
        n, mu_hat, K,
        upper_bound=upper_bound, lower_bound=lower_bound,
        side=side, mode=mode,
        grid_lo=tol, grid_up=1-tol, grid_size=grid_size, n_try=n_try,
        safe=safe,
        )


def empirical_small_samples_ptlm(
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    side: str = 'lower',
    mode: str = 'sum',
    n_MC: int = int(1e4),
    **kwargs,
) -> Union[List, float]:
    """Towards Practical Mean Bounds for Small Samples
    See http://proceedings.mlr.press/v139/phan21a.html.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

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

    n_MC: int
    Number of Monte Carlo simulations to estimate quantile.
    """
    check_side(side)

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    # For empirical bounds, we need to make the union of three events:
    # one for each side, and one for the upper bound on the variance.
    if side == 'both':
        delta /= 2

    if lower_bound is None:
        lower_bound = -upper_bound

    # Sample size
    n = len(samples)

    # Sample mean (true scale)
    mu_hat = np.mean(samples)

    # Transform samples to be between 0 and 1
    samples = (samples - lower_bound) / (upper_bound - lower_bound)

    if side == 'lower':
        return return_bound(
            n,
            mu_hat
            - lower_bound
            - (upper_bound - lower_bound)
            * (1. - b_alpha_l2norm(1. - samples, delta, num_samples=n_MC)),
            side, 'mean', mode,
            )
    elif side == 'upper':
        return return_bound(
            n,
            -mu_hat
            + lower_bound
            + (upper_bound - lower_bound)
            * b_alpha_l2norm(samples, delta, num_samples=n_MC),
            side, 'mean', mode,
            )
    else:
        return (
            return_bound(
                n,
                mu_hat
                - lower_bound
                - (upper_bound - lower_bound)
                * (1. - b_alpha_l2norm(1. - samples, delta, num_samples=n_MC)),
                'lower', 'mean', mode,
                ),
            return_bound(
                n,
                -mu_hat
                + lower_bound
                + (upper_bound - lower_bound)
                * b_alpha_l2norm(samples, delta, num_samples=n_MC),
                'upper', 'mean', mode,
                )
            )


def empirical_symmetric_bentkus_efron_bound(
    samples: List = [],
    delta: float = 0.05,
    side: str = 'lower',
    mode: str = 'sum',
    root_search_params: dict = {},
    **kwargs,
) -> Union[List, float]:
    """Empirical Bentkus-Efron concentration bound for symmetric distributions.
    https://projecteuclid.org/journals/bernoulli/volume-21/issue-2/
    A-tight-Gaussian-bound-for-weighted-sums-of-Rademacher-random/
    10.3150/14-BEJ603.full

    Efron showed in https://www.jstor.org/stable/2286068 that the
    self-normalized sum of centred symmetric random variables is a mixture
    of normalized Rademacher sums. As it turns out, the tail probability
    of such Rademacher sums can be sharply controlled by the Gaussian tail
    probability up to a (tight) constant c. We apply this control to the
    centred samples X-mu and solve for mu to obtain confidence sets for
    the mean.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

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
    from scipy.stats import norm

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == 'both':
        delta /= 2

    grid_scale = root_search_params.get('grid_scale', 1.0)
    grid_size = root_search_params.get('grid_size', 10)
    n_try = root_search_params.get('n_try', 7)

    # Sample size
    n = len(samples)

    # Sample mean
    mu_hat = np.mean(samples)

    c = 0.25 / (1 - norm.cdf(np.sqrt(2)))

    def f_lower(m):
        return (
            np.sqrt(1 / n * np.mean((samples - m) ** 2))
            * norm.ppf(1 - delta / c)
            + m - mu_hat
            )

    def f_upper(m):
        return (
            np.sqrt(1 / n * np.mean((samples - m) ** 2))
            * norm.ppf(1 - delta / c)
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
