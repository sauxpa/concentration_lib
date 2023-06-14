"""
Functions to calculate time-uniform concentration bounds.
These are not allowed to depend on prior knowledge such as the exact variance
of the underlying distributions, and instead rely on data-dependent statistics.
"""

# Author: Patrick Saux <patrick.saux@inria.fr>

import numpy as np
from typing import Callable, List, Union
from .concentration_bounds import bentkus_bound
from scipy.stats import norm
from .utils import (
    check_mode,
    check_side,
    init_nan,
    return_interval_root_search_bound,
)


def empirical_bentkus_peeling_uniform_bound_generic(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    h: Callable = None,
    eta: float = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    tol: float = 1e-8,
    update_bounds: bool = False,
    **kwargs,
) -> List[float]:
    """Bentkus uniform concentration (geometric time peeling with
    power budget function).

    delta: float
    Confidence level.

    n: List
    Sample size.

    samples: List
    Empirical samples.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound (-upper_bound by default).

    h: Callable
    Sizing function for time peeling (confidence budget function).

    eta: float
    Geometric time peeling parameter.

    side: str ('upper', 'lower')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.

    tol: float
    Small epsilon to add for an edge case in computing quantiles.

    update_bounds: bool
    Whether to refine bounds online from previous confidence estimates.
    (Theorem 4 in Arun Kuchibhotla-Zheng)
    """
    # Cast to numpy array if not already the case
    # (np.asarray does nothing if it is, hence not costly to do so)
    samples = np.asarray(samples)

    n = len(samples)
    nn = np.linspace(1, n, n)

    # Sample mean
    S = np.cumsum(samples)
    mu_hat = S / nn

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    # For empirical bounds, we need to make the union of three events:
    # one for each side, and one for the upper bound on the variance.
    if side == "both":
        delta /= 3
    else:
        delta /= 2

    if n_end == -1:
        n_end = n

    bounds = init_nan(n, side)

    if lower_bound is None:
        lower_bound = -upper_bound

    k_temp = None
    sigma_bound = np.nan

    for i in range(n_start, n_end):
        _n = nn[i]
        k = np.ceil(np.log(_n) / np.log(eta)) - 1

        if k == k_temp:
            if side == "both":
                bounds[:, i] = bounds[:, i - 1]
            else:
                bounds[i] = bounds[i - 1]
        else:
            sigma_hat = np.std(samples[: int(_n)])
            c = np.floor(eta ** (k + 1)).astype("int")
            g = (
                np.sqrt(np.floor(c / 2))
                / (2 * np.sqrt(2) * _n)
                * (upper_bound - lower_bound)
                * norm.ppf(1 - 2 * (delta / h(k)) / (1 * np.exp(2)))
            )
            sigma_bound = np.nanmin([sigma_bound, g + np.sqrt(g**2 + sigma_hat**2)])
            if side == "both":
                if update_bounds and i > n_start:
                    bounds[0, i] = bentkus_bound(
                        c,
                        delta / h(k),
                        sigma_bound,
                        upper_bound,
                        mu_hat[i - 1] - bounds[0, i - 1],
                        "lower",
                        mode,
                        tol,
                        override_delta_division=True,
                    )
                    bounds[1, i] = bentkus_bound(
                        c,
                        delta / h(k),
                        sigma_bound,
                        mu_hat[i - 1] + bounds[1, i - 1],
                        lower_bound,
                        "upper",
                        mode,
                        tol,
                        override_delta_division=True,
                    )
                else:
                    bounds[:, i] = bentkus_bound(
                        c,
                        delta / h(k),
                        sigma_bound,
                        upper_bound,
                        lower_bound,
                        side,
                        mode,
                        tol,
                        override_delta_division=True,
                    )
            else:
                bounds[i] = bentkus_bound(
                    c,
                    delta / h(k),
                    sigma_bound,
                    upper_bound,
                    lower_bound,
                    side,
                    mode,
                    tol,
                    override_delta_division=True,
                )
            k_temp = k
    return bounds


def empirical_bentkus_peeling_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    exponent: float = None,
    eta: float = None,
    n_max: Union[None, int] = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    tol: float = 1e-8,
    update_bounds: bool = False,
    **kwargs,
) -> List[float]:
    """Bentkus uniform concentration (geometric time peeling with
    power budget function).

    delta: float
    Confidence level.

    n: List
    Sample size.

    samples: List
    Empirical samples.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound (-upper_bound by default).

    exponent: float
    Exponent for budget function h(k) = C * h ^ (-exponent),
    where C is a normalisation constant so that sum 1 / h(k) = 1.

    eta: float
    Geometric time peeling parameter.

    n_max: int or None
    Maximum sample size. If not None, bounded time peeling will be used.

    side: str ('upper', 'lower')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.

    tol: float
    Small epsilon to add for an edge case in computing quantiles.

    update_bounds: bool
    Whether to refine bounds online from previous confidence estimates.
    (Theorem 4 in Arun Kuchibhotla-Zheng)
    """
    if n_max is None:
        from scipy.special import zeta

        normalisation = zeta(exponent)
    else:
        assert len(samples) <= n_max, "Sample is larger than peeling horizon."
        normalisation = np.sum(np.linspace(1, n_max, n_max) ** (-exponent))

    return empirical_bentkus_peeling_uniform_bound_generic(
        n_start=n_start,
        samples=samples,
        delta=delta,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        h=lambda k: normalisation * (k + 1) ** exponent,
        eta=eta,
        side=side,
        mode=mode,
        n_end=n_end,
        tol=tol,
    )


def empirical_hedged_capital_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    theta: float = 0.5,
    c: float = 0.5,
    bias_mean: float = 0.5,
    bias_var: float = 0.25,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    root_search_params: dict = {},
    **kwargs,
) -> List[float]:
    """Empirical Hedged Capital bound.
    See https://arxiv.org/pdf/2010.09686.pdf.

    n_start: int
    Mnimal sample size.

    delta: float
    Confidence level.

    samples: List
    Empirical samples.

    upper_bound: float
    Support upper bound.

    lower_bound: float
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

    side: str ('upper', 'lower')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.

    root_search_params: dict
    Dictionary of parameters for interal root search.
        tol: float
        Small epsilon to add for numberical stability
        (division by zero when root searching).

        grid_size: int
        Initial size of the of the grid to initialise root search.

        n_try: int
        How many times to double the grid size before giving up.
    """
    check_mode(mode)
    check_side(side)

    # Cast to numpy array if not already the case
    # (np.asarray does nothing if it is, hence not costly to do so)
    samples = np.asarray(samples)

    tol = root_search_params.get("tol", 1e-8)
    grid_size = root_search_params.get("grid_size", 10)
    n_try = root_search_params.get("n_try", 7)

    if lower_bound is None:
        lower_bound = -upper_bound

    n = len(samples)
    nn = np.linspace(1, n, n)

    # Sample mean (true scale)
    mu_hat = np.cumsum(samples) / nn

    # Transform samples to be between 0 and 1
    samples = (samples - lower_bound) / (upper_bound - lower_bound)

    if n_end == -1:
        n_end = n

    bounds = init_nan(n, side)

    # Biased sample statistics used in the predictable mixing weights
    mu_hat_biased = (bias_mean + np.cumsum(samples)) / np.linspace(2, n + 1, n)
    var_hat_biased = (
        bias_var + np.cumsum((samples - mu_hat_biased) ** 2)
    ) / np.linspace(2, n + 1, n)

    # Predictable mixture weights
    lambda_PM = np.zeros(n)
    lambda_PM[1:] = np.sqrt(
        2 * np.log(1 / delta) / (nn[1:] * np.log(nn[1:] + 1) * var_hat_biased[:-1])
    )

    for t in range(n_start, n_end):
        # Capital process - 1 / delta
        def K(m):
            lambda_PM_plus = np.minimum(np.abs(lambda_PM[:t]), c / m)
            lambda_PM_minus = np.minimum(np.abs(lambda_PM[:t]), c / (1 - m))
            K_plus = np.prod(1 + lambda_PM_plus * (samples[:t] - m))
            K_minus = np.prod(1 - lambda_PM_minus * (samples[:t] - m))
            return np.maximum(theta * K_plus, (1 - theta) * K_minus) - 1 / delta

        bound_ = return_interval_root_search_bound(
            t,
            mu_hat[t],
            K,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            side=side,
            mode=mode,
            grid_lo=tol,
            grid_up=1 - tol,
            grid_size=grid_size,
            n_try=n_try,
            force_one_sided=True,
        )

        if side == "both":
            bounds[:, t] = bound_[0], bound_[1]
        else:
            bounds[t] = bound_
    return bounds
