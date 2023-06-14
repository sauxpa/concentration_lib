"""
Functions to calculate time-uniform concentration bounds.
These are allowed to depend on prior knowledge such as the exact variance
of the underlying distributions, by opposition to the empirical bounds.
"""

# Author: Patrick Saux <patrick.saux@inria.fr>

import numpy as np
from typing import Callable, List, Union
from .utils import (
    return_bound,
    init_nan,
    return_interval_root_search_bound,
    root_scalar,
    set_nan,
)


def laplace_chernoff_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    R: float = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    c: float = 1.0,
    **kwargs,
) -> List[float]:
    """Laplace Chernoff uniform concentration bound.

    n_start: int
    Minimal sample size for which the bound is computed.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    R: float
    Sub-gaussian parameter.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.

    c: float
    Regularisation parameter in Laplace method.
    """
    # Cast to numpy array if not already the case
    # (np.asarray does nothing if it is, hence not costly to do so)
    samples = np.asarray(samples)

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == "both":
        delta /= 2

    n = len(samples)
    nn = np.linspace(1, n, n)

    bounds = R * np.sqrt(2 * (c + nn) * np.log(np.sqrt(1 + nn / c) / delta))
    bounds = return_bound(nn, bounds, side, "sum", mode)
    return set_nan(bounds, side, n_start, n_end)


def laplace_hoeffding_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    c: float = 1.0,
    **kwargs,
) -> List[float]:
    """Laplace Hoeffding uniform concentration bound.
    Same as Chernoff, here for naming convention mostly.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    R: float
    Sub-gaussian parameter.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound (-upper_bound by default).

    side: str ('upper', 'lower')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.

    c: float
    Regularisation parameter in Laplace method.
    """
    if lower_bound is None:
        lower_bound = -upper_bound

    return laplace_chernoff_uniform_bound(
        n_start=n_start,
        samples=samples,
        delta=delta,
        R=(upper_bound - lower_bound) / 2,
        side=side,
        mode=mode,
        n_end=n_end,
        c=c,
    )


def laplace_chernoff_mixture_peeling_uniform_bound_generic(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    R: float = None,
    h: Callable = None,
    grid_step: float = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    **kwargs,
) -> List[float]:
    """Laplace Chernoff uniform concentration bound with geometric time peeling
    (with arbitrary budget function) on the mixture parameter.

    n_start: int
    Minimal sample size for which the bound is computed.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    R: float
    Sub-gaussian parameter.

    h: Callable
    Sizing function for time peeling (confidence budget function).

    grid_step: float
    Geometric time peeling parameter.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.
    """
    from scipy.special import lambertw as W

    # Cast to numpy array if not already the case
    # (np.asarray does nothing if it is, hence not costly to do so)
    samples = np.asarray(samples)

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == "both":
        delta /= 2

    gamma_delta = np.real(-1 / (1 + W(-(delta**2) / np.exp(1), k=-1)))

    n = len(samples)
    nn = np.linspace(1, n, n)

    if n_end == -1:
        n_end = n

    # t_k = floor(grid_step ** k)
    # t_{kmax} <= n < t_{kmax+1}
    # (use ceil to better handle boundary -- powers of grid_step)
    kmax = np.minimum(int(np.ceil(np.log(n) / np.log(grid_step))) - 1, n - 1)

    # Geometric grid
    ttk = np.zeros(kmax + 2, dtype=int)
    for k in range(1, kmax + 2):
        t_ = int(np.floor(grid_step**k))
        if t_ <= ttk[k - 1]:
            t_ = ttk[k - 1] + 1
        ttk[k] = t_

    # Bounds at grid knots
    boundknots = np.ones(len(ttk) - 1) * np.nan

    # Bounds at all times
    bounds = np.ones(n) * np.nan

    for k, tk in enumerate(ttk[1:]):
        boundknots[k] = R * np.sqrt(
            2
            * (1 + gamma_delta)
            * tk
            * np.log(h(k + 1) * np.sqrt(1 + 1 / gamma_delta) / delta)
        )
        bounds[ttk[k] : ttk[k + 1]] = boundknots[k]

    bounds = return_bound(nn, bounds, side, "sum", mode)
    return set_nan(bounds, side, n_start, n_end)


def laplace_chernoff_mixture_peeling_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    R: float = None,
    exponent: float = None,
    grid_step: float = None,
    n_max: Union[None, int] = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    **kwargs,
) -> List[float]:
    """Laplace Chernoff uniform concentration bound with geometric time peeling
    (with power budget function) on the mixture parameter.

    n_start: int
    Minimal sample size for which the bound is computed.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    R: float
    Sub-gaussian parameter.

    exponent: float
    Exponent for budget function h(k) = C * h ^ (-exponent),
    where C is a normalisation constant so that sum 1 / h(k) = 1.

    grid_step: float
    Geometric time peeling parameter.

    n_max: int or None
    Maximum sample size for peeling.
    If not None, bounded time peeling will be used.


    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.
    """
    if n_max is None:
        from scipy.special import zeta

        normalisation = zeta(exponent)
    else:
        assert len(samples) <= n_max, "Sample is larger than peeling horizon."
        normalisation = np.sum(np.linspace(1, n_max, n_max) ** (-exponent))

    return laplace_chernoff_mixture_peeling_uniform_bound(
        n_start=n_start,
        samples=samples,
        delta=delta,
        R=R,
        h=lambda k: normalisation * (k + 1) ** exponent,
        grid_step=grid_step,
        side=side,
        mode=mode,
        n_end=n_end,
    )


def laplace_hoeffding_mixture_peeling_uniform_bound_generic(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    h: Callable = None,
    grid_step: float = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    **kwargs,
) -> List[float]:
    """Laplace Hoeffding uniform concentration bound with geometric time peeling
    (with arbitrary budget function) on the mixture parameter.
    Same as Chernoff, here for naming convention mostly.


    n_start: int
    Minimal sample size for which the bound is computed.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound (-upper_bound by default).

    h: Callable
    Sizing function for time peeling (confidence budget function).

    grid_step: float
    Geometric time peeling parameter.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.
    """
    if lower_bound is None:
        lower_bound = -upper_bound

    return laplace_chernoff_mixture_peeling_uniform_bound_generic(
        n_start=n_start,
        samples=samples,
        delta=delta,
        R=(upper_bound - lower_bound) / 2,
        h=h,
        grid_step=grid_step,
        side=side,
        mode=mode,
        n_end=n_end,
    )


def laplace_hoeffding_mixture_peeling_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    exponent: float = None,
    grid_step: float = None,
    n_max: Union[None, int] = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    **kwargs,
) -> List[float]:
    """Laplace Hoeffding uniform concentration bound with geometric time peeling
    (with power budget function) on the mixture parameter.
    Same as Chernoff, here for naming convention mostly.

    n_start: int
    Minimal sample size for which the bound is computed.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound (-upper_bound by default).

    exponent: float
    Exponent for budget function h(k) = C * h ^ (-exponent),
    where C is a normalisation constant so that sum 1 / h(k) = 1.

    grid_step: float
    Geometric time peeling parameter.

    n_max: int or None
    Maximum sample size for peeling.
    If not None, bounded time peeling will be used.


    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.
    """
    if lower_bound is None:
        lower_bound = -upper_bound

    if n_max is None:
        from scipy.special import zeta

        normalisation = zeta(exponent)
    else:
        assert len(samples) <= n_max, "Sample is larger than peeling horizon."
        normalisation = np.sum(np.linspace(1, n_max, n_max) ** (-exponent))

    return laplace_chernoff_mixture_peeling_uniform_bound_generic(
        n_start=n_start,
        samples=samples,
        delta=delta,
        R=(upper_bound - lower_bound) / 2,
        h=lambda k: normalisation * (k + 1) ** exponent,
        grid_step=grid_step,
        side=side,
        mode=mode,
        n_end=n_end,
    )


def chernoff_peeling_uniform_bound_generic(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    R: float = None,
    h: Callable = None,
    grid_step: float = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    **kwargs,
) -> List[float]:
    """Chernoff uniform concentration (geometric time peeling with
    arbitrary budget function).

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    R: float
    Sub-gaussian parameter.

    h: Callable
    Sizing function for time peeling (confidence budget function).

    grid_step: float
    Geometric time peeling parameter.

    side: str ('upper', 'lower')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.
    """
    from .concentration_bounds import chernoff_subgaussian_bound

    n = len(samples)
    nn = np.linspace(1, n, n)

    if n_end == -1:
        n_end = n

    # t_k = floor(grid_step ** k)
    # t_{kmax} <= n < t_{kmax+1}
    kmax = np.minimum(int(np.floor(np.log(n) / np.log(grid_step))), n - 1)

    # Geometric grid
    ttk = np.zeros(kmax + 2, dtype=int)
    ttk[-1] = n
    for k in range(1, kmax + 1):
        t_ = int(np.floor(grid_step**k))
        if t_ <= ttk[k - 1]:
            t_ = ttk[k - 1] + 1
        ttk[k] = t_

    # Bounds at grid knots
    boundknots = np.ones(len(ttk) - 1) * np.nan

    # Bounds at all times
    bounds = np.ones(n) * np.nan

    for k, tk in enumerate(ttk[1:]):
        boundknots[k] = chernoff_subgaussian_bound(
            tk,
            delta / h(k + 1),
            R,
            "lower",  # side does not matter here, the bound is symmetric
            mode,
        )

        if k < len(ttk[1:]) - 1:
            bounds[ttk[k] : ttk[k + 1]] = boundknots[k]
        else:
            bounds[ttk[k] :] = boundknots[k]

    bounds = return_bound(nn, bounds, side, "sum", mode)
    return set_nan(bounds, side, n_start, n_end)


def chernoff_peeling_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    R: float = None,
    exponent: float = None,
    grid_step: float = None,
    n_max: Union[None, int] = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    **kwargs,
) -> List[float]:
    """Chernoff uniform concentration (geometric time peeling with
    power budget function).

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    R: float
    Sub-gaussian parameter.

    exponent: float
    Exponent for budget function h(k) = C * h ^ (-exponent),
    where C is a normalisation constant so that sum 1 / h(k) = 1.

    grid_step: float
    Geometric time peeling parameter.

    n_max: int or None
    Maximum sample size for peeling.
    If not None, bounded time peeling will be used.

    side: str ('upper', 'lower')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.
    """
    if n_max is None:
        from scipy.special import zeta

        normalisation = zeta(exponent)
    else:
        assert len(samples) <= n_max, "Sample is larger than peeling horizon."
        normalisation = np.sum(np.linspace(1, n_max, n_max) ** (-exponent))

    return chernoff_peeling_uniform_bound_generic(
        n_start=n_start,
        samples=samples,
        delta=delta,
        R=R,
        h=lambda k: normalisation * k**exponent,
        grid_step=grid_step,
        side=side,
        mode=mode,
        n_end=n_end,
    )


def hoeffding_peeling_uniform_bound_generic(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    h: Callable = None,
    grid_step: float = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    **kwargs,
) -> List[float]:
    """Hoeffding uniform concentration (geometric time peeling with
    arbitrary budget function).
    Same as Chernoff, here for naming convention mostly.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound (-upper_bound by default).

    h: Callable
    Sizing function for time peeling (confidence budget function).

    grid_step: float
    Geometric time peeling parameter.

    side: str ('upper', 'lower')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.
    """
    if lower_bound is None:
        lower_bound = -upper_bound

    return chernoff_peeling_uniform_bound_generic(
        n_start=n_start,
        samples=samples,
        delta=delta,
        R=(upper_bound - lower_bound) / 2,
        h=h,
        grid_step=grid_step,
        side=side,
        mode=mode,
        n_end=n_end,
    )


def hoeffding_peeling_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    exponent: float = None,
    grid_step: float = None,
    n_max: Union[None, int] = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    **kwargs,
) -> List[float]:
    """Chernoff uniform concentration (geometric time peeling with
    power budget function).
    Same as Chernoff, here for naming convention mostly.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound (-upper_bound by default).

    exponent: float
    Exponent for budget function h(k) = C * h ^ (-exponent),
    where C is a normalisation constant so that sum 1 / h(k) = 1.

    grid_step: float
    Geometric time peeling parameter.

    n_max: int or None
    Maximum sample size for peeling.
    If not None, bounded time peeling will be used.

    side: str ('upper', 'lower')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.
    """
    if n_max is None:
        from scipy.special import zeta

        normalisation = zeta(exponent)
    else:
        assert len(samples) <= n_max, "Sample is larger than peeling horizon."
        normalisation = np.sum(np.linspace(1, n_max, n_max) ** (-exponent))

    return hoeffding_peeling_uniform_bound_generic(
        n_start=n_start,
        samples=samples,
        delta=delta,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        h=lambda k: normalisation * k**exponent,
        grid_step=grid_step,
        side=side,
        mode=mode,
        n_end=n_end,
    )


def chernoff_union_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    R: float = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    **kwargs,
) -> List[float]:
    """Chernoff uniform concentration (union bound).

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    R: float
    Sub-gaussian parameter.

    side: str ('upper', 'lower')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.
    """
    from .concentration_bounds import chernoff_subgaussian_bound

    n = len(samples)
    nn = np.linspace(1, n, n)
    bounds = chernoff_subgaussian_bound(nn, delta / n, R, side, mode)
    return set_nan(bounds, side, n_start, n_end)


def hoeffding_union_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    **kwargs,
) -> List[float]:
    """Hoeffding uniform concentration (union bound).
    Same as Chernoff, here for naming convention mostly.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound (-upper_bound by default).

    side: str ('upper', 'lower')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.
    """
    if lower_bound is None:
        lower_bound = -upper_bound

    return chernoff_union_uniform_bound(
        n_start=n_start,
        samples=samples,
        delta=delta,
        R=(upper_bound - lower_bound) / 2,
        side=side,
        mode=mode,
        n_end=n_end,
    )


def bentkus_peeling_uniform_bound_generic(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    sigma: float = None,
    upper_bound: float = None,
    lower_bound: float = None,
    h: Callable = None,
    grid_step: float = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    tol: float = 1e-8,
    update_bounds: bool = False,
    **kwargs,
) -> List[float]:
    """Bentkus uniform concentration (geometric time peeling with
    arbitrary budget function).

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    sigma: float
    Standard deviation bound.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound.

    h: Callable
    Sizing function for time peeling (confidence budget function).

    grid_step: float
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

    from .concentration_bounds import bentkus_bound

    n = len(samples)
    nn = np.linspace(1, n, n)

    if n_end == -1:
        n_end = n

    # Sample mean
    S = np.cumsum(samples)
    mu_hat = S / nn

    bounds = init_nan(n, side)

    k_temp = None

    if lower_bound is None:
        lower_bound = -upper_bound

    for i in range(n_start, n_end):
        _n = nn[i]
        k = np.ceil(np.log(_n) / np.log(grid_step)) - 1
        if k == k_temp:
            if side == "both":
                bounds[:, i] = bounds[:, i - 1]
            else:
                bounds[i] = bounds[i - 1]
        else:
            c = np.floor(grid_step ** (k + 1)).astype("int")
            if side == "both":
                if update_bounds and i > n_start:
                    bounds[0, i] = bentkus_bound(
                        c,
                        delta / h(k),
                        sigma,
                        upper_bound,
                        mu_hat[i - 1] - bounds[0, i - 1],
                        "lower",
                        mode,
                        tol,
                    )
                    bounds[1, i] = bentkus_bound(
                        c,
                        delta / h(k),
                        sigma,
                        mu_hat[i - 1] + bounds[1, i - 1],
                        lower_bound,
                        "upper",
                        mode,
                        tol,
                    )
                else:
                    bounds[:, i] = bentkus_bound(
                        c,
                        delta / h(k),
                        sigma,
                        upper_bound,
                        lower_bound,
                        side,
                        mode,
                        tol,
                    )
            else:
                bounds[i] = bentkus_bound(
                    c, delta / h(k), sigma, upper_bound, lower_bound, side, mode, tol
                )
            k_temp = k
    return bounds


def bentkus_peeling_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    sigma: float = None,
    upper_bound: float = None,
    lower_bound: float = None,
    exponent: float = None,
    grid_step: float = None,
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

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    sigma: float
    Standard deviation bound.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound.

    exponent: float
    Exponent for budget function h(k) = C * h ^ (-exponent),
    where C is a normalisation constant so that sum 1 / h(k) = 1.

    grid_step: float
    Geometric time peeling parameter.

    n_max: int or None
    Maximum sample size for peeling.
    If not None, bounded time peeling will be used.

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
    """
    if n_max is None:
        from scipy.special import zeta

        normalisation = zeta(exponent)
    else:
        assert len(samples) <= n_max, "Sample is larger than peeling horizon."
        normalisation = np.sum(np.linspace(1, n_max, n_max) ** (-exponent))

    return bentkus_peeling_uniform_bound_generic(
        n_start=n_start,
        samples=samples,
        delta=delta,
        sigma=sigma,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        h=lambda k: normalisation * (k + 1) ** exponent,
        grid_step=grid_step,
        side=side,
        mode=mode,
        n_end=n_end,
        tol=tol,
        update_bounds=update_bounds,
    )


def kaufmann_koolen_gaussian_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    sigma: float = 1.0,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    root_search_params: dict = {},
    **kwargs,
) -> List[float]:
    """Kaufmann-Koolen uniform concentration bound for Gaussian distributions.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    sigma: float
    Standard deviation.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.

    root_search_params: dict
    Dictionary of parameters for interal root search.
        grid_lo: float
        Root searching grid lower bound.

        grid_up: float
        Root searching grid upper bound.

        grid_size: int
        Initial size of the of the grid to initialise root search.

        n_try: int
        How many times to double the grid size before giving up.
    """
    # Cast to numpy array if not already the case
    # (np.asarray does nothing if it is, hence not costly to do so)
    samples = np.asarray(samples)

    from scipy.special import zeta
    from scipy.optimize import minimize_scalar

    grid_lo = root_search_params.get("grid_lo", -10)
    grid_up = root_search_params.get("grid_up", 10)
    grid_size = root_search_params.get("grid_size", 10)
    n_try = root_search_params.get("n_try", 7)

    n = len(samples)
    nn = np.linspace(1, n, n)

    if n_end == -1:
        n_end = n

    # Sample mean
    mu_hat = np.cumsum(samples) / nn

    bounds = init_nan(n, side)

    log_delta = np.log(1 / delta)

    def g(l_):
        return (
            2 * l_ * (1 - np.log(4 * l_)) + np.log(zeta(2 * l_)) - 0.5 * np.log(1 - l_)
        )

    ret = minimize_scalar(
        lambda l: (g(l) + log_delta) / l,
        bounds=(0.5, 1),
        method="bounded",
    )
    if ret.success:
        C = ret.x
    else:
        return bounds

    for t in range(n_start, n_end):

        def K(m):
            return (
                (mu_hat[t] - m) ** 2 / (2 * sigma**2)
                - 2 / t * np.log(4 + np.log(t))
                - 1 / t * C
            )

        bound_ = return_interval_root_search_bound(
            t,
            mu_hat[t],
            K,
            side=side,
            mode=mode,
            grid_lo=grid_lo,
            grid_up=grid_up,
            grid_size=grid_size,
            n_try=n_try,
        )

        if side == "both":
            bounds[:, t] = bound_[0], bound_[1]
        else:
            bounds[t] = bound_
    return bounds


def kaufmann_koolen_exponential_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    root_search_params: dict = {},
    **kwargs,
) -> List[float]:
    """Kaufmann-Koolen uniform concentration bound for exponential
    distributions.

    n_start: int
    Minimal sample size.

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

    n_end: int
    Maximum sample size for which the bound is computed.

    root_search_params: dict
    Dictionary of parameters for interal root search.
        grid_lo: float
        Root searching grid lower bound.

        grid_up: float
        Root searching grid upper bound.

        grid_size: int
        Initial size of the of the grid to initialise root search.

        n_try: int
        How many times to double the grid size before giving up.
    """
    # Cast to numpy array if not already the case
    # (np.asarray does nothing if it is, hence not costly to do so)
    samples = np.asarray(samples)

    from scipy.special import zeta
    from scipy.optimize import minimize_scalar

    grid_lo = root_search_params.get("grid_lo", -10)
    grid_up = root_search_params.get("grid_up", 10)
    grid_size = root_search_params.get("grid_size", 10)
    n_try = root_search_params.get("n_try", 7)

    n = len(samples)
    nn = np.linspace(1, n, n)

    if n_end == -1:
        n_end = n

    # Sample mean
    mu_hat = np.cumsum(samples) / nn

    bounds = init_nan(n, side)

    log_delta = np.log(1 / delta)

    def g(l_):
        return 2 * l_ * (1 - np.log(4 * l_)) + np.log(zeta(2 * l_)) - np.log(1 - l_)

    ret = minimize_scalar(
        lambda l: (g(l) + log_delta) / l,
        bounds=(0.5, 1),
        method="bounded",
    )
    if ret.success:
        C = ret.x
    else:
        return bounds

    for t in range(n_start, n_end):

        def K(x):
            # Argument in log scale.
            m = np.exp(x)
            r = mu_hat[t] / m
            return r - 1 - np.log(r) - 2 / t * np.log(4 + np.log(t)) - 1 / t * C

        bound_ = return_interval_root_search_bound(
            t,
            mu_hat[t],
            K,
            side=side,
            mode=mode,
            grid_lo=grid_lo,
            grid_up=grid_up,
            grid_size=grid_size,
            n_try=n_try,
            arg_transform=lambda x: np.exp(x),
        )

        if side == "both":
            bounds[:, t] = bound_[0], bound_[1]
        else:
            bounds[t] = bound_
    return bounds


def laplace_subexp_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    R: float = None,
    b: float = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    c: float = 1.0,
    **kwargs,
) -> List[float]:
    """Laplace subexponential uniform concentration bound with truncated
    Gaussian mixture distribution.

    n_start: int
    Minimal sample size for which the bound is computed.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    R: float
    Sub-gaussian parameter.

    b: float
    MGF is controlled by a Gaussian MGF on (-1/b, 1/b).

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.

    c: float
    Regularisation parameter in Laplace method.
    """
    from scipy.stats import norm

    Phi = norm.cdf

    # Cast to numpy array if not already the case
    # (np.asarray does nothing if it is, hence not costly to do so)
    samples = np.asarray(samples)

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == "both":
        delta /= 2

    n = len(samples)

    if n_end == -1:
        n_end = n

    bounds = init_nan(n, side)

    for t in range(n_start, n_end):

        def F(x):
            mu = x / R**2 / (c + t)
            return (
                np.log(
                    1
                    / (2 * Phi(R * np.sqrt(c) / b) - 1)
                    * 1
                    / np.sqrt(1 + t / c)
                    * (
                        Phi((1 / b - mu) * R * np.sqrt(c + t))
                        - Phi((-1 / b - mu) * R * np.sqrt(c + t))
                    )
                )
                + x**2 / (2 * R**2 * (c + t))
                - np.log(1 / delta)
            )

        ret = root_scalar(F, method="brentq", bracket=(1e-8, 1e2))
        if ret.converged:
            bound = ret.root
        else:
            bound = np.nan

        if side == "both":
            bounds[:, t] = return_bound(t, bound, side, "sum", mode)
        else:
            bounds[t] = return_bound(t, bound, side, "sum", mode)

    return bounds
