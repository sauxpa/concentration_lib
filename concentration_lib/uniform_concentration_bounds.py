import numpy as np
from typing import Callable, Union, List
from .concentration_bounds import bentkus_bound, chernoff_subgaussian_bound


def laplace_chernoff_uniform_bound(
    delta: float,
    n: Union[List, int],
    R: float,
    mode: str = 'sum',
) -> Union[List, float]:
    """Laplace Chernoff uniform concentration bound.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    R: float
    Sub-gaussian parameter.

    mode: str
    Concentration of sum or mean.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)
    bound_sum = R * np.sqrt(2 * (1 + n) * np.log(np.sqrt(1 + n) / delta))
    if mode == 'sum':
        return bound_sum
    else:
        return bound_sum / n


def laplace_hoeffding_uniform_bound(
    delta: float,
    n: Union[List, int],
    B: float,
    B_minus: float = None,
    mode: str = 'sum',
) -> Union[List, float]:
    """Laplace Hoeffding uniform concentration bound.
    Same as Chernoff, here for naming convention mostly.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    R: float
    Sub-gaussian parameter.

    B: float
    Support upper bound.

    B_minus: float
    Support lower bound (-B by default).

    mode: str
    Concentration of sum or mean.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)
    if not B_minus:
        B_minus = -B
    R = (B - B_minus) / 2
    return laplace_chernoff_uniform_bound(delta, n, R, mode)


def chernoff_peeling_uniform_bound(
    delta: float,
    n: Union[List, int],
    R: float,
    h: Callable,
    eta: float = 1.1,
    mode: str = 'sum',
) -> Union[List, float]:
    """Chernoff uniform concentration (geometric time peeling).

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    R: float
    Sub-gaussian parameter.

    h: Callable
    Sizing function for time peeling.

    eta: float
    Geometric time peeling parameter.

    mode: str
    Concentration of sum or mean.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)

    bounds = np.zeros(len(n))
    k_temp = None

    for i, _n in enumerate(n):
        k = np.ceil(np.log(_n) / np.log(eta)) - 1
        if k == k_temp:
            bounds[i] = bounds[i - 1]
        else:
            c = np.floor(eta ** (k + 1)).astype('int')
            bounds[i] = chernoff_subgaussian_bound(
                delta / h(k), c, R, mode=mode
                )
            k_temp = k
    return bounds


def hoeffding_peeling_uniform_bound(
    delta: float,
    n: Union[List, int],
    B: float,
    h: Callable,
    eta: float = 1.1,
    B_minus: float = None,
    mode: str = 'sum',
) -> Union[List, float]:
    """Chernoff uniform concentration (geometric peeling).
    Same as Chernoff, here for naming convention mostly.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    B: float
    Support upper bound.

    B_minus: float
    Support lower bound (-B by default).

    h: Callable
    Sizing function for time peeling.

    eta: float
    Geometric time peeling parameter.

    mode: str
    Concentration of sum or mean.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)
    if not B_minus:
        B_minus = -B
    R = (B - B_minus) / 2
    return chernoff_peeling_uniform_bound(delta, n, R, h, eta, mode)


def chernoff_union_uniform_bound(
    delta: float,
    n: Union[List, int],
    R: float,
    mode: str = 'sum',
) -> Union[List, float]:
    """Chernoff uniform concentration (union bound).

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    R: float
    Sub-gaussian parameter.

    mode: str
    Concentration of sum or mean.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)
    bound_sum = R * np.sqrt(2 * n * np.log(n * (1 + n) / delta))
    if mode == 'sum':
        return bound_sum
    if mode == 'mean':
        return bound_sum / n


def hoeffding_union_uniform_bound(
    delta: float,
    n: Union[List, int],
    B: float,
    B_minus: float = None,
    mode: str = 'sum',
) -> Union[List, float]:
    """Chernoff uniform concentration (union bound).
    Same as Chernoff, here for naming convention mostly.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    B: float
    Support upper bound.

    B_minus: float
    Support lower bound (-B by default).

    mode: str
    Concentration of sum or mean.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)
    if not B_minus:
        B_minus = -B
    R = (B - B_minus) / 2
    return chernoff_union_uniform_bound(delta, n, R, mode)


def bentkus_peeling_uniform_bound(
    delta: float,
    n: Union[List, int],
    A: float,
    B: float,
    h: Callable,
    eta: float = 1.1,
    mode: str = 'sum',
    tol: float = 1e-8,
) -> Union[List, float]:
    """Bentkus uniform concentration (geometric time peeling).

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    A: float
    Standard deviation bound.

    B: float
    Support upper bound.

    h: Callable
    Sizing function for time peeling.

    eta: float
    Geometric time peeling parameter.

    mode: str
    Concentration of sum or mean.

    tol: float
    Small epsilon to add for an edge case in computing quantiles.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)

    bounds = np.zeros(len(n))
    k_temp = None

    for i, _n in enumerate(n):
        k = np.ceil(np.log(_n) / np.log(eta)) - 1
        if k == k_temp:
            bounds[i] = bounds[i - 1]
        else:
            c = np.floor(eta ** (k + 1)).astype('int')
            bounds[i] = bentkus_bound(
                delta / h(k), c, A, B, mode=mode, tol=tol
                )
            k_temp = k
    return bounds
