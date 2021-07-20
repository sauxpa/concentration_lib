import numpy as np
from typing import Union, List


def chernoff_subgaussian_bound(
    delta: float,
    n: Union[List, int],
    R: float,
    mode: str = 'sum',
) -> float:
    """Chernoff sub-Gaussian concentration bound.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    R: float
    Sub-Gaussian parameter.

    mode: str
    Concentration of sum or mean.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)
    bound_sum = R * np.sqrt(2 * n * np.log(1 / delta))
    if mode == 'sum':
        return bound_sum
    else:
        return bound_sum / n


def hoeffding_bound(
    delta: float,
    n: Union[List, int],
    B: float,
    B_minus: float = None,
    mode: str = 'sum',
) -> float:
    """Hoeffding concentration bound.
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
    return chernoff_subgaussian_bound(delta, n, R, mode)


def bernstein_bound(
    delta: float,
    n: Union[List, int],
    sigma: float,
    B: float,
    B_minus: float = None,
    mode: str = 'sum',
) -> float:
    """Bernstein concentration bound.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    sigma: float
    Standard deviation.

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

    bound_sum = sigma * np.sqrt(
            2 * n * np.log(1 / delta)
            ) + (B - B_minus) * np.log(1 / delta) / 3
    if mode == 'sum':
        return bound_sum
    else:
        return bound_sum / n


def bentkus_bound(
    delta: float,
    n: Union[List, int],
    A: float,
    B: float,
    mode: str = 'sum',
    tol: float = 1e-8,
) -> float:
    """Bentkus concentration bound using binomial quantile.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    A: float
    Standard deviation bound.

    B: float
    Support upper bound.

    mode: str
    Concentration of sum or mean.

    tol: float
    Small epsilon to add for an edge case in computing quantiles.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)
    if np.issubdtype(type(n), np.integer):
        bound = _bentkus_bound(delta, n, A, B, mode, tol)
    else:
        bound = np.zeros(len(n))
        for i in range(len(n)):
            bound[i] = _bentkus_bound(delta, n[i], A, B, mode, tol)
    return bound


def _bentkus_bound(
    delta: float,
    n: int,
    A: float,
    B: float,
    mode: str = 'sum',
    tol: float = 1e-8,
) -> float:
    """Bentkus concentration bound using binomial quantile.
    Helper function (will need better vectorisation in the future).

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    A: float
    Standard deviation bound.

    B: float
    Support upper bound.

    mode: str
    Concentration of sum or mean.

    tol: float
    Small epsilon to add for an edge case in computing quantiles.
    """
    from scipy.stats import binom

    pAB = A ** 2 / (A ** 2 + B ** 2)
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

    bound_sum = ((A ** 2 + B ** 2) * x - n * A ** 2) / B
    if mode == 'sum':
        return bound_sum
    else:
        return bound_sum / n
