import numpy as np
from typing import Union, List
from .concentration_bounds import bentkus_bound


def empirical_bernstein_bound(
    delta: float,
    n: Union[List, int],
    sigma_hat: Union[List, float],
    B: float,
    B_minus: float = None,
    mode: str = 'sum',
) -> float:
    """Empirical Bernstein concentration bound.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    sigma_hat: float
    Empirical standard deviation.

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

    bound_sum = sigma_hat * np.sqrt(
        2 * n * np.log(2 / delta)
        ) + 7 * n * (B - B_minus) * np.log(2 / delta) / (3 * (n - 1))
    if mode == 'sum':
        return bound_sum
    else:
        return bound_sum / n


def empirical_bentkus_bound(
    delta: float,
    n: Union[List, int],
    sigma_hat: Union[List, float],
    B: float,
    B_minus: float = None,
    mode: str = 'sum',
    tol: float = 1e-8,
) -> float:
    """Empirical Bentkus concentration bound with Pinelis relaxation
    for the variance bound (same as the version presented in the
    ICML 2021 article on near-optimal confidence sequences).

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    sigma_hat: float
    Empirical standard deviation.

    B: float
    Support upper bound.

    mode: str
    Concentration of sum or mean.

    tol: float
    Small epsilon to add for an edge case in computing quantiles.
    """
    from scipy.stats import norm
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)

    if not B_minus:
        B_minus = -B

    g = 1 / (2 * np.sqrt(2 * n)) * (B - B_minus) * norm.ppf(
        1 - 2 * delta / np.exp(2)
        )
    A = g + np.sqrt(g ** 2 + sigma_hat ** 2)
    return bentkus_bound(delta / 2, n, A, B, mode=mode, tol=tol)


def empirical_maurer_pontil_bentkus_bound(
    delta: float,
    n: Union[List, int],
    sigma_hat: float,
    B: float,
    B_minus: float = None,
    mode: str = 'sum',
    tol: float = 1e-8,
) -> float:
    """Empirical Bentkus concentration bound with Maurer-Pontil
    variance bound.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    sigma_hat: float
    Empirical standard deviation.

    B: float
    Support upper bound.

    mode: str
    Concentration of sum or mean.

    tol: float
    Small epsilon to add for an edge case in computing quantiles.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)

    if not B_minus:
        B_minus = -B

    A = sigma_hat + (B - B_minus) * np.sqrt(2 * np.log(2 / delta) / (n - 1))
    return bentkus_bound(delta / 2, n, A, B, mode=mode, tol=tol)
