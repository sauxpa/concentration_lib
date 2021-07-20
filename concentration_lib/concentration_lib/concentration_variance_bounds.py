import numpy as np
from typing import Union, List
from .concentration_bounds import bentkus_bound


def hoeffding_std_dev_bound(
    delta: float,
    n: Union[List, int],
    B: float,
    B_minus: float = None,
    mode: str = 'mean',
) -> float:
    """Hoeffding standard deviation concentration bound:
    P(sigma - sigma_hat >= bound) <= delta.

    Follows from writing the sample variance as a sum of U statistics,
    which are themselves bounded if the underlying X_1, ..., X_n are.
    A slightly tighter constant is derived from Popoviciu's inequality
    on variance.

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

    bound_mean = np.sqrt(3 / 8) * (B - B_minus) * (
        2 / np.floor(n / 2) * np.log(1 / delta)
        ) ** 0.25
    if mode == 'mean':
        return bound_mean
    else:
        return bound_mean * n


def maurer_pontil_std_dev_bound(
    delta: float,
    n: Union[List, int],
    B: float,
    B_minus: float = None,
    mode: str = 'mean',
) -> float:
    """Maurer-Pontil standard deviation concentration bound:
    P(sigma - sigma_hat >= bound) <= delta.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    B: float
    Support upper bound.

    B_minus: float
    Support lower bound (-B by default).

    sigma_hat: float or list
    Empirical standard deviation of X.

    mode: str
    Concentration of sum or mean.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)

    if not B_minus:
        B_minus = -B

    bound_mean = (B - B_minus) * np.sqrt(2 / (n - 1) * np.log(1 / delta))

    if mode == 'mean':
        return bound_mean
    else:
        return bound_mean * n


def bentkus_std_dev_bound(
    delta: float,
    n: Union[List, int],
    B: float,
    B_minus: float = None,
    upper_or_lower: str = 'lower',
    mode: str = 'mean',
) -> float:
    """Bentkus standard deviation concentration bound.

    Follows from writing the sample variance as a sum of U statistics,
    which are themselves bounded if the underlying X_1, ..., X_n are.
    A slightly tighter constant is derived from Popoviciu's inequality
    on variance.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    B: float
    Support upper bound.

    B_minus: float
    Support lower bound (-B by default).

    upper_or_lower: str
    Bentkus bound is asymmetric for the variance.
    Upper: P(sigma  > sigma_hat + U_+(delta)) < delta
    Lower: P(sigma  < sigma_hat - U_-(delta)) < delta

    mode: str
    Concentration of sum or mean.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)
    assert upper_or_lower in ['lower', 'upper']

    if not B_minus:
        B_minus = -B

    n2 = np.floor(n / 2).astype('int')

    if upper_or_lower == 'upper':
        B2 = (B - B_minus) ** 2 / 4
    else:
        B2 = (B - B_minus) ** 2 / 2

    bound_mean = np.sqrt(
        bentkus_bound(
            delta,
            n2,
            (B - B_minus) ** 2 * 3 / 16,
            B2,
            mode='mean'
            )
        )

    if mode == 'mean':
        return bound_mean
    else:
        return bound_mean * n


def bentkus_pinelis_std_dev_bound(
    delta: float,
    n: Union[List, int],
    sigma_hat: Union[List, float],
    B: float,
    B_minus: float = None,
    mode: str = 'mean',
) -> float:
    """Bentkus standard deviation concentration bound
    with Pinelis relaxation:
    P(sigma - sigma_hat >= bound) <= delta.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    sigma_hat: float or list
    Empirical standard variation.

    B: float
    Support upper bound.

    B_minus: float
    Support lower bound (-B by default).

    mode: str
    Concentration of sum or mean.
    """
    assert mode in ['sum', 'mean'], 'Unknown mode {:s}'.format(mode)

    from scipy.stats import norm

    if not B_minus:
        B_minus = -B

    n2 = np.floor(n / 2).astype('int')
    c = np.exp(2) / 2
    q = (B - B_minus) * norm.ppf(1 - delta / c) / (2 * np.sqrt(2 * n2))

    if np.issubdtype(type(n), np.integer):
        bound_mean = -sigma_hat + q + np.sqrt(q ** 2 + sigma_hat ** 2)
    else:
        bound_mean = np.zeros(len(n))
        for i in range(len(n)):
            bound_mean = -sigma_hat[i] + q[i] + np.sqrt(
                q[i] ** 2 + sigma_hat[i] ** 2
                )

    if mode == 'mean':
        return bound_mean
    else:
        return bound_mean * n
