import numpy as np
from typing import Union, List
from .concentration_bounds import bentkus_bound


def hoeffding_std_dev_bound(
    delta: float,
    n: Union[List, int],
    B: float,
    B_minus: float = None,
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
    """
    if not B_minus:
        B_minus = -B

    return np.sqrt(3 / 8) * (B - B_minus) * (
        2 / np.floor(n / 2) * np.log(1 / delta)
        ) ** 0.25


def maurer_pontil_std_dev_bound(
    delta: float,
    n: Union[List, int],
    B: float,
    B_minus: float = None,
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
    """
    if not B_minus:
        B_minus = -B

    return (B - B_minus) * np.sqrt(2 / (n - 1) * np.log(1 / delta))


def bentkus_std_dev_bound(
    delta: float,
    n: Union[List, int],
    B: float,
    B_minus: float = None,
) -> float:
    """Bentkus standard deviation concentration bound:
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
    """
    if not B_minus:
        B_minus = -B

    n2 = np.floor(n / 2).astype('int')
    if isinstance(n, int):
        bound = np.sqrt(
            bentkus_bound(
                delta,
                n2,
                (B - B_minus) ** 2 * 3 / 16,
                (B - B_minus) ** 2 / 2,
                mode='mean'
                )
            )
    else:
        bound = np.zeros(len(n))
        for i in range(len(n)):
            bound[i] = np.sqrt(
                bentkus_bound(
                    delta,
                    n2[i],
                    (B - B_minus) ** 2 * 3 / 16,
                    (B - B_minus) ** 2 / 2,
                    mode='mean',
                    )
                )
    return bound


def bentkus_pinelis_std_dev_bound(
    delta: float,
    n: Union[List, int],
    B: float,
    X: List,
) -> float:
    """Bentkus standard deviation concentration bound
    with Pinelis relaxation:
    P(sigma - sigma_hat >= bound) <= delta.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    B: float
    Support in [-B, B].

    X: list
    Samples of X.
    """
    from scipy.stats import norm
    sigma_hat = np.sqrt(np.mean((X[0:-1:2] - X[1::2]) ** 2) / 2)

    n2 = np.floor(n / 2).astype('int')
    c = np.exp(2) / 2
    q = B * norm.ppf(1 - delta / c) / (np.sqrt(2 * n2))

    if isinstance(n, int):
        bound = -sigma_hat + q + np.sqrt(q ** 2 + sigma_hat ** 2)
    else:
        bound = np.zeros(len(n))
        for i in range(len(n)):
            bound = -sigma_hat[i] + q[i] + np.sqrt(
                q[i] ** 2 + sigma_hat[i] ** 2
                )
    return bound
