"""
Functions to calculate marginal (non-uniform) concentration bounds for
the standard deviation.
"""

# Author: Patrick Saux <patrick.saux@inria.fr>


import numpy as np
from typing import Union, List
from .concentration_bounds import bentkus_bound
from .utils import check_mode, check_side, return_bound


def chi2_std_dev_bound(
    samples: List = [],
    delta: float = 0.05,
    side: str = 'lower',
    mode: str = 'sum',
    **kwargs,
) -> Union[List, float]:
    """Chi^2 quantile concentration bound.
    Usually very tight but only valid for Gaussian samples.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(sigma > sigma_hat + U(delta)) < delta
    Lower: P(sigma < sigma_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.
    """
    check_mode(mode)
    check_side(side)

    from scipy.stats import chi2

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == 'both':
        delta /= 2

    # Sample size
    n = len(samples)

    # Empirical standard deviation
    sigma_hat = np.std(samples)

    if side == 'lower':
        bound_mean = sigma_hat - sigma_hat * np.sqrt(
            (n - 1) / chi2(n - 1).ppf(1 - delta)
            )
    elif side == 'upper':
        bound_mean = -sigma_hat + sigma_hat * np.sqrt(
            (n - 1) / chi2(n - 1).ppf(delta)
            )
    else:
        bound_mean = np.array(
            [
                sigma_hat - sigma_hat * np.sqrt(
                    (n - 1) / chi2(n - 1).ppf(1 - delta)
                ),
                -sigma_hat + sigma_hat * np.sqrt(
                    (n - 1) / chi2(n - 1).ppf(delta)
                )
            ]
        )

    if mode == 'mean':
        return bound_mean
    else:
        return bound_mean * n


def hoeffding_std_dev_bound(
    n: Union[List, int],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    side: str = 'lower',
    mode: str = 'mean',
    **kwargs,
) -> Union[List, float]:
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

    bound_mean = (upper_bound - lower_bound) / 2 * (
        2 / np.floor(n / 2) * np.log(1 / delta)
        ) ** 0.25
    return return_bound(
        n, bound_mean, side, 'mean', mode
        )


def maurer_pontil_std_dev_bound(
    n: Union[List, int],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    side: str = 'lower',
    mode: str = 'mean',
    **kwargs,
) -> Union[List, float]:
    """Maurer-Pontil standard deviation concentration bound:
    P(sigma - sigma_hat >= bound) <= delta.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound (-upper_bound by default).

    sigma_hat: float or list
    Empirical standard deviation of X.

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

    bound_mean = (upper_bound - lower_bound) * np.sqrt(
        2 / (n - 1) * np.log(1 / delta)
        )
    return return_bound(
        n, bound_mean, side, 'mean', mode
        )


def bentkus_std_dev_bound_crude(
    n: Union[List, int],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    side: str = 'lower',
    mode: str = 'mean',
    **kwargs,
) -> Union[List, float]:
    """Bentkus standard deviation concentration bound.

    Follows from writing the sample variance as a sum of U statistics,
    which are themselves bounded if the underlying X_1, ..., X_n are.
    A slightly tighter constant is derived from Popoviciu's inequality
    on variance.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    upper_bound: float
    Support upper bound.

    lower_bound: float
    Support lower bound (-upper_bound by default).

    side: str
    Bentkus bound is asymmetric for the variance.
    Upper: P(sigma  > sigma_hat + U_+(delta)) < delta
    Lower: P(sigma  < sigma_hat - U_-(delta)) < delta

    mode: str
    Concentration of sum or mean.
    """
    if lower_bound is None:
        lower_bound = -upper_bound

    n2 = np.floor(n / 2).astype('int')

    if side == 'upper':
        return np.sqrt(
            bentkus_bound(
                n2,
                delta,
                (upper_bound - lower_bound) ** 2 * 3 / 16,
                (upper_bound - lower_bound) ** 2 / 4,
                side=side,
                mode=mode,
                )
            )
    elif side == 'lower':
        return np.sqrt(
            bentkus_bound(
                n2,
                delta,
                (upper_bound - lower_bound) ** 2 * 3 / 16,
                (upper_bound - lower_bound) ** 2 / 2,
                side=side,
                mode=mode,
                )
            )
    else:
        return [
            np.sqrt(
                bentkus_bound(
                    n2,
                    delta / 2,
                    (upper_bound - lower_bound) ** 2 * 3 / 16,
                    (upper_bound - lower_bound) ** 2 / 2,
                    side='lower',
                    mode=mode,
                    )
                ),
            np.sqrt(
                bentkus_bound(
                    n2,
                    delta / 2,
                    (upper_bound - lower_bound) ** 2 * 3 / 16,
                    (upper_bound - lower_bound) ** 2 / 4,
                    side='upper',
                    mode=mode,
                    )
                )
            ]


def bentkus_pinelis_std_dev_bound(
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = None,
    lower_bound: float = None,
    side: str = 'lower',
    mode: str = 'mean',
    tight: bool = False,
    **kwargs,
) -> Union[List, float]:
    """Bentkus standard deviation concentration bound
    with Pinelis relaxation:
    P(sigma - sigma_hat >= bound) <= delta.

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
    Upper: P(sigma  > sigma_hat + U(delta)) < delta
    Lower: P(sigma  < sigma_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    tightest: bool
    Whether to use the tight Rademacher-Gaussian tail control

    """
    check_mode(mode)
    check_side(side)

    from scipy.stats import norm

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == 'both':
        delta /= 2

    if lower_bound is None:
        lower_bound = -upper_bound

    # Sample size
    n = len(samples)
    n2 = np.floor(n / 2).astype('int')

    # Empirical standard deviation
    sigma_hat = np.std(samples)

    if tight:
        c = 0.25 / (1 - norm.cdf(np.sqrt(2)))
        # Result from Pinelis 2007
        # (Theorem 1 in
        # https://www.esaim-ps.org/articles/ps/pdf/2007/01/ps0642.pdf)
        # which was later tightened.
        # c *=  (
        #     1 + 1 / 250 * (
        #         1 + norm.pdf(np.sqrt(3)) / (1 - norm.cdf(np.sqrt(3)))
        #         )
        #     )
    else:
        c = np.exp(2) / 2

    q = (
        (upper_bound - lower_bound)
        * norm.ppf(1 - delta / c) / (2 * np.sqrt(2 * n2))
        )
    if side == 'lower':
        bound_mean = sigma_hat + q - np.sqrt(q ** 2 + sigma_hat ** 2)
    elif side == 'upper':
        bound_mean = -sigma_hat + q + np.sqrt(q ** 2 + sigma_hat ** 2)
    else:
        bound_mean = np.array(
            [
                sigma_hat + q - np.sqrt(q ** 2 + sigma_hat ** 2),
                -sigma_hat + q + np.sqrt(q ** 2 + sigma_hat ** 2)
            ]
        )

    if mode == 'mean':
        return bound_mean
    else:
        return bound_mean * n


def chi2_zero_mean_std_dev_bound(
    samples: List = [],
    delta: float = 0.05,
    side: str = 'lower',
    mode: str = 'sum',
    **kwargs,
) -> Union[List, float]:
    """Chi^2 quantile concentration bound assuming centred distribution.
    Usually very tight but only valid for Gaussian samples.
2 ** (1 / (1 + eta))
    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    n: int or list
    Sample size.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(sigma > sigma_hat + U(delta)) < delta
    Lower: P(sigma < sigma_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.
    """
    check_mode(mode)
    check_side(side)

    from scipy.stats import chi2

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == 'both':
        delta /= 2

    # Sample size
    n = len(samples)

    # Empirical standard deviation
    sigma_hat = np.std(samples)

    # Mean of squares
    S2 = np.mean(samples ** 2)

    if side == 'lower':
        bound_mean = sigma_hat - np.sqrt(
            S2 / (chi2(n).ppf(1 - delta) / n)
            )
    elif side == 'upper':
        bound_mean = -sigma_hat + np.sqrt(
            S2 / (chi2(n).ppf(delta) / n)
            )
    else:
        bound_mean = np.array(
            [
                sigma_hat - np.sqrt(
                    S2 / (chi2(n).ppf(1 - delta) / n)
                    ),
                -sigma_hat + np.sqrt(
                    S2 / (chi2(n).ppf(delta) / n)
                    )
            ]
        )

    if mode == 'mean':
        return bound_mean
    else:
        return bound_mean * n


def empirical_chernoff_zero_mean_std_dev_bound(
    samples: List = [],
    delta: float = 0.05,
    rho: float = 1.0,
    eta: float = 1.0,
    side: str = 'lower',
    mode: str = 'sum',
    **kwargs,
) -> Union[List, float]:
    """Empirical Chernoff standard deviation concentration bound assuming
    centred distribution with standard deviation equal to the sub-chi^2
    parameter (tight sub-chi^2).

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    rho: float
    Ratio of standard deviation and sub-Chi2 parameter.
    X^2 is rho*sigma-sub-Chi2, rho <= 1.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(sigma  > sigma_hat + U(delta)) < delta
    Lower: P(sigma  < sigma_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.
    """
    check_mode(mode)
    check_side(side)

    # By union, a two-sided bound at level delta requires
    # two one-sided bounds at level delta / 2.
    if side == 'both':
        delta /= 2

    # Sample size
    n = len(samples)

    # Bound not valid for too small sample/too large confidence
    if delta < np.exp(-8 * n / (1 + eta)):
        if side == 'both':
            return np.ones(2) * np.nan
        else:
            return np.nan

    # Empirical standard deviation
    sigma_hat = np.std(samples)

    # Empirical mean of squares (unbiased estimator of variance
    # under the assumption of zero expectation)
    S2 = np.mean(np.abs(samples) ** (1 + eta))

    if side == 'lower':
        bound_mean = sigma_hat - 1 / rho * (
            1 / 2 * S2 / (
                np.sqrt(1 / (1 + eta)) + np.sqrt(1 / n * np.log(1 / delta))
                ) ** 2
            ) ** (1 / (1 + eta))
    elif side == 'upper':
        bound_mean = -sigma_hat + 1 / rho * (
            S2 / (
                np.sqrt(2 / (1 + eta)) - np.sqrt(1 / n * np.log(1 / delta))
                ) ** 2
            ) ** (1 / (1 + eta))
    else:
        bound_mean = np.array(
            [
                 sigma_hat - 1 / rho * (
                     1 / 2 * S2 / (
                         np.sqrt(1 / (1 + eta)) + np.sqrt(
                             1 / n * np.log(1 / delta)
                             )
                         ) ** 2
                     ) ** (1 / (1 + eta)),
                 -sigma_hat + 1 / rho * (
                     S2 / (
                         np.sqrt(2 / (1 + eta)) - np.sqrt(
                             1 / n * np.log(1 / delta)
                             )
                         ) ** 2
                     ) ** (1 / (1 + eta))
            ]
        )

    if mode == 'mean':
        return bound_mean
    else:
        return bound_mean * n
