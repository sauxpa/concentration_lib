"""
Utility function to help handle various modes (concentration of sum or mean,
one or two sided bounds...)
"""

# Author: Patrick Saux <patrick.saux@inria.fr>

import numpy as np
from typing import Callable, List, Union
from scipy.optimize import root_scalar as root_scalar_scipy
from scipy.optimize.zeros import RootResults


def check_mode(mode: str):
    assert mode in {'sum', 'mean'}, 'Unknown mode {:s}'.format(mode)


def check_side(side: str):
    assert side in {'lower', 'upper', 'both'}, 'Unknown side {:s}'.format(side)


def return_bound(
    n: Union[List, int],
    bound: Union[List, float],
    side: str = 'lower',
    current_mode: str = 'sum',
    mode: str = 'sum',
) -> Union[List, float]:
    """Utility function to return bound, depending on the mode
    (sum or mean) and side (lower, upper, both),

    n: int or List
    Sample size.

    bound: float or List
    Calculated bound.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    current_mode: str
    Whether bound was calculated in sum or mean mode.

    mode: str
    Concentration of sum or mean.
    """
    check_mode(mode)
    check_side(side)

    if mode == 'mean' and current_mode == 'sum':
        bound /= n
    elif mode == 'sum' and current_mode == 'mean':
        bound *= n

    # If not specified otherwise, just assume the bounds are symmetric:
    # Upper: P(mu  > mu_hat + U(delta)) < delta
    # Lower: P(mu  < mu_hat - U(delta)) < delta
    if side == 'both':
        return np.array([bound, bound])
    else:
        return bound


def root_scalar(
    f: Callable,
    method='brentq',
    bracket=(),
    safe=True,
    **kwargs,
):
    try:
        ret = root_scalar_scipy(f, method=method, bracket=bracket, **kwargs)
    except ValueError as e:
        if safe:
            return RootResults(np.nan, 0, 0, -3)
        else:
            raise e
    else:
        return ret


def return_interval_root_search_bound(
    n: int,
    mu_hat: float,
    K: Callable,
    upper_bound: float = 1.0,
    lower_bound: float = 0.0,
    side: str = 'lower',
    mode: str = 'sum',
    grid_lo: float = 1e-8,
    grid_up: float = 1 - 1e-8,
    grid_size: int = 10,
    n_try: int = 7,
    arg_transform: Callable = None,
    force_one_sided: bool = False,
    safe: bool = False,
) -> Union[float, List[float]]:
    """Utility function to compute bounds defined as boundaries of an
    interval corresponding to level set of a given function.

    n: int
    Sample size.

    mu_hat: float
    Empirical mean.

    K: Callable
    Function that defines the interval as a level set.

    upper_bound: float
    Support upper bound (1.0 by default).

    lower_bound: float
    Support lower bound (0.0 by default).

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    grid_lo: float
    Root searching grid lower bound.

    grid_up: float
    Root searching grid upper bound.

    grid_size: int
    Initial size of the of the grid to initialise root search.

    n_try: int
    How many times to double the grid size before giving up.

    arg_transform: Callable
    Function to transform argument in the root search
    (e.g exp if K is in log scale).

    force_one_sided: bool
    If True and root search fails, return the support bound instead.

    safe: bool
    If True, don't raise if root_scalar fails (see force_one_sided for a
    different exception management policy).
    """
    # The root search on K requires to start with values of opposite signs.
    # First do a coarse grid search to find such values.
    ii = 0
    while ii < n_try:
        mm = np.linspace(grid_lo, grid_up, grid_size * (2 ** ii))
        grid = np.vectorize(K)(mm)
        # K should be positive near its 0 and 1, and negative
        # somewhere in between. If this is not the case for grid,
        # just use a finer grid.
        grid_sign = np.where(grid < 0)[0]
        if grid[0] > 0 and grid[-1] > 0:
            if len(grid_sign) > 0:
                break
        ii += 1
    else:
        if not force_one_sided or len(grid_sign) == 0:
            if side == 'both':
                return np.nan, np.nan
            else:
                return np.nan

    # indices where K is > 0
    left = np.maximum(grid_sign[0] - 1, 0)
    right = np.minimum(grid_sign[-1] + 1, len(grid) - 1)
    # index where K is < 0
    mid = (left + right) // 2

    if side == 'both':
        try:
            ret_lo = root_scalar(
                K, method='brentq',
                bracket=(mm[left], mm[mid]), safe=safe,
                )
        except ValueError as e:
            if force_one_sided:
                bound_mean_lo = mu_hat - lower_bound
            else:
                raise e
        else:
            if ret_lo.converged:
                root = ret_lo.root
                if arg_transform is not None:
                    root = arg_transform(root)
                bound_mean_lo = (
                    mu_hat - lower_bound
                    - (upper_bound - lower_bound) * root
                    )
            else:
                bound_mean_lo = np.nan
        try:
            ret_up = root_scalar(
                K, method='brentq',
                bracket=(mm[mid], mm[right]), safe=safe,
                )
        except ValueError as e:
            if force_one_sided:
                bound_mean_up = (
                    -mu_hat + lower_bound + (upper_bound - lower_bound)
                    )
            else:
                raise e
        else:
            if ret_up.converged:
                root = ret_up.root
                if arg_transform is not None:
                    root = arg_transform(root)
                bound_mean_up = (
                    -mu_hat + lower_bound
                    + (upper_bound - lower_bound) * root
                    )
            else:
                bound_mean_up = np.nan
        return (
            return_bound(n, bound_mean_lo, 'lower', 'mean', mode),
            return_bound(n, bound_mean_up, 'upper', 'mean', mode)
            )
    else:
        if side == 'upper':
            try:
                ret = root_scalar(
                    K, method='brentq',
                    bracket=(mm[mid], mm[right]), safe=safe,
                    )
            except ValueError as e:
                if force_one_sided:
                    bound_mean_up = (
                        -mu_hat + lower_bound + (upper_bound - lower_bound)
                        )
                else:
                    raise e
            else:
                if ret.converged:
                    root = ret.root
                    if arg_transform is not None:
                        root = arg_transform(root)
                    bound_mean = (
                        -mu_hat + lower_bound
                        + (upper_bound - lower_bound) * root
                        )
                else:
                    bound_mean = np.nan
        elif side == 'lower':
            try:
                ret = root_scalar(
                    K, method='brentq',
                    bracket=(mm[left], mm[mid]), safe=safe,
                    )
            except ValueError as e:
                if force_one_sided:
                    bound_mean = mu_hat - lower_bound
                else:
                    raise e
            else:
                if ret.converged:
                    root = ret.root
                    if arg_transform is not None:
                        root = arg_transform(root)
                    bound_mean = (
                        mu_hat - lower_bound
                        - (upper_bound - lower_bound) * root
                        )
                else:
                    bound_mean = np.nan
        return return_bound(n, bound_mean, side, 'mean', mode)


def return_interval_root_search_bound2(
    n: int,
    mu_hat: float,
    f: Callable,
    side: str = 'lower',
    mode: str = 'sum',
    grid_scale: float = 1.0,
    grid_size: int = 10,
    n_try: int = 7,
) -> Union[float, List[float]]:
    """Utility function to compute bounds defined as boundaries of an
    interval corresponding to level set of a given function.
    Difference between this function and return_interval_root_search_bound2 is
    that here the interval is assumed to be one-sidedself.
    Also since its use case is empirical Chernoff, it does not assume
    boundedness.

    n: int
    Sample size.

    mu_hat: float
    Empirical mean.

    f: Callable
    Function that defines the interval as a level set.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    grid_scale: float
    Initial diameter of the grid to initialise root search.

    grid_size: int
    Initial size of the of the grid to initialise root search.

    n_try: int
    How many times to double the grid size before giving up.
    """
    # The root search on f requires to start with values of opposite signs.
    # First do a coarse grid search to find such values.
    ii = 0
    do_proceed = True
    while ii < n_try:
        mm = np.linspace(-grid_scale, grid_scale, grid_size)
        grid = np.vectorize(f)(mm)
        # f should be negative at some point.
        # If this is not the case on this grid, just use a bigger grid.
        grid_sign = np.where(grid < 0)[0]
        if len(grid_sign) > 0:
            break
        grid_scale *= 2
        ii += 1
    else:
        do_proceed = False

    if do_proceed:
        # bracketing of the minimum m such that f(m) < 0
        left = mm[grid_sign[0] - 1]
        right = mm[grid_sign[0]]

        ret = root_scalar(f, method='brentq', bracket=(left, right))
        if ret.converged:
            if side == 'lower':
                bound_mean = mu_hat - ret.root
            elif side == 'upper':
                bound_mean = -mu_hat + ret.root
        else:
            bound_mean = np.nan
    else:
        bound_mean = np.nan

    return return_bound(
        n, bound_mean, side, 'mean', mode
        )
