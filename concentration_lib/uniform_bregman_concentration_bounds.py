import numpy as np
from typing import Callable, List
from .utils import init_nan, return_bound, return_interval_root_search_bound, set_nan


def bregman_bernoulli_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    upper_bound: float = 1.0,
    lower_bound: float = 0.0,
    c: float = 1.0,
    c_func: Callable = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    root_search_params: dict = {},
    **kwargs,
) -> List[float]:
    """Laplace-Bregman uniform concentration bound for Bernoulli distributions.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    upper_bound: float
    Support upper bound (1 by default).

    lower_bound: float
    Support lower bound (0 by default).

    c: float
    Regularisation parameter.

    c_func: Callable
    Regularisation parameter as a function of sample size.
    (Only a heuristic)

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
        tol: float
        Small epsilon to add for numberical stability
        (division by zero when root searching).

        grid_size: int
        Initial size of the of the grid to initialise root search.

        n_try: int
        How many times to double the grid size before giving up.
    """
    # Cast to numpy array if not already the case
    # (np.asarray does nothing if it is, hence not costly to do so)
    samples = np.asarray(samples)

    from scipy.special import loggamma

    tol = root_search_params.get("tol", 1e-8)
    grid_size = root_search_params.get("grid_size", 10)
    n_try = root_search_params.get("n_try", 7)

    n = len(samples)
    nn = np.linspace(1, n, n)

    # Sample mean (true scale)
    S = np.cumsum(samples)
    mu_hat = S / nn

    # Transform samples to be between 0 and 1
    samples = (samples - lower_bound) / (upper_bound - lower_bound)

    if n_end == -1:
        n_end = n

    bounds = init_nan(n, side)

    log_delta = np.log(1 / delta)

    dynamic_tuning = c_func is not None

    for t in range(n_start, n_end):
        if dynamic_tuning:
            c_ = c_func(t)
        else:
            c_ = c

        def K(m):
            if (m <= 0) or (m >= 1):
                return np.inf
            return (
                (
                    S[t] * np.log(1 / m)
                    + (t - S[t]) * np.log(1 / (1 - m))
                    + loggamma(S[t] + c_ * m)
                    + loggamma(t - S[t] + c_ * (1 - m))
                    - loggamma(c_ * m)
                    - loggamma(c_ * (1 - m))
                )
                - log_delta
                - loggamma(t + c_)
                + loggamma(c_)
            )

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


def bregman_gaussian_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    sigma: float = 1.0,
    c: float = 1.0,
    c_func: Callable = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    **kwargs,
) -> List[float]:
    """Laplace-Bregman uniform concentration bound for Gaussian distributions.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    sigma: float
    Standard deviation.

    c: float
    Regularisation parameter.

    c_func: Callable
    Regularisation parameter as a function of sample size.
    (Only a heuristic)

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.
    """
    n = len(samples)
    nn = np.linspace(1, n, n)

    if n_end == -1:
        n_end = n

    dynamic_tuning = c_func is not None

    if dynamic_tuning:
        c_ = c_func(nn)
    else:
        c_ = c

    bounds_mean = (
        sigma
        * np.sqrt((nn + c_) * (2 * np.log(1 / delta) + np.log((nn + c_) / c_)))
        / nn
    )
    bounds = return_bound(nn, bounds_mean, side, "mean", mode)
    return set_nan(bounds, side, n_start, n_end)


def bregman_gaussian_variance_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    c: float = 1.0,
    c_func: Callable = None,
    mu: float = 0.0,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    root_search_params: dict = {},
    **kwargs,
) -> List[float]:
    """Laplace-Bregman uniform concentration bound for Gaussian
    distributions with known mean, unknown variance.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    c: float
    Regularisation parameter.

    c_func: Callable
    Regularisation parameter as a function of sample size.
    (Only a heuristic)

    mu: float
    Known mean.

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(sigma  > sigma_hat + U(delta)) < delta
    Lower: P(sigma  < sigma_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.

    integration_params: dict
    Dictionary of parameters for numerical integration:
        int_domain_lo: float
        Integration domain lower bound.

        int_domain_up: float
        Integration domain upper bound.

        k_max: int
        Number of numerical integration steps.

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

    from scipy.special import loggamma

    grid_lo = root_search_params.get("grid_lo", -10)
    grid_up = root_search_params.get("grid_up", 10)
    grid_size = root_search_params.get("grid_size", 10)
    n_try = root_search_params.get("n_try", 7)

    n = len(samples)
    nn = np.linspace(1, n, n)

    # Sum of squares
    Q = np.cumsum((samples - mu) ** 2)

    # Sample standard deviation
    sigma_hat = np.sqrt(Q / nn)

    if n_end == -1:
        n_end = n

    bounds = init_nan(n, side)

    log_delta = np.log(1 / delta)

    dynamic_tuning = c_func is not None

    for t in range(n_start, n_end):
        if dynamic_tuning:
            c_ = c_func(t)
        else:
            c_ = c

        def K(x):
            # Argument in log scale.
            s = np.exp(x)
            return (
                -((t + c_) / 2 + 1) * np.log((c_ + Q[t] / s**2) / (t + c_))
                + Q[t] / (2 * s**2)
                - log_delta
                + t / 2 * np.log(2)
                + (c_ / 2 + 1) * np.log(c_)
                - ((t + c_) / 2 + 1) * np.log(t + c_)
                - loggamma(c_ / 2 + 2)
                + loggamma((t + c_) / 2 + 2)
            )

        bound_ = return_interval_root_search_bound(
            t,
            sigma_hat[t],
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


def bregman_exponential_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    c: float = 1.0,
    c_func: Callable = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    root_search_params: dict = {},
    **kwargs,
) -> List[float]:
    """Laplace-Bregman uniform concentration bound for exponential
    distributions.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    c: float
    Regularisation parameter.

    c_func: Callable
    Regularisation parameter as a function of sample size.
    (Only a heuristic)

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

    from scipy.special import loggamma

    grid_lo = root_search_params.get("grid_lo", -10)
    grid_up = root_search_params.get("grid_up", 10)
    grid_size = root_search_params.get("grid_size", 10)
    n_try = root_search_params.get("n_try", 7)

    n = len(samples)
    nn = np.linspace(1, n, n)

    # Sample mean
    mu_hat = np.cumsum(samples) / nn

    if n_end == -1:
        n_end = n

    bounds = init_nan(n, side)

    log_delta = np.log(1 / delta)

    dynamic_tuning = c_func is not None

    for t in range(n_start, n_end):
        if dynamic_tuning:
            c_ = c_func(t)
        else:
            c_ = c

        def K(x):
            # Argument in log scale.
            m = np.exp(x)
            r = mu_hat[t] / m
            return (
                t / (t + c_) * r
                - (1 + 1 / (t + c_)) * np.log((t / (t + c_)) * r + (c_ / (t + c_)))
                - np.log(t + c_)
                - 1 / (t + c_) * (loggamma(c_) - loggamma(t + c_))
                - 1 / (t + c_) * log_delta
                + c_ / (t + c_) * np.log(c_)
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
            arg_transform=lambda x: np.exp(x),
        )

        if side == "both":
            bounds[:, t] = bound_[0], bound_[1]
        else:
            bounds[t] = bound_
    return bounds


def bregman_gamma_fixed_shape_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    c: float = 1.0,
    c_func: Callable = None,
    k: float = 1.0,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    root_search_params: dict = {},
    **kwargs,
) -> List[float]:
    """Laplace-Bregman uniform concentration bound for fixed-shape Gamma
    distributions.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    c: float
    Regularisation parameter.

    c_func: Callable
    Regularisation parameter as a function of sample size.
    (Only a heuristic)

    k: float
    Shape parameter.

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

    from scipy.special import loggamma

    grid_lo = root_search_params.get("grid_lo", -10)
    grid_up = root_search_params.get("grid_up", 10)
    grid_size = root_search_params.get("grid_size", 10)
    n_try = root_search_params.get("n_try", 7)

    n = len(samples)
    nn = np.linspace(1, n, n)

    S = np.cumsum(samples)
    # Sample mean
    mu_hat = S / nn

    if n_end == -1:
        n_end = n

    bounds = init_nan(n, side)

    log_delta = np.log(1 / delta)

    dynamic_tuning = c_func is not None

    for t in range(n_start, n_end):
        if dynamic_tuning:
            c_ = c_func(t)
        else:
            c_ = c

        def K(x):
            # Argument in log scale.
            lambda_ = np.exp(x)
            return (
                k * (t + c_) * ((S[t] + c_ * k * lambda_) / ((t + c_) * k * lambda_))
                - (k * (t + c_) + 1)
                * np.log((S[t] + c_ * k * lambda_) / ((t + c_) * k * lambda_))
                - log_delta
                - k * (t + c_) * np.log(k * (t + c_))
                - c_ * k * (1 - np.log(c_ * k))
                - loggamma(c_ * k)
                + loggamma(k * (t + c_))
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
            arg_transform=lambda x: np.exp(x),
        )

        if side == "both":
            bounds[:, t] = bound_[0], bound_[1]
        else:
            bounds[t] = bound_
    return bounds


def bregman_weibull_fixed_shape_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    c: float = 1.0,
    c_func: Callable = None,
    k: float = 1.0,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    root_search_params: dict = {},
    **kwargs,
) -> List[float]:
    """Laplace-Bregman uniform concentration bound for fixed-shape Weibull
    distributions.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    c: float
    Regularisation parameter.

    c_func: Callable
    Regularisation parameter as a function of sample size.
    (Only a heuristic)

    k: float
    Shape parameter.

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

    from scipy.special import loggamma

    grid_lo = root_search_params.get("grid_lo", -10)
    grid_up = root_search_params.get("grid_up", 10)
    grid_size = root_search_params.get("grid_size", 10)
    n_try = root_search_params.get("n_try", 7)

    n = len(samples)
    nn = np.linspace(1, n, n)

    Sk = np.cumsum(samples**k)
    # Sample mean
    mu_hat = np.cumsum(samples) / nn

    if n_end == -1:
        n_end = n

    bounds = init_nan(n, side)

    log_delta = np.log(1 / delta)

    dynamic_tuning = c_func is not None

    for t in range(n_start, n_end):
        if dynamic_tuning:
            c_ = c_func(t)
        else:
            c_ = c

        def K(x):
            # Argument in log scale.
            lambda_ = np.exp(x)
            return (
                (t + c_) * ((Sk[t] + c_ * lambda_**k) / ((t + c_) * lambda_**k))
                - ((t + c_) + 1)
                * np.log((Sk[t] + c_ * lambda_**k) / ((t + c_) * lambda_**k))
                - log_delta
                - (t + c_) * np.log((t + c_))
                - c_ * (1 - np.log(c_))
                - loggamma(c_)
                + loggamma(t + c_)
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
            arg_transform=lambda x: np.exp(x),
        )

        if side == "both":
            bounds[:, t] = bound_[0], bound_[1]
        else:
            bounds[t] = bound_
    return bounds


def bregman_chi2_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    c: float = 1.0,
    c_func: Callable = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    integration_params: dict = {},
    root_search_params: dict = {},
    safe: bool = False,
    continuous_prior: bool = False,
    **kwargs,
) -> List[float]:
    """Laplace-Bregman uniform concentration bound for chi2 distributions.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    c: float
    Regularisation parameter.

    c_func: Callable
    Regularisation parameter as a function of sample size.
    (Only a heuristic)

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(mu  > mu_hat + U(delta)) < delta
    Lower: P(mu  < mu_hat - U(delta)) < delta

    mode: str
    Concentration of sum or mean.

    n_end: int
    Maximum sample size for which the bound is computed.

    integration_params: dict
    Dictionary of parameters for numerical integration:
        int_domain_lo: float
        Integration domain lower bound (log scale).

        int_domain_up: float
        Integration domain upper bound (log scale).

        k_max: int
        Number of numerical integration steps.

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

    safe: bool
    If True, don't raise if root_scalar fails.
    """
    # Cast to numpy array if not already the case
    # (np.asarray does nothing if it is, hence not costly to do so)
    samples = np.asarray(samples)

    from scipy.special import digamma, loggamma, logsumexp

    k_max = integration_params.get("k_max", 100)

    if continuous_prior:
        # integration domain in log scale
        int_domain_lo = integration_params.get("int_domain_lo", -10)
        int_domain_up = integration_params.get("int_domain_up", 10)

    grid_lo = root_search_params.get("grid_lo", -10)
    grid_up = root_search_params.get("grid_up", 10)
    grid_size = root_search_params.get("grid_size", 10)
    n_try = root_search_params.get("n_try", 7)

    n = len(samples)
    nn = np.linspace(1, n, n)

    # Sample mean
    S = np.cumsum(np.log(samples / 2))
    mu_hat = np.cumsum(samples) / nn

    if n_end == -1:
        n_end = n

    bounds = init_nan(n, side)

    log_delta = np.log(1 / delta)

    dynamic_tuning = c_func is not None

    if continuous_prior:
        kk = np.exp(np.linspace(int_domain_lo, int_domain_up, k_max))
    else:
        kk = np.linspace(1, k_max, k_max)

    for t in range(n_start, n_end):
        if dynamic_tuning:
            c_ = c_func(t)
        else:
            c_ = c

        if continuous_prior:

            def K(x):
                # Argument in log scale.
                m = np.exp(x)
                # Rectangular integration scheme written in logexpsum
                # for overflow purpose.
                A = logsumexp(
                    -c_ * loggamma(kk[:-1] / 2)
                    + kk[:-1] / 2 * c_ * digamma(m / 2)
                    + np.log(np.diff(kk))
                )
                B = logsumexp(
                    -(t + c_) * loggamma(kk[:-1] / 2)
                    + kk[:-1] / 2 * (c_ * digamma(m / 2) + S[t])
                    + np.log(np.diff(kk))
                )
                return t * loggamma(m / 2) - m / 2 * S[t] - log_delta - A + B

        else:

            def K(x):
                # Argument in log scale.
                m = np.exp(x)
                # Summation written in logexpsum
                # for overflow purpose.
                A = logsumexp(-c_ * loggamma(kk / 2) + kk / 2 * c_ * digamma(m / 2))
                B = logsumexp(
                    -(t + c_) * loggamma(kk / 2) + kk / 2 * (c_ * digamma(m / 2) + S[t])
                )
                return t * loggamma(m / 2) - m / 2 * S[t] - log_delta - A + B

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
            safe=safe,
        )

        if side == "both":
            if continuous_prior:
                bounds[:, t] = bound_[0], bound_[1]
            else:
                bounds[0, t] = mu_hat[t] - np.ceil(mu_hat[t] - bound_[0])
                bounds[1, t] = -mu_hat[t] + np.floor(mu_hat[t] + bound_[1])
        elif side == "lower":
            if continuous_prior:
                bounds[t] = bound_
            else:
                bounds[t] = mu_hat[t] - np.ceil(mu_hat[t] - bound_)
        elif side == "upper":
            if continuous_prior:
                bounds[t] = bound_
            else:
                bounds[t] = -mu_hat[t] + np.floor(mu_hat[t] + bound_)

    return bounds


def bregman_poisson_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    c: float = 1.0,
    c_func: Callable = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    integration_params: dict = {},
    root_search_params: dict = {},
    **kwargs,
) -> List[float]:
    """Laplace-Bregman uniform concentration bound for Poisson distributions.

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    c: float
    Regularisation parameter.

    c_func: Callable
    Regularisation parameter as a function of sample size.
    (Only a heuristic)

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(sigma  > sigma_hat + U(delta)) < delta
    Lower: P(sigma  < sigma_hat - U(delta)) < delta

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

    from scipy.special import logsumexp

    k_max = integration_params.get("k_max", 100)

    int_domain_lo = integration_params.get("int_domain_lo", -5)
    int_domain_up = integration_params.get("int_domain_up", 5)

    grid_lo = root_search_params.get("grid_lo", -10)
    grid_up = root_search_params.get("grid_up", 10)
    grid_size = root_search_params.get("grid_size", 10)
    n_try = root_search_params.get("n_try", 7)

    n = len(samples)
    nn = np.linspace(1, n, n)

    # Sample mean
    S = np.cumsum(samples)
    mu_hat = S / nn

    if n_end == -1:
        n_end = n

    bounds = init_nan(n, side)

    log_delta = np.log(1 / delta)

    dynamic_tuning = c_func is not None

    theta_k = np.linspace(int_domain_lo, int_domain_up, k_max)

    for t in range(n_start, n_end):
        if dynamic_tuning:
            c_ = c_func(t)
        else:
            c_ = c

        def K(x):
            # Argument in log scale.
            lambda_ = np.exp(x)
            # Rectangular integration scheme written in logexpsum
            # for overflow purpose.
            A = logsumexp(
                -c_ * np.exp(theta_k[:-1])
                + c_ * lambda_ * theta_k[:-1]
                + np.log(np.diff(theta_k))
            )
            B = logsumexp(
                -(t + c_) * np.exp(theta_k[:-1])
                + (S[t] + c_ * lambda_) * theta_k[:-1]
                + np.log(np.diff(theta_k))
            )
            return t * lambda_ - S[t] * np.log(lambda_) - log_delta - A + B

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
        elif side == "lower":
            bounds[t] = bound_
        elif side == "upper":
            bounds[t] = bound_

    return bounds


def bregman_pareto_uniform_bound(
    n_start: int = 1,
    samples: List = [],
    delta: float = 0.05,
    c: float = 1.0,
    c_func: Callable = None,
    side: str = "lower",
    mode: str = "mean",
    n_end: int = -1,
    integration_params: dict = {},
    root_search_params: dict = {},
    **kwargs,
) -> List[float]:
    """Laplace-Bregman uniform concentration bound for Pareto distributions
    supported on [1, +infty).

    n_start: int
    Minimal sample size.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    c: float
    Regularisation parameter.

    c_func: Callable
    Regularisation parameter as a function of sample size.
    (Only a heuristic)

    side: str ('upper', 'lower', 'both')
    Whether to compute an upper or lower confidence bound U(delta) such that
    Upper: P(sigma  > sigma_hat + U(delta)) < delta
    Lower: P(sigma  < sigma_hat - U(delta)) < delta

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

    from scipy.special import loggamma

    grid_lo = root_search_params.get("grid_lo", -10)
    grid_up = root_search_params.get("grid_up", 10)
    grid_size = root_search_params.get("grid_size", 10)
    n_try = root_search_params.get("n_try", 7)

    n = len(samples)
    nn = np.linspace(1, n, n)

    # Sample sum of log
    L = np.cumsum(np.log(samples))
    # Sample mean
    mu_hat = np.cumsum(samples) / nn

    if n_end == -1:
        n_end = n

    bounds = init_nan(n, side)

    log_delta = np.log(1 / delta)

    dynamic_tuning = c_func is not None

    for t in range(n_start, n_end):
        if dynamic_tuning:
            c_ = c_func(t)
        else:
            c_ = c

        def K(x):
            # Argument in log scale.
            a = np.exp(x)

            return (
                -(t + c_ + 1) * np.log(c_ + a * L[t])
                + a * L[t]
                - log_delta
                + c_ * np.log(c_)
                + np.log(t + c_)
                - loggamma(c_)
                + loggamma(t + c_)
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
            arg_transform=lambda x: np.exp(x),
        )

        if side == "both":
            bounds[:, t] = bound_[0], bound_[1]
        elif side == "lower":
            bounds[t] = bound_
        elif side == "upper":
            bounds[t] = bound_

    return bounds


def bregman_gaussian_mean_variance_uniform_bound(
    samples,
    delta: float,
    c: float = 1.0,
    res_m: int = 512,
    res_s: int = 512,
    m_min: float = -5.0,
    m_max: float = 5.0,
    s_min: float = 0.1,
    s_max: float = 5.0,
) -> List[float]:
    """Laplace-Bregman uniform concentration bound for Gaussian distributions
    with unknown mean and variance.

    samples: List
    Empirical samples.

    delta: float
    Confidence level.

    c: float
    Regularisation parameter.

    res_m: int
    Resolution of the mu grid.

    res_s: int
    Resolution of the sigma grid.

    m_min: float
    Minimum value of the mu grid.

    m_max: float
    Maximum value of the mu grid.

    s_min: float
    Minimum value of the sigma grid.

    s_max: float
    Maximum value of the sigma grid.
    """
    # Cast to numpy array if not already the case
    # (np.asarray does nothing if it is, hence not costly to do so)
    samples = np.asarray(samples)

    from scipy.special import loggamma

    n = len(samples)
    mu_hat = np.mean(samples)

    def K(m, s):
        Z = np.sum(((samples - m) / s) ** 2)
        Z_hat = np.sum(((samples - mu_hat) / s) ** 2)
        return (
            -(n + c + 3) / 2 * np.log(n / (n + c) * Z_hat + c / (n + c) * Z + c)
            + 1 / 2 * Z
            - np.log(1 / delta)
            + n / 2 * np.log(2)
            + (c / 2 + 2) * np.log(c)
            - 1 / 2 * np.log(n + c)
            - loggamma((c + 3) / 2)
            + loggamma((n + c + 3) / 2)
        )

    mm = np.linspace(m_min, m_max, res_m)
    ss = np.linspace(s_min, s_max, res_s)

    heatmap = np.empty((res_m, res_s))
    for i, s in enumerate(ss[::-1]):
        for j, m in enumerate(mm):
            heatmap[i, j] = K(m, s)
    return mm, ss, (heatmap < 0).astype(float)
