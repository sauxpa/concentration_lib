import numpy as np
from concentration_lib import (
    bentkus_bound,
    bercu_touati_bound,
    bercu_touati_zero_mean_bound,
    bernstein_bound,
    chernoff_subgaussian_bound,
    gaussian_bound,
    hoeffding_bound,
    symmetric_bentkus_efron_bound,
)


N = 100
DELTA = 0.05
SIGMA = 1.0
MU = 0.5


def test_bentkus_bound(seed: int = 0):
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = bentkus_bound(
                N,
                DELTA,
                np.sqrt(MU * (1 - MU)),
                upper_bound=1.0,
                lower_bound=0.0,
                mode=mode,
                side=side,
            )
            if side == "both":
                assert not (np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not (np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not (np.isnan(bound) or np.isinf(bound))


def test_bercu_touati_bound(seed: int = 0):
    rng = np.random.default_rng(seed)
    samples = MU + rng.normal(size=N) * SIGMA
    a = 9 / 16
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = bercu_touati_bound(
                samples,
                DELTA,
                SIGMA,
                a,
                mode=mode,
                side=side,
            )
            if side == "both":
                assert not (np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not (np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not (np.isnan(bound) or np.isinf(bound))


def test_bercu_touati_zero_mean_bound(seed: int = 0):
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=N) * SIGMA
    a = 9 / 16
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = bercu_touati_zero_mean_bound(
                samples,
                DELTA,
                SIGMA,
                a,
                mode=mode,
                side=side,
            )
            if side == "both":
                assert not (np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not (np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not (np.isnan(bound) or np.isinf(bound))


def test_bernstein_bound(seed: int = 0):
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = bernstein_bound(
                N,
                DELTA,
                np.sqrt(MU * (1 - MU)),
                upper_bound=1.0,
                lower_bound=0.0,
                mode=mode,
                side=side,
            )
            if side == "both":
                assert not (np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not (np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not (np.isnan(bound) or np.isinf(bound))


def test_chernoff_subgaussian_bound(seed: int = 0):
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = chernoff_subgaussian_bound(
                N,
                DELTA,
                SIGMA,
                mode=mode,
                side=side,
            )
            if side == "both":
                assert not (np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not (np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not (np.isnan(bound) or np.isinf(bound))


def test_hoeffding_bound(seed: int = 0):
    msg = "Chernoff and Hoefdding are not equivalent in the bounded case."
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = hoeffding_bound(
                N,
                DELTA,
                upper_bound=1.0,
                lower_bound=0.0,
                mode=mode,
                side=side,
            )
            bound_chernoff = chernoff_subgaussian_bound(
                N,
                DELTA,
                0.5,
                mode=mode,
                side=side,
            )
            if side == "both":
                assert bound[0] == bound_chernoff[0], msg
                assert bound[1] == bound_chernoff[1], msg
                assert not (np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not (np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert bound == bound_chernoff, msg
                assert not (np.isnan(bound) or np.isinf(bound))


def test_gaussian(seed: int = 0):
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = gaussian_bound(N, DELTA, SIGMA, mode=mode, side=side)
            if side == "both":
                assert not (np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not (np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not (np.isnan(bound) or np.isinf(bound))


def test_symmetric_bentkus_efron_bound(seed: int = 0):
    rng = np.random.default_rng(seed)
    samples = MU + rng.normal(size=N) * SIGMA
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = symmetric_bentkus_efron_bound(samples, DELTA, mode=mode, side=side)
            if side == "both":
                assert not (np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not (np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not (np.isnan(bound) or np.isinf(bound))
