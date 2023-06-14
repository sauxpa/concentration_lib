import numpy as np
from concentration_lib import (
    bentkus_pinelis_std_dev_bound,
    bentkus_std_dev_bound_crude,
    chi2_std_dev_bound,
    chi2_zero_mean_std_dev_bound,
    hoeffding_std_dev_bound,
    maurer_pontil_std_dev_bound,
)


N = 100
DELTA = 0.05
SIGMA = 1.0
MU = 0.5


def test_bentkus_pinelis_std_dev_bound(seed: int = 0):
    rng = np.random.default_rng(seed)
    samples = (rng.random(size=N) < MU).astype(float)
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = bentkus_pinelis_std_dev_bound(
                samples,
                DELTA,
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


def test_bentkus_std_dev_bound_crude(seed: int = 0):
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = bentkus_std_dev_bound_crude(
                N,
                DELTA,
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


def test_chi2_std_dev_bound(seed: int = 0):
    rng = np.random.default_rng(seed)
    samples = MU + rng.normal(size=N) * SIGMA
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = chi2_std_dev_bound(samples, DELTA, mode=mode, side=side)
            if side == "both":
                assert not (np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not (np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not (np.isnan(bound) or np.isinf(bound))


def test_chi2_zero_mean_std_dev_bound(seed: int = 0):
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=N) * SIGMA
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = chi2_zero_mean_std_dev_bound(
                samples,
                DELTA,
                mode=mode,
                side=side,
            )
            if side == "both":
                assert not (np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not (np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not (np.isnan(bound) or np.isinf(bound))


def test_hoeffding_std_dev_bound(seed: int = 0):
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = hoeffding_std_dev_bound(
                N,
                DELTA,
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


def test_maurer_pontil_std_dev_bound(seed: int = 0):
    for mode in ["sum", "mean"]:
        for side in ["lower", "upper", "both"]:
            bound = maurer_pontil_std_dev_bound(
                N,
                DELTA,
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
