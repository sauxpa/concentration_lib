import numpy as np
from concentration_lib import chi2_std_dev_bound
from concentration_lib import hoeffding_std_dev_bound
from concentration_lib import maurer_pontil_std_dev_bound
from concentration_lib import bentkus_std_dev_bound_crude
from concentration_lib import bentkus_pinelis_std_dev_bound
from concentration_lib import chi2_zero_mean_std_dev_bound
from concentration_lib import empirical_chernoff_zero_mean_std_dev_bound


N = 100
DELTA = 0.05
SIGMA = 1.0
MU = 0.5


def test_chi2_std_dev_bound(seed: int = 0):
    np.random.seed(seed)
    samples = MU + np.random.randn(N) * SIGMA
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = chi2_std_dev_bound(
                samples, DELTA, mode=mode, side=side
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))


def test_hoeffding_std_dev_bound(seed: int = 0):
    np.random.seed(seed)
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = hoeffding_std_dev_bound(
                N, DELTA, upper_bound=1.0, lower_bound=0.0,
                mode=mode, side=side,
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))


def test_maurer_pontil_std_dev_bound(seed: int = 0):
    np.random.seed(seed)
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = maurer_pontil_std_dev_bound(
                N, DELTA, upper_bound=1.0, lower_bound=0.0,
                mode=mode, side=side,
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))


def test_bentkus_std_dev_bound_crude(seed: int = 0):
    np.random.seed(seed)
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = bentkus_std_dev_bound_crude(
                N, DELTA, upper_bound=1.0, lower_bound=0.0,
                mode=mode, side=side,
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))


def test_bentkus_pinelis_std_dev_bound(seed: int = 0):
    np.random.seed(seed)
    samples = (np.random.rand(N) < MU).astype(float)
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = bentkus_pinelis_std_dev_bound(
                samples, DELTA, upper_bound=1.0, lower_bound=0.0,
                mode=mode, side=side,
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))


def test_chi2_zero_mean_std_dev_bound(seed: int = 0):
    np.random.seed(seed)
    samples = np.random.randn(N) * SIGMA
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = chi2_zero_mean_std_dev_bound(
                samples, DELTA, mode=mode, side=side,
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))


def test_empirical_chernoff_zero_mean_std_dev_bound(seed: int = 0):
    np.random.seed(seed)
    samples = np.random.randn(N) * SIGMA
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = empirical_chernoff_zero_mean_std_dev_bound(
                samples, DELTA, rho=1.0, mode=mode, side=side,
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))
