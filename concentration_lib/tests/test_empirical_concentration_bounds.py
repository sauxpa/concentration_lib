import numpy as np
from concentration_lib import empirical_student_bound
from concentration_lib import empirical_bernstein_bound
from concentration_lib import empirical_bentkus_bound
from concentration_lib import empirical_hedged_capital_bound
from concentration_lib import empirical_small_samples_ptlm
from concentration_lib import empirical_symmetric_bentkus_efron_bound


N = 100
DELTA = 0.05
SIGMA = 1.0
MU = 0.5


def test_empirical_student_bound(seed: int = 0):
    np.random.seed(seed)
    samples = MU + np.random.randn(N) * SIGMA
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = empirical_student_bound(
                samples, DELTA, mode=mode, side=side
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))


def test_empirical_bernstein_bound(seed: int = 0):
    np.random.seed(seed)
    samples = (np.random.rand(N) < MU).astype(float)
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = empirical_bernstein_bound(
                samples, DELTA, upper_bound=1.0, lower_bound=0.0,
                mode=mode, side=side,
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))


def test_empirical_bentkus_bound(seed: int = 0):
    np.random.seed(seed)
    samples = (np.random.rand(N) < MU).astype(float)
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = empirical_bentkus_bound(
                samples, DELTA, upper_bound=1.0, lower_bound=0.0,
                mode=mode, side=side,
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))


def test_empirical_hedged_capital_bound(seed: int = 0):
    np.random.seed(seed)
    samples = (np.random.rand(N) < MU).astype(float)
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = empirical_hedged_capital_bound(
                samples, DELTA, upper_bound=1.0, lower_bound=0.0,
                mode=mode, side=side,
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))


def test_empirical_small_samples_ptlm(seed: int = 0):
    np.random.seed(seed)
    samples = (np.random.rand(N) < MU).astype(float)
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = empirical_small_samples_ptlm(
                samples, DELTA, upper_bound=1.0, lower_bound=0.0,
                mode=mode, side=side,
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))


def test_empirical_symmetric_bentkus_efron_bound(seed: int = 0):
    np.random.seed(seed)
    samples = MU + np.random.randn(N) * SIGMA
    for mode in ['sum', 'mean']:
        for side in ['lower', 'upper', 'both']:
            bound = empirical_symmetric_bentkus_efron_bound(
                samples, DELTA, mode=mode, side=side,
                )
            if side == 'both':
                assert not(np.isnan(bound[0]) or np.isinf(bound[0]))
                assert not(np.isnan(bound[1]) or np.isinf(bound[1]))
            else:
                assert not(np.isnan(bound) or np.isinf(bound))
