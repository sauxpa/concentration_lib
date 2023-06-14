# Concentration bounds library

* concentration_bounds: fixed sample size confidence bounds on the mean.
* concentration_variance_bounds: fixed sample size confidence bounds on the standard deviation.
* empirical_concentration_bounds: fixed sample size confidence bounds with data-dependent estimators rather than fixed parameters (e.g. the variance in Bernstein bound is estimated from the data instead of being prior knowledge).
* uniform_concentration_bounds.py: confidence sequences on the mean, uniformly valid for all (random) sample sizes.
* empirical_uniform_concentration_bounds.py: confidence sequences on the mean, uniformly valid for all (random) sample sizes, with data-dependent estimators rather than fixed parameters.
* uniform_bregman_concentration_bounds.py: confidence sequences for natural parameters of generic exponential families using the associated Bregman divergences.

To install : pip install concentration_lib https://pypi.org/project/concentration-lib/

To test : python -m pytest
