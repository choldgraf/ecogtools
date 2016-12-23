"""Useful statistical functions."""

import numpy as np
from scipy import stats, linalg
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import scale


__all__ = ['bootstrap_coefficients']


def bootstrap_coefficients(x, y, n_boots=1000, fit_intercept=True):
    """Bootstreap coefficients for a linear model."""
    # Bootstrap coefficient for slope
    n_samples, n_features = x.shape
    if fit_intercept is True:
        n_features += 1
        x = np.hstack([x, np.ones([x.shape[0], 1])])

    coefs = np.zeros([n_boots, n_features])
    for iboot in range(n_boots):
        ixs = np.random.randint(0, n_samples, n_samples)
        i_x = x[ixs]
        i_y = y[ixs]
        w, res, _, _ = np.linalg.lstsq(i_x, i_y)
        coefs[iboot] = w.squeeze()
    coefs = np.array(coefs)
    return coefs
