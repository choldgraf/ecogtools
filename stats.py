"""Useful statistical functions."""

import numpy as np
from scipy import stats, linalg
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import scale


__all__ = ['partial_corr',
           'bootstrap_coefficients']


def partial_corr(C, do_scale=False):
    """
    Returns the sample linear partial correlation coefficients between pairs
    of variables in C, controlling for the remaining variables in C.

    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken
        as a variable
    do_scale : bool
        Whether to scale each column of C to mean==0 and variance==1
        before calculations

    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j]
        controlling for the remaining variables in C.

    Information
    -----------
    Partial Correlation in Python (clone of Matlab's partialcorr)
    This uses the linear regression approach to compute the partial 
    correlation (might be slow for a huge number of variables). The 
    algorithm is detailed here:
        http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    Taking X and Y two variables of interest and Z the matrix with all the
    variable minus {X, Y}, the algorithm can be summarized as
        1) perform a normal linear least-squares regression with X as the
            target and Z as the predictor
        2) calculate the residuals in Step #1
        3) perform a normal linear least-squares regression with Y as the
            target and Z as the predictor
        4) calculate the residuals in Step #3
        5) calculate the correlation coefficient between the residuals from
            Steps #2 and #4
        The result is the partial correlation between X and Y while
            controlling for the effect of Z

    Date: Nov 2014
    Author: Fabian Pedregosa-Izquierdo, f@bianp.net
    Testing: Valentina Borghesani, valentinaborghesani@gmail.com
    URL: https://gist.github.com/fabianp/9396204419c7b638d38f
    """

    C = np.asarray(C)
    if do_scale is True:
        C = scale(C, axis=0)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


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
