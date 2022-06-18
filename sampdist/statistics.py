"""Implements some commonly used statistics.

Implemented functions:
- mean
- geometric_mean
- harmonic_mean
- trimmed_mean_factory
- median
- maximum
- minimum
- skew
- kurtosis
- interquartile_range
- standard_deviation
- median_absolute_deviation
- quantile_factory
- corr_pearson
- corr_spearman

These statistical functions are defined to be compatible
with requirements of the `SampDist` class in `sampling` module
and hence they are mainly targeted to be used alongside of that class.

Most of the functions use setting `axis=1` meaning that in order to
use them for a typical data of shape n x 1, the data must be reshaped
to a shape of 1 x n, which can be done e.g. by NumPy's reshape(1,-1).
In addition, notice that for a data of shape n x p, taking the transpose
produces the correct shape.

Functions ending with the `factory` suffix are actually closure factory
functions indicating that they create and return closures upon function
calls. Closures are simply functions that remember their enclosing scopes,
here e.g. `trimmed_mean_factory` returns a closure `trimmed_mean` that
uses the parameter `cut` from the enclosing scope.

Two of the functions, corr_person and corr_spearman, require the data
to be in shape n x 2 but they must be called with an additional dimension.
E.g. for a data X of shape n x 2 one should call the Pearson's correlation
by `corr_pearson(X[np.newaxis,:,:])`.
"""
import logging

import numpy as np

from scipy.stats import (
    hmean,
    trim_mean,
    skew as scipy_skew,
    kurtosis as scipy_kurtosis,
    iqr,
    median_abs_deviation,
    rankdata,
)

logger = logging.getLogger(__name__)


def mean(x: np.ndarray):
    return np.mean(x, axis=1)


def geometric_mean(x: np.ndarray):
    return np.exp(np.log(x).mean(axis=1))


def harmonic_mean(x: np.ndarray):
    return hmean(x, axis=1)


def trimmed_mean_factory(cut: float):
    if not (cut > 0 and cut < 1):
        raise ValueError("Give `cut` within (0,1)")

    def trimmed_mean(x: np.ndarray):
        return trim_mean(x, cut, axis=1)

    return trimmed_mean


def median(x: np.ndarray):
    return np.median(x, axis=1)


def maximum(x: np.ndarray):
    return np.max(x, axis=1)


def minimum(x: np.ndarray):
    return np.min(x, axis=1)


def skew(x: np.ndarray):
    return scipy_skew(x, axis=1, bias=False)


def kurtosis(x: np.ndarray):
    return scipy_kurtosis(x, axis=1, bias=False)


def interquartile_range(x: np.ndarray):
    return iqr(x, axis=1)


def standard_deviation(x: np.ndarray):
    return np.std(x, axis=1, ddof=1)


def median_absolute_deviation(x: np.ndarray):
    return median_abs_deviation(x, axis=1)


def quantile_factory(q: float):
    if not (q >= 0 and q <= 1):
        raise ValueError("Give `q` within (0,1)")

    def quantile(x: np.ndarray):
        return np.quantile(x, q, axis=1)

    return quantile


def corr_pearson(x: np.ndarray):
    x_mx = x - np.mean(x, axis=1)[:, np.newaxis]
    x_cov = np.sum(x_mx[:, :, 0] * x_mx[:, :, 1], axis=1)
    x_std = np.sqrt(np.sum(x_mx**2, axis=1))

    return x_cov / (x_std[:, 0] * x_std[:, 1])


def corr_spearman(x: np.ndarray):
    return corr_pearson(rankdata(x, method="average", axis=1))
