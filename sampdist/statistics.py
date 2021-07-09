"""Implements some commonly used statistics.

These statistical functions are defined to be compatible
with requirements of the `SampDist` class in `sampling` module
and hence they are mainly targeted to be used alongside of that class.

When using the `SampDist` class one may use directly statistical functions
defined here or implement totally new ones, whatever the use case requires.
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
)

logger = logging.getLogger(__name__)


def mean(x: np.ndarray):
    return np.mean(x, axis=1)


def geom_mean(x: np.ndarray):
    return np.exp(np.log(x).mean(axis=1))


def harm_mean(x: np.ndarray):
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


def median_abs_dev(x: np.ndarray):
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
    x_std = np.sqrt(np.sum(x_mx ** 2, axis=1))

    return x_cov / (x_std[:, 0] * x_std[:, 1])
