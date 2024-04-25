"""Implements some commonly used statistics.

Implemented functions
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

The statistical functions are defined to be compatible with requirements
of the `SampDist` class in `sampling` module and hence are mainly targeted
to be used alongside of that class and this library.

Most of the functions (one-dimensional) use parameter `axis=1` meaning
that they must be called with input data of shape 1 x n (or p x n).
One-dimensional means here that the function produces single output (1 x 1)
for every data unit of shape 1 x n and p outputs for data with shape p x n.
In principle, this is exactly as most of the NumPy or Scipy's statistical
functions behave but only the dimensions have been reversed.

Functions ending with the `factory` suffix are actually closure factory
functions indicating that they create and return closures upon function calls.
Closures are simply functions that remember their enclosing scopes, here e.g.
`trimmed_mean_factory` returns a closure `trimmed_mean` that uses the parameter
`cut` from the enclosing scope.

Two of the functions (multidimensional), `corr_pearson` and `corr_spearman`,
must be called with an additional dimension inserted. E.g., for data X with shape
n x 2 one must call the Pearson's correlation function by `corr_pearson(X[np.newaxis,:,:])`.
This will insert one extra dimension to the front, e.g. n x 2 turns to 1 x n x 2.
"""

from typing import Any, Callable
import numpy as np

from scipy.stats import (  # type: ignore[import-untyped]
    hmean,
    trim_mean,
    skew as scipy_skew,
    kurtosis as scipy_kurtosis,
    iqr,
    median_abs_deviation,
    rankdata,
)


def mean(x: np.ndarray) -> Any:
    return np.mean(x, axis=1)


def geometric_mean(x: np.ndarray) -> Any:
    return np.exp(np.log(x).mean(axis=1))


def harmonic_mean(x: np.ndarray) -> Any:
    return hmean(x, axis=1)


def trimmed_mean_factory(cut: float) -> Callable[[np.ndarray], Any]:
    if not (cut > 0 and cut < 1):
        raise ValueError("Give `cut` within (0,1)")

    def trimmed_mean(x: np.ndarray) -> Any:
        return trim_mean(x, cut, axis=1)

    return trimmed_mean


def median(x: np.ndarray) -> Any:
    return np.median(x, axis=1)


def maximum(x: np.ndarray) -> Any:
    return np.max(x, axis=1)


def minimum(x: np.ndarray) -> Any:
    return np.min(x, axis=1)


def skew(x: np.ndarray) -> Any:
    return scipy_skew(x, axis=1, bias=False)


def kurtosis(x: np.ndarray) -> Any:
    return scipy_kurtosis(x, axis=1, bias=False)


def interquartile_range(x: np.ndarray) -> Any:
    return iqr(x, axis=1)


def standard_deviation(x: np.ndarray) -> Any:
    return np.std(x, axis=1, ddof=1)


def median_absolute_deviation(x: np.ndarray) -> Any:
    return median_abs_deviation(x, axis=1)


def quantile_factory(q: float) -> Callable[[np.ndarray], Any]:
    if not (q >= 0 and q <= 1):
        raise ValueError("Give `q` within (0,1)")

    def quantile(x: np.ndarray) -> Any:
        return np.quantile(x, q, axis=1)

    return quantile


def corr_pearson(x: np.ndarray) -> Any:
    x_mx = x - np.mean(x, axis=1)[:, np.newaxis]
    x_cov = np.sum(x_mx[:, :, 0] * x_mx[:, :, 1], axis=1)
    x_std = np.sqrt(np.sum(x_mx**2, axis=1))
    return x_cov / (x_std[:, 0] * x_std[:, 1])


def corr_spearman(x: np.ndarray) -> Any:
    return corr_pearson(rankdata(x, method="average", axis=1))
