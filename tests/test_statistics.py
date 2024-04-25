"""Tests for some statistical functions implemented in `statistics` module."""

from math import isclose

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from bootstrap_sampling_distribution import (
    trimmed_mean_factory,
    quantile_factory,
    corr_pearson,
    corr_spearman,
)

rg = np.random.default_rng()


def test_trimmed_mean():
    trim_mean_func = trimmed_mean_factory(cut=0.1)

    x = np.arange(100).reshape(1, -1)
    res = trim_mean_func(x)[0]

    assert isclose(res, 49.5)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_trimmed_mean_nan():
    trim_mean_func = trimmed_mean_factory(cut=0.5)

    x = np.arange(100).reshape(1, -1)
    res = trim_mean_func(x)[0]

    assert np.isnan(res)


def test_quantile():
    quantile_func = quantile_factory(q=1)

    x = np.arange(100).reshape(1, -1)
    res = quantile_func(x)

    assert isclose(res[0], 99)


def test_pearson_corr():
    X = np.array([[-1, 1], [0, 0], [1, -1]])

    assert isclose(
        pearsonr(X[:, 0], X[:, 1])[0],
        corr_pearson(X[np.newaxis, :, :])[0],
        abs_tol=0.0000001,
    )


def test_pearson_corr_other():
    a = np.arange(50)
    X = np.column_stack((a, a[::-1]))

    assert isclose(
        pearsonr(X[:, 0], X[:, 1])[0],
        corr_pearson(X[np.newaxis, :, :])[0],
        abs_tol=0.0000001,
    )


def test_pearson_corr_no_linear_correlation():
    x = np.linspace(-1, 1, 50)
    y = np.sqrt(1 - x**2)
    X = np.column_stack((x, y))

    assert isclose(
        pearsonr(X[:, 0], X[:, 1])[0],
        corr_pearson(X[np.newaxis, :, :])[0],
        abs_tol=0.000000001,
    )


def test_pearson_corr_return_shape():
    result_corr_values = 5
    x = rg.normal(size=(result_corr_values, 10, 2))

    assert corr_pearson(x).size == result_corr_values


def test_spearman_corr():
    X = np.array([[-1, 1], [0, 0], [1, -1]])

    assert isclose(
        spearmanr(X[:, 0], X[:, 1])[0],
        corr_spearman(X[np.newaxis, :, :])[0],
        abs_tol=0.0000001,
    )


def test_spearman_corr_other():
    a = np.arange(50)
    X = np.column_stack((a, a[::-1]))

    assert isclose(
        spearmanr(X[:, 0], X[:, 1])[0],
        corr_spearman(X[np.newaxis, :, :])[0],
        abs_tol=0.0000001,
    )


def test_spearman_corr_return_shape():
    result_corr_values = 5
    x = rg.normal(size=(result_corr_values, 10, 2))

    assert corr_spearman(x).size == result_corr_values
