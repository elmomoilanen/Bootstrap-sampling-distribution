"""Tests for `SampDist` class."""

from math import isclose

import numpy as np
import pytest

from bootstrap_sampling_distribution import SampDist

from bootstrap_sampling_distribution import (
    mean,
    geometric_mean,
    harmonic_mean,
    trimmed_mean_factory,
    median,
    maximum,
    minimum,
    skew,
    kurtosis,
    interquartile_range,
    standard_deviation,
    median_absolute_deviation,
    quantile_factory,
    corr_pearson,
    corr_spearman,
)

from bootstrap_sampling_distribution.errors import StatisticError, BcaError

# all one dimensional statistics except those behind factory function
all_one_dim_statistics = (
    mean,
    geometric_mean,
    harmonic_mean,
    median,
    maximum,
    minimum,
    skew,
    kurtosis,
    interquartile_range,
    standard_deviation,
    median_absolute_deviation,
)

all_multi_dim_statistics = (corr_pearson, corr_spearman)

# init random number generator
rg = np.random.default_rng()


def test_setting_statistic_property():
    samp = SampDist(mean)
    samp.actual_stat = np.array([0])

    for statistic in all_one_dim_statistics:
        name_statistic = statistic.__name__
        samp.statistic = statistic

        assert samp.statistic.__name__ == name_statistic
        assert np.isnan(samp.actual_stat)


def test_setting_not_callable_statistic():
    not_callable_func = None

    with pytest.raises(TypeError, match=r"Statistic must be callable"):
        SampDist(not_callable_func)


def test_setting_alpha_property():
    alpha_values = (90, 90.00, 95.00001, 99, 99.9999)

    for alpha in alpha_values:
        samp = SampDist(median, alpha=alpha)

        assert isclose(samp.alpha, alpha)


def test_setting_not_numeric_alpha():
    not_valid_alpha = "90"

    with pytest.raises(TypeError, match=r"Alpha must be an integer or float"):
        SampDist(median, alpha=not_valid_alpha)


def test_setting_not_within_bounds_alpha():
    not_valid_alpha_values = (-1, 0, 89, 89.99, 99.99999, 100)

    for alpha in not_valid_alpha_values:
        with pytest.raises(ValueError, match=r"Alpha must be within.*"):
            SampDist(median, alpha=alpha)


def test_statistic_validity_check_one_dim_statistics():
    for statistic in all_one_dim_statistics:
        samp = SampDist(statistic)

        # validity checker should catch Axis or Index error and return None

        assert samp._test_statistic_validity(multid=False) is None


def test_statistic_validity_check_factories():
    trim_mean_func = trimmed_mean_factory(cut=0.5)
    quantile_func = quantile_factory(q=0.5)

    samp = SampDist(trim_mean_func)
    assert samp._test_statistic_validity(multid=False) is None

    samp = SampDist(quantile_func)
    assert samp._test_statistic_validity(multid=False) is None


def test_statistic_validity_check_multi_dim_statistics():
    for statistic in all_multi_dim_statistics:
        samp = SampDist(statistic)

        assert samp._test_statistic_validity(multid=True) is None


def test_statistic_validity_check_error():
    samp = SampDist(lambda x: x)

    # Statistic cannot handle case axis=1
    # Note that passed input data doesn't matter to this error

    with pytest.raises(StatisticError):
        samp.estimate(np.array([1, 2, 3]))


def test_bca_error():
    samp = SampDist(mean, alpha=99)

    # BCa should fail when the data is exactly identical

    with pytest.raises(BcaError):
        samp.estimate(np.array([1, 1, 1, 1]))


def test_sampdist_compute_actual_statistic_for_mean():
    test_data_size = (5, 3)
    test_data = rg.normal(size=test_data_size)

    abs_tol = 0.0000001

    samp = SampDist(mean)
    samp._compute_actual_statistic(test_data[:, 0], multid=False)

    assert isclose(
        np.mean(test_data[:, 0]),
        samp.actual_stat[0],
        abs_tol=abs_tol,
    )

    samp._compute_actual_statistic(test_data, multid=False)
    correct_means = np.mean(test_data, axis=0)

    assert correct_means.shape == samp.actual_stat.shape

    assert all(
        isclose(corr, cand, abs_tol=abs_tol) for corr, cand in zip(correct_means, samp.actual_stat)
    )


def test_sampdist_draw_bootstrap_samples():
    test_data_size = (5, 3)
    test_data = rg.normal(size=test_data_size)

    samp = SampDist(mean, smooth_bootstrap=True)
    iterations = 6

    boot_samples = samp._draw_bootstrap_samples(test_data[:, 0], iterations, multid=False)

    # test that received bootstrap sample has the correct shape
    assert boot_samples.ndim == 2
    assert boot_samples.shape == (iterations, test_data.shape[0])

    boot_samples = samp._draw_bootstrap_samples(test_data, iterations, multid=False)

    assert boot_samples.ndim == 3
    assert boot_samples.shape == (iterations, test_data.shape[0], test_data.shape[1])


def test_sampdist_estimate_api_single_attribute():
    test_data_size = (100, 2)
    test_data = rg.normal(size=test_data_size)

    quantile = quantile_factory(0.95)

    samp = SampDist(quantile, smooth_bootstrap=True)
    samp.estimate(test_data[:, 0])

    assert samp.ci.size == 2
    assert samp.se.size == 1
    assert samp.b_stats.ndim == 1


def test_sampdist_estimate_api_multiple_attributes():
    test_data_size = (100, 3)
    test_data = rg.normal(size=test_data_size)

    quantile = quantile_factory(0.95)

    samp = SampDist(quantile, smooth_bootstrap=True)
    samp.estimate(test_data[:, [0, 2]])

    assert samp.ci.size == 4
    assert samp.se.size == 2
    assert samp.b_stats.ndim == 2
