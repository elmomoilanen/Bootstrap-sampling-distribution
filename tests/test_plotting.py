"""Tests for `Plotting` class."""

from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from sampdist import plotting


@pytest.fixture
def plotter():
    plotter = plotting.Plotting()
    plotter._set_text_field = MagicMock()
    plotter._set_annotation = MagicMock()
    plotter._compute_percentile_ci = MagicMock()
    plotter._compute_percentile_ci.return_value = np.array([1, 3])
    return plotter


@pytest.fixture
def plot_data():
    return {
        "b_stats": np.array([1, 2, 3]),
        "se": 0.5,
        "ci": np.array([1, 3]),
        "actual_stat": 2.0,
        "alpha": 95,
    }


@pytest.fixture
def plot_config():
    return {
        "bins": 10,
        "statistic": "mean",
    }


@patch("sampdist.plotting.plt", autospec=True)
def test_plot_estimates_without_comparison(mock_plt, plotter, plot_data, plot_config):
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    plotter.plot_estimates(plot_data, plot_config)

    mock_plt.subplots.assert_called_once_with(figsize=(8, 6))
    mock_plt.style.context.assert_called_once_with("default")
    mock_plt.show.assert_called_once()
    mock_plt.close.assert_called_once()

    plotter._compute_percentile_ci.assert_not_called()
    assert plotter._set_text_field.call_count == 3
    assert plotter._set_annotation.call_count == 3


@patch("sampdist.plotting.plt", autospec=True)
def test_plot_estimates_with_comparison(mock_plt, plotter, plot_data, plot_config):
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    plotter.plot_estimates(plot_data, plot_config, plot_comparison=True)

    mock_plt.subplots.assert_called_once_with(figsize=(8, 6))
    mock_plt.style.context.assert_called_once_with("default")
    mock_plt.show.assert_called_once()
    mock_plt.close.assert_called_once()

    plotter._compute_percentile_ci.assert_called_once_with(plot_data["b_stats"], plot_data["alpha"])
    assert plotter._set_text_field.call_count == 4
    assert plotter._set_annotation.call_count == 4
