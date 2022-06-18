"""Implements sampling distribution functionality."""
import logging
from typing import Union

import numpy as np
from scipy.stats import norm

from .plotting import Plotting

logger = logging.getLogger(__name__)


class SampDist:
    """Estimates the sampling distribution of a chosen statistic using random sampling with replacement.

    This method enables estimation of precision of the statistic and is convenient in circumstances
    where analytical evaluation of the precision is either very difficult or totally impossible.

    In total, the sampling (bootstrap) distribution, standard error (SE) and bias-corrected and accelerated
    confidence interval (BCa) of the statistic are computed. Standard error is the standard
    deviation of the statistic under the sampling distribution and BCa confidence interval
    adjusts for bias and skewness in the bootstrap distribution.

    Notice that few assumptions are taken place, namely that empirical distribution function of the
    observed data serves as a representative approximating distribution (model of population)
    from which the samples are drawn and that observations are from an iid population.
    Hence the estimated results might be misleading if the unknown population distribution
    is not well represented by the empirical distribution or does not have a finite variance.
    Of cource, these aspects are rarely known in advance.

    As a further warning notice, be cautious to use a redundantly high number of resamples.
    This may cause memory issues if dimensionality of the data sample equals two (passed as a matrix)
    or improvements in estimation results might be totally negligible.

    Parameters
    ----------
    statistic: callable
        One or multi-dimensional function, e.g. mean, median, kurtosis,
        correlation or any other function of data that takes numerical data as input.
        Passed 1d functions must be defined with axis == 1, i.e. they must operate row-wise.
        Multi-dimensional case is more complicated as these functions must accept 3d tensor-kind
        inputs, please look at the Example section below for further advice. Also the `statistics`
        module contains few examples, e.g. Pearson and Spearman's correlations.

    Kwargs
    ------
    alpha: numeric
        Coverage level of the confidence interval.

    actual_stat: numeric
        Observed value of the statistic, computed initially with the
        passed data.

    b_stats: NumPy array
        Bootstap (sampling) distribution of the statistic.

    se: numeric
        Standard error of the statistic, i.e. standard deviation
        of the statistic under the sampling distribution.

    ci: NumPy array:
        Bias-corrected and accelerated (BCa) confidence interval of the statistic.

    smooth_bootstrap: bool
        If True, add random noise to each bootstrap sample in order to reduce
        the discreteness of the bootstrap distribution. As a default, equals
        to False.

    plot_style: str
        Pyplot plotting style. Plotting class in module `plotting` defines
        the allowed styles and the passed style will be checked against them.
        If the passed style is not allowed, an exception is raised.

    Examples
    --------
    Example for an one-dimensional statistic, a quantile. Estimation is run simultaneously
    for two of the columns of data X (with total shape N x P). NumPy's quantile method
    is used directly, but the version from statistics module could also be used.

    >>> import numpy as np
    >>> from sampdist import SampDist
    >>> def quantile(x): return np.quantile(x, q=0.1, axis=1)
    >>> samp = SampDist(quantile, alpha=99, smooth_bootstrap=True)
    >>> samp.estimate(X[:, [0,2]]) # estimate sampling distribution for columns 0 and 2
    >>> samp.plot(column=0)
    >>> samp.plot(column=1)

    Example for a two-dimensional statistic, Pearson's linear correlation. Estimation is run
    for two of the columns of data X and their correlation coefficient is received as a return
    value. Correlation function is imported from the statistics module where it has been
    implemented to accept 3d data as input.

    >>> from sampdist import SampDist
    >>> from sampdist.statistics import corr_pearson
    >>> samp = SampDist(corr_pearson) # use default alpha 95 and not smoothed bootstrap
    >>> samp.estimate(X[:, :2], multid=True) # estimate sampling distribution of linear correlation for columns 0 and 1
    >>> samp.se, samp.ci # access the computed standard error and BCa confidence interval
    >>> samp.plot()

    References
    ----------
    Efron, B. 1979. Bootstrap methods: Another look at the jackknife. Ann. Statist., 7(1), 1-26.
    Efron, B. 1987. Better bootstrap confidence intervals. J. Amer. Statist. Assoc., 82(397), 171-200.
    Efron, B., Hastie T. 2016. Computer age statistical inference. Cambridge University Press.
    """

    def __init__(self, statistic, **kwargs):
        self.actual_stat = np.nan
        self.b_stats = np.nan
        self.se = np.nan
        self.ci = np.nan

        self._acceleration = np.nan
        self._z0 = np.nan

        self._alpha_min = 90
        self._alpha_max = 99.9999

        self.statistic = statistic
        self.alpha = kwargs.get("alpha", 95)

        self.smooth_bootstrap = kwargs.get("smooth_bootstrap", False)

        self._plot_style = kwargs.get("plot_style")
        general_plot_config = (self._plot_style and {"plot_style_sheet": self._plot_style}) or {}

        self._plot_obj = Plotting(**general_plot_config)

    def __repr__(self):
        return f"{self.__class__.__name__}(statistic={self.statistic.__name__}, alpha={self.alpha})"

    def _reset_estimates(self):
        self.actual_stat = np.nan
        self.b_stats = np.nan
        self.se = np.nan
        self.ci = np.nan

        self._acceleration = np.nan
        self._z0 = np.nan

    @property
    def statistic(self):
        return self._statistic

    @statistic.setter
    def statistic(self, func):
        if not callable(func):
            raise TypeError("Statistic must be callable")

        self._statistic = func
        self._reset_estimates()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Alpha must be numerical")

        if not (value >= self._alpha_min and value <= self._alpha_max):
            raise ValueError(f"Alpha must be within [{self._alpha_min},{self._alpha_max}]")

        self._alpha = value

    def _test_statistic_validity(self, multid):
        test_array = np.ones(2) if not multid else np.ones((2, 2))

        logger.debug(
            "testing statistic %s with data of shape %s", self.statistic.__name__, test_array.shape
        )

        try:
            self.statistic(test_array)

        except (np.AxisError, IndexError) as err:
            logger.debug("test resulted error `%s`, good to go", err)

        else:
            dim = f"{'multi' if multid else 'one'}-dimensional"
            raise ValueError(f"Statistic is ill-defined for {dim} data, see the instructions")

    def _compute_actual_statistic(self, sample, multid):
        if multid:
            self.actual_stat = self.statistic(sample[np.newaxis, :, :])

        else:
            if sample.ndim == 1:
                self.actual_stat = self.statistic(sample.reshape(1, -1))
            else:
                self.actual_stat = self.statistic(sample.T)

        logger.debug("actual statistic value computed")

    def _draw_bootstrap_samples(self, sample, iterations, multid):
        rg = np.random.default_rng()

        if sample.ndim == 1 or multid:
            dimensions = (iterations, sample.shape[0])
            row_indices = rg.integers(0, sample.shape[0], size=dimensions)

            data = sample[row_indices]

        else:
            dimensions = tuple([iterations] + list(sample.shape))
            row_indices = rg.integers(0, sample.shape[0], size=dimensions)

            data = sample[row_indices, np.arange(sample.shape[1])]

        if self.smooth_bootstrap:
            logger.debug("using smooth bootstrap, thus add random noise to data")
            sigma = 1 / np.sqrt(sample.shape[0])
            data = data + rg.normal(scale=sigma, size=data.shape)

        logger.debug("bootstrap samples drawn")

        return data

    def _get_estimate_of_acceleration(self, sample, multid):
        stat_count = sample.shape[0]

        logger.debug("starting jackknife calculations for estimate of acceleration")

        if sample.ndim == 1:
            jackknife_stats = np.array(
                [self.statistic(np.delete(sample, idx).reshape(1, -1)) for idx in range(stat_count)]
            )

        else:
            if multid:
                jackknife_stats = np.array(
                    [
                        self.statistic(np.delete(sample, idx, axis=0)[np.newaxis, :, :])
                        for idx in range(stat_count)
                    ]
                )
            else:
                jackknife_stats = np.array(
                    [self.statistic(np.delete(sample, idx, axis=0).T) for idx in range(stat_count)]
                )

        logger.debug("jackknife statistics calculated")

        difference = jackknife_stats - np.mean(jackknife_stats, axis=0)
        denominator = np.sum(difference**2, axis=0) ** 1.5
        nominator = np.sum(difference**3, axis=0)

        acceleration = np.full(denominator.shape, np.inf)
        almost_zero = denominator < 1e-300
        acceleration[~almost_zero] = 1 / 6 * nominator[~almost_zero] / denominator[~almost_zero]

        logger.debug("estimate of acceleration calculated")

        self._acceleration = acceleration

    def _get_estimate_of_bias_correction(self):
        p0 = np.sum(self.b_stats <= self.actual_stat, axis=0) / self.b_stats.shape[0]

        if self.b_stats.ndim == 1:
            p0 = np.array([p0])

        invalid_p0 = (p0 == 0) | (p0 == 1)

        if np.any(invalid_p0, axis=0):
            invalid_cols = invalid_p0.nonzero()[0]

            raise ValueError(
                f"BCa confidence interval cannot be computed for columns {invalid_cols}"
            )

        self._z0 = norm.ppf(p0)

    def _get_bca_confidence_intervals(self, alpha):
        self._get_estimate_of_bias_correction()

        x0 = self._z0 + (self._z0 + norm.ppf(alpha)) / (
            1 - self._acceleration * (self._z0 + norm.ppf(alpha))
        )
        p = norm.cdf(x0)
        p_rounded = np.round(p * 100, decimals=3)

        q_values = np.percentile(self.b_stats, p_rounded, axis=0, interpolation="linear")

        if q_values.ndim == 2:
            q_values = np.diagonal(q_values)

        logger.debug("confidence interval point for alpha %s calculated", alpha)

        return q_values

    def _compute_estimates(self, sample, iterations, multid):
        bootstrapped_data = self._draw_bootstrap_samples(sample, iterations, multid)

        self.b_stats = self.statistic(bootstrapped_data)
        self.se = np.std(self.b_stats, ddof=1, axis=0)

        if self.b_stats.ndim == 1:
            self.se = np.array([self.se])

        self._get_estimate_of_acceleration(sample, multid)

        self.ci = np.array(
            [
                self._get_bca_confidence_intervals(0 + ((100 - self.alpha) / 2) / 100),
                self._get_bca_confidence_intervals(1 - ((100 - self.alpha) / 2) / 100),
            ]
        )

    def plot(self, column: int = 0, bins: Union[int, str] = "sqrt", comparison: bool = False):
        """Plot bootstrap distribution with SE, observed value and BCa CIs.

        Distribution is represented by a histogram, roughly illustrating the density of
        the underlying distribution. Standard error, observed value and BCa confidence interval values
        are represented on the right side of the histogram. In addition, black and red triangles
        illustrates the observed value and confidence interval points, respectively, on the histogram.

        Parameters
        ----------
        column: int
            Number of the column to be plotted, defaults to first column of the data.

        bins: int or str
            Count of consecutive and non-overlapping intervals for the x-axis of the histogram.
            Default value is set to be "sqrt", indicating a particular binning strategy.
            If given as integer, it represents precisely the interval count. On the contrary,
            when given as str value, it must describe a binning strategy, which must be
            one of the following: "auto","sturges","fd","doane","scott","rice" or "sqrt".

        comparison: bool
            Defaults to False, when the plotted histogram will not have naive percentile CIs
            included in addition to the BCa confidence interval. If True, the percentile
            confidence interval is plotted alongside the BCa interval.
        """
        if not isinstance(column, int):
            raise TypeError("Column must be an integer")

        if not isinstance(self.b_stats, np.ndarray):
            raise ValueError("Nothing to plot yet, please run estimation step first")

        if self.b_stats.ndim > 1:
            if np.abs(column) >= self.b_stats.shape[1]:
                raise IndexError(
                    f"Column {column} out of bounds for axis of size {self.b_stats.shape[1]}"
                )

        if isinstance(bins, int):
            if not (bins > 0 and bins <= self.b_stats.shape[0]):
                raise ValueError(f"Bins must be defined within [1,{self.b_stats.shape[0]}]")

        elif isinstance(bins, str):
            allowed_strategies = {"auto", "sturges", "fd", "doane", "scott", "rice", "sqrt"}
            if bins not in allowed_strategies:
                raise ValueError(f"Bins strategy must be one of {', '.join(allowed_strategies)}")

        else:
            raise TypeError("Bins must be an integer or a string describing the binning strategy")

        plot_data = {
            "b_stats": self.b_stats if self.b_stats.ndim == 1 else self.b_stats[:, column],
            "se": self.se[column],
            "ci": self.ci[:, column],
            "actual_stat": self.actual_stat[column],
            "alpha": self.alpha,
        }
        plot_config = {
            "bins": bins,
            "statistic": self.statistic.__name__,
        }

        self._plot_obj.plot_estimates(plot_data, plot_config, comparison)

    def estimate(self, sample: np.ndarray, iterations: int = 5000, multid: bool = False):
        """Compute sampling distribution, SE and BCa confidence intervals for given statistic.

        Prior any computations, validity of the statistic is evaluated. It must apply
        correctly either to one- or two-dimensional data.

        Parameters
        ----------
        sample: numpy.ndarray
            Sample representing the observed data, ndim = 1 or = 2.
            E.g., given data matrix X, both X[:,col] and X[:,[col_1:col_N]] are valid inputs.

        iterations: int
            Number of (bootstrap) resamples to be drawn.

        multid: bool
            False, if the statistic should apply one-dimensionally to data, i.e. it
            would provide one return value for one-dimensional data and p values for
            two-dimensional data (a matrix) with p initial attribute columns. Consider
            e.g. statistics median or 90 percentile in this case. If set to `True`,
            refers to multi-dimensional case where the statistic would produce one
            return value for a two-dimensional data. Consider e.g. Pearson or Spearman's
            correlation.
        """
        if not isinstance(sample, np.ndarray):
            raise TypeError("Sample must be a NumPy array, type `numpy.ndarray`")

        if sample.ndim > 2:
            raise ValueError("No implementation for a sample with dimensionality > 2")

        if not (isinstance(iterations, int) and iterations > 0):
            raise ValueError("Iterations must be a positive integer")

        if multid and sample.ndim != 2:
            raise ValueError("Multidimensional statistic requires two-dimensional data")

        self._test_statistic_validity(multid)

        self._compute_actual_statistic(sample, multid)
        self._compute_estimates(sample, iterations, multid)
