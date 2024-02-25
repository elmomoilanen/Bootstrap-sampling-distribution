"""Implements sampling distribution functionality."""
from typing import Any, Optional, Union, Callable

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

from .errors import StatisticError, BcaError
from .plotting import Plotting, PlotData


class SampDist:
    """Estimates the sampling distribution of a statistic using random sampling with replacement.

    This method enables estimation of statistical accuracy of the statistic and is
    convenient in circumstances where analytical evaluation of the accuracy is either
    very difficult or totally impossible.

    In total, the sampling (bootstrap) distribution, standard error (SE) and bias-corrected
    and accelerated confidence interval (BCa) of the statistic are computed. Standard error
    is standard deviation of the statistic under the sampling distribution and BCa confidence
    interval adjusts for bias and skewness in the bootstrap distribution.

    Notice that few assumptions are taken place here, namely that empirical distribution
    function of the observed data serves as a representative approximating distribution
    (model of population) from which the samples are drawn and that the observations are
    from an iid population. Thus, estimated results might be misleading if the unknown
    population distribution is not well represented by the empirical distribution or does
    not have a finite variance. Of cource, these aspects are rarely known in advance.

    Furthermore, be cautious to use a redundantly high number of resamples. This may cause
    memory issues if dimensionality of the data sample equals two (passed as a matrix) or
    improvements in estimation results might be totally negligible.

    Parameters
    ----------
    statistic : callable (ndarray) -> Any
        Function that takes a numerical data array as input and produces some
        output (preferably a numerical to make computations to work). Function
        can be one-dimensional (maps 1 x n input to 1 output or p x n to p outputs)
        like mean, median and kurtosis or multidimensional (maps k x n to 1 output)
        like e.g. Pearson and Spearman's correlation functions (these are 2 x n). Note
        from the above that the input array shape is reversed compared to typical
        n x k case. Thus, 1-d functions must be defined with axis=1 (operate row-wise)
        and multi-d functions must accept 3-d tensor-kind inputs. Latter can be done
        by using np.newaxis to create one extra dimension. For more on this, please
        take a look at the Examples section below and the statistics module which contains
        implementations for which the mentioned aspects have been taken into account.

    alpha : int or float
        Coverage level of the confidence interval. Default is 95.

    smooth_bootstrap : bool
        Default is False. If True, adds random noise to each bootstrap sample
        in order to reduce discreteness of the bootstrap distribution.

    plot_style : str or None
        Pyplot plotting style. If None (default), uses pyplot's default style.
        Plotting class in module plotting defines allowed styles and the passed
        style will be checked against them. If the passed style is not allowed,
        an exception will be raised.

    Examples
    --------
    Consider first a one-dimensional statistic quantile and imaginary numerical data X
    with shape n x p (n observations, p attributes). For now, sampling distribution
    estimation is run simultaneously for two columns of the data X and hence there
    will be two separate results. NumPy's own quantile function is used, but the version
    from the statistics module could also be used.

    >>> import numpy as np
    >>> from sampdist import SampDist
    >>> def quantile(x): return np.quantile(x, q=0.1, axis=1)
    >>> samp = SampDist(quantile, alpha=99, smooth_bootstrap=True)
    >>> samp.estimate(X[:, [0,2]]) # estimate sampling distribution for columns 0 and 2
    >>> samp.se, samp.ci # both standard errors and BCa confidence intervals
    >>> samp.plot(column=0) # sampling distribution plot, column=1 would plot for the X[:, 2]

    Consider then a two-dimensional statistic Pearson's linear correlation and the same
    data as above. In this case, estimation must be run for two columns of the data X
    and their correlation coefficient will be received for output. Correlation function
    is imported from the statistics module where it has been implemented to accept 3-d
    data as for input (SampDist requires extra dimension to be inserted in this case).

    >>> from sampdist import SampDist
    >>> from sampdist.statistics import corr_pearson
    >>> samp = SampDist(corr_pearson)
    >>> samp.estimate(X[:, :2], multid=True) # estimate sampling distribution for columns 0 and 1
    >>> samp.se, samp.ci # standard error and BCa confidence interval
    >>> samp.plot() # plot the sampling distribution for correlation

    References
    ----------
    Efron, B. 1979. Bootstrap methods: Another look at the jackknife. Ann. Statist., 7(1), 1-26.

    Efron, B. 1987. Better bootstrap confidence intervals. J. Amer. Statist. Assoc., 82(397), 171-200.

    Efron, B., Hastie T. 2016. Computer age statistical inference. Cambridge University Press.
    """

    def __init__(
        self,
        statistic: Callable[[np.ndarray], Any],
        alpha: Union[int, float] = 95,
        smooth_bootstrap: bool = False,
        plot_style: Optional[str] = None,
    ) -> None:
        self.actual_stat: Union[float, np.ndarray] = np.nan
        self.b_stats: Union[float, np.ndarray] = np.nan
        self.se: Union[float, np.ndarray] = np.nan
        self.ci: Union[float, np.ndarray] = np.nan

        self._acceleration: Union[float, np.ndarray] = np.nan
        self._z0 = np.nan

        self._alpha_min = 90
        self._alpha_max = 99.9999

        self.statistic = statistic
        self.alpha = alpha

        self.smooth_bootstrap = bool(smooth_bootstrap)

        self._plot_style = plot_style
        general_plot_config = (self._plot_style and {"plot_style_sheet": self._plot_style}) or {}

        self._plot_obj = Plotting(**general_plot_config)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(statistic={self.statistic.__name__}, alpha={self.alpha})"

    def _reset_estimates(self) -> None:
        self.actual_stat = np.nan
        self.b_stats = np.nan
        self.se = np.nan
        self.ci = np.nan

        self._acceleration = np.nan
        self._z0 = np.nan

    @property
    def statistic(self) -> Callable[[np.ndarray], Any]:
        return self._statistic

    @statistic.setter
    def statistic(self, func: Callable[[np.ndarray], Any]) -> None:
        if not callable(func):
            raise TypeError("Statistic must be callable")

        self._statistic = func
        self._reset_estimates()

    @property
    def alpha(self) -> Union[int, float]:
        return self._alpha

    @alpha.setter
    def alpha(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("Alpha must be an integer or float")

        if not (value >= self._alpha_min and value <= self._alpha_max):
            raise ValueError(f"Alpha must be within [{self._alpha_min},{self._alpha_max}]")

        self._alpha = value

    def _test_statistic_validity(self, multid: bool) -> None:
        test_array = np.ones(2) if not multid else np.ones((2, 2))
        # Quick and dirty validity test
        try:
            self.statistic(test_array)
        except (np.AxisError, IndexError):
            # Test resulted in one of accepted errors
            # 1 dim: axis 1 is out of bounds for array of dimension 1
            # multid: too many indices for array: array is 2-dimensional, but 3 were indexed
            pass
        else:
            data_dim = f"{'multi' if multid else 'one'}-dimensional data"
            help_msg = f"Make sure that {'extra dimension' if multid else 'axis=1'} is set."
            raise StatisticError(
                f"{self.statistic.__name__} is ill-defined for {data_dim}. {help_msg}"
            )

    def _compute_actual_statistic(self, sample: np.ndarray, multid: bool) -> None:
        if multid:
            self.actual_stat = self.statistic(sample[np.newaxis, :, :])
        elif sample.ndim == 1:
            self.actual_stat = self.statistic(sample.reshape(1, -1))
        else:
            self.actual_stat = self.statistic(sample.T)

    def _draw_bootstrap_samples(
        self, sample: np.ndarray, iterations: int, multid: bool
    ) -> np.ndarray:
        rg = np.random.default_rng()

        if sample.ndim == 1 or multid:
            dims2 = (iterations, sample.shape[0])
            row_indices = rg.integers(0, sample.shape[0], size=dims2)

            data: np.ndarray = sample[row_indices]

        else:
            # 1-d statistic but estimation run for multiple columns
            dims3 = tuple([iterations] + list(sample.shape))
            row_indices = rg.integers(0, sample.shape[0], size=dims3)

            data = sample[row_indices, np.arange(sample.shape[1])]

        if self.smooth_bootstrap:
            sigma = 1 / np.sqrt(sample.shape[0])
            data = data + rg.normal(scale=sigma, size=data.shape)

        return data

    def _get_estimate_of_acceleration(self, sample: np.ndarray, multid: bool) -> None:
        stat_count = sample.shape[0]

        if sample.ndim == 1:
            jackknife_stats = np.array(
                [self.statistic(np.delete(sample, idx).reshape(1, -1)) for idx in range(stat_count)]
            )
        elif multid:
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

        difference = jackknife_stats - np.mean(jackknife_stats, axis=0)
        denominator = np.sum(difference**2, axis=0) ** 1.5
        nominator = np.sum(difference**3, axis=0)

        acceleration = np.full(denominator.shape, np.inf)
        almost_zero = denominator < 1e-300
        acceleration[~almost_zero] = 1 / 6 * nominator[~almost_zero] / denominator[~almost_zero]

        self._acceleration = acceleration

    def _get_estimate_of_bias_correction(self) -> None:
        p0 = np.sum(self.b_stats <= self.actual_stat, axis=0) / self.b_stats.shape[0]

        if self.b_stats.ndim == 1:
            p0 = np.array([p0])

        invalid_p0 = (p0 == 0) | (p0 == 1)

        if np.any(invalid_p0, axis=0):
            invalid_cols = invalid_p0.nonzero()[0]

            raise BcaError(f"BCa confidence interval cannot be computed for columns {invalid_cols}")

        self._z0 = norm.ppf(p0)

    def _get_bca_confidence_intervals(self, alpha: float) -> np.ndarray:
        self._get_estimate_of_bias_correction()

        x0 = self._z0 + (self._z0 + norm.ppf(alpha)) / (
            1 - self._acceleration * (self._z0 + norm.ppf(alpha))
        )
        p = norm.cdf(x0)
        p_rounded = np.round(p * 100, decimals=3)

        q_values: np.ndarray = np.percentile(self.b_stats, p_rounded, axis=0, method="linear")

        if q_values.ndim == 2:
            q_values = np.diagonal(q_values)

        return q_values

    def _compute_estimates(self, sample: np.ndarray, iterations: int, multid: bool) -> None:
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

    def plot(
        self, column: int = 0, bins: Union[int, str] = "sqrt", comparison: bool = False
    ) -> None:
        """Plot bootstrap distribution with SE, observed value and BCa CIs.

        Distribution is represented by a histogram, roughly illustrating the density of
        the underlying distribution. Standard error, observed value and BCa confidence
        interval values are represented on the right side of the histogram. In addition,
        black and red triangles illustrate the observed value and confidence interval points
        on the histogram, respectively.

        Invokes pyplot's show method which may require to set the correct or desired
        matploblib backend.

        Parameters
        ----------
        column : int
            Number of the column to be plotted, defaults to the first column of the data.

        bins : int or str
            Count of consecutive and non-overlapping intervals for the x-axis of the histogram.
            Default value is "sqrt", indicating a particular binning strategy. If given
            as integer, it represents precisely the interval count. On the contrary, when
            given as str value, it must describe a binning strategy, which must be one of
            the following: "auto", "sturges", "fd", "doane", "scott", "rice" or "sqrt".

        comparison : bool
            Default is False when the plotted histogram will not have naive percentile CIs
            included in addition to the BCa confidence interval. If True, it's the
            contrary case.
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
                raise ValueError(f"Bins must be given within [1,{self.b_stats.shape[0]}]")

        elif isinstance(bins, str):
            allowed_strategies = {"auto", "sturges", "fd", "doane", "scott", "rice", "sqrt"}
            if bins not in allowed_strategies:
                raise ValueError(f"Bins strategy must be one of {', '.join(allowed_strategies)}")

        else:
            raise TypeError("Bins must be an integer or string describing the binning strategy")

        plot_data: PlotData = {
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

    def estimate(self, sample: np.ndarray, iterations: int = 5000, multid: bool = False) -> None:
        """Compute sampling distribution, SE and BCa confidence interval for given statistic.

        Prior any computations, validity of the statistic is evaluated. It must apply
        correctly either to one- or two-dimensional data.

        Parameters
        ----------
        sample : numpy.ndarray
            Sample representing the observed data, ndim = 1 or = 2. Pass the data in normal
            format as n x p (n observations, p attributes). E.g., given data matrix X,
            both X[:,col] and X[:,[col_1:col_N]] would be valid inputs.

        iterations : int
            Number of (bootstrap) resamples to be drawn. Default is 5000.

        multid : bool
            Default `False` means that the statistic applies one-dimensionally to data, i.e.,
            one output for n x 1 data and p outputs for n x p data. If set to `True`, it
            refers to multidimensional case where the statistic would produce one output for
            two-dimensional data. Consider e.g. Pearson or Spearman's correlation as examples.

        Raises
        ------
        StatisticError : subclass of SampDistError
            `self.statistic` is ill-defined to be used in estimation.

        BcaError : subclass of SampDistError
            Sampling distribution is degenerate and thus cannot compute BCa CIs.
            This can happen for example if the data (in bootstrap samples) is almost identical.
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
