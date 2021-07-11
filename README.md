# Bootstrap sampling distribution #

A library that uses the statistical resampling method bootstrap to estimate a sampling distribution of a specific statistic from the original data that is used as a model of the unknown population. Estimates of standard error and confidence interval of the statistic can subsequently be determined from the obtained distribution, confidence interval being adjusted for both bias and skewness.

Generally speaking, in statistical inference the primary interest is to quantify the effect size or in other words practical significance of a measurement and as a secondary but yet important thing would be to evaluate uncertainty of the measurement. This library provides computational tools for the latter thing by making an initial assumption that the original data is a representative sample of the unknown population allowing one to use the sample to generate new samples by the bootstrap resampling method to finally obtain the sampling distribution of the statistic under consideration. Sampling distribution enables one to compute standard error and confidence interval for the statistic, both quantifying precision of the measurement. The library uses the standard deviation of computed values of the statistic as the standard error and confidence intervals are constructed using the bias-corrected and accelerated (BCa) bootstrap approach, indicating that the intervals are adjusted for bias and skewness and thus making them differ e.g. from the confidence intervals computed with the naive percentile method.

## Installation ##

Python and poetry (package and dependency manager for Python) are prerequisites for a successful installation. For independent usage, after cloning and navigating to the target folder, run the command `poetry install` in order to create an own virtual environment for the library and install required third-party dependencies (such as NumPy) inside it. Unit tests can be run by the command `poetry run pytest`.

## Usage ##

This section provides two short examples for usage of the library and by any means they should not be considered as giving a complete guide of inner parts of the library. Instead one is adviced to read the docstring of `SampDist` class prior usage.

Consider first the following example, where we assume X to be a numerical data with shape N x P (N observations and P attributes) and 10-quantile be the statistic of interest. Let's further assume that for the data P is equal to or larger than three.

If needed, run the command `poetry shell` to activate the virtual environment related to this library.

```python
import numpy as np
from sampdist import SampDist

# library requires one-dimensional statistics to be defined with axis=1
def quantile(x): return np.quantile(x, q=0.1, axis=1)

samp = SampDist(quantile, alpha=99, smooth_bootstrap=True) # provide values for kwargs alpha and smooth_bootstrap

# estimate sampling distribution simultaneously for columns 0 and 2 of the data (column indices run from 0 to P-1)
samp.estimate(X[:, [0,2]])

# now samp.b_stats, samp.se and samp.ci are available for usage, they can be inspected also from a figure

samp.plot(column=0) # plot the sampling distribution for first column (se and ci will be included)
```

After necessary module imports, a custom quantile function was defined which calls NumPy's quantile routine notably with axis parameter set to one. Lastly, after an object of the `SampDist` class has been instantiated, the estimate method was called to compute the sampling distribution, standard error and BCa confidence interval that can then be plotted by the plot method. Following figure is a result of the plot call. In addition to the histogram (sampling distribution) it shows the observed value (value of the statistic in original data), standard error and BCa confidence interval highlighted in red.

![](docs/boostrap_distribution_quantile.png)

As an other example, let's consider the sampling distribution estimation process for a multidimensional statistic, namely Pearson's linear correlation. Keeping the previously mentioned assumptions regarding the data X, the following code estimates the sampling distribution and in the final row, renders the histogram plot similarly to the figure above.

```python
# import custom implementation of Pearson's correlation from statistics module
from sampdist.statistics import corr_pearson

samp = SampDist(corr_pearson)
samp.estimate(X[:, :2], multid=True) # multid must be set to True

samp.plot()
```

![](docs/bootstrap_distribution_corr.png)

To wrap up this section, for one-dimensional statistics (takes one column/attribute of the data as an input and produce single output) this library is quite convenient to use but unfortunately, for multidimensional statistics like correlation (take k columns as input and produce single output, in particular k=2 for correlation) usage is little tricky as one is required to implement functions to take data in higher-dimensional format (e.g. in 3d for correlation). Please take a look at the statistics module to see the implementation of Pearson's correlation and of course the code related to bootstrap estimation in sampling module.
