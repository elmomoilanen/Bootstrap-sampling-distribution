# Bootstrap sampling distribution #

[![main](https://github.com/elmomoilanen/Bootstrap-sampling-distribution/actions/workflows/main.yml/badge.svg)](https://github.com/elmomoilanen/Bootstrap-sampling-distribution/actions/workflows/main.yml)

Library that uses the statistical resampling method bootstrap to estimate a sampling distribution of a specific statistic from the provided data. Estimates of standard error and confidence interval of the statistic can subsequently be determined from the obtained distribution, confidence interval being adjusted for both bias and skewness.

Generally speaking, in statistical inference the primary interest is to quantify the effect size or in other words practical significance of a measurement and as a secondary but yet important thing would be to evaluate uncertainty of the measurement. This library provides computational tools for the latter by making an initial assumption that the original data is a representative sample of the unknown population enabling usage of the available sample to generate new samples by the bootstrap resampling method to obtain at the end the sampling distribution of the statistic under consideration.

Sampling distribution makes it possible to compute standard error and confidence interval for the statistic, both quantifying precision of the measurement. Library uses the standard deviation of computed values of the statistic as the standard error and confidence intervals are constructed using the bias-corrected and accelerated (BCa) bootstrap approach, indicating that these intervals are adjusted for bias and skewness and thus making them possibly differ e.g. from the confidence intervals computed with the naive percentile method.

## Installation ##

Poetry (a package and dependency manager for Python) is recommended for installation and the following short guide uses it. Optionally, you can just look up the dependencies from *pyproject.toml* and start to use this library right away when you have set up an appropriate environment.

After cloning and navigating to the target folder, run the following command in a shell

```bash
poetry install --no-dev
```

which creates a virtual environment for the library and installs required non-development third-party dependencies (such as NumPy) inside it. Virtual environment setup is controlled by the *poetry.toml* file. As the *--no-dev* option skips installation of the development dependencies, don't include it in the command above if e.g. you want to be able to run the unit tests (pytest is needed for that).

## Usage ##

This section provides two short examples for usage of the library.

Let's consider first a case, where we assume X to be a numerical data with shape n x p (n observations, p attributes) and 10-quantile to be the statistic of interest. Let's further assume that the number of attributes p is equal to or larger than three.

```python
import numpy as np
from sampdist import SampDist

# One-dimensional statistics must be defined with axis=1
def quantile(x): return np.quantile(x, q=0.1, axis=1)

# Override default alpha and add random noise to bootstrap samples
samp = SampDist(quantile, alpha=99, smooth_bootstrap=True)

# Estimate sampling distribution simultaneously for columns 0 and 2
samp.estimate(X[:, [0,2]])

# Now samp.b_stats contains the sampling distribution of the quantile
# Standard error (samp.se) and BCa CI (samp.ci) are also available

# Plot the sampling distribution for the first column
samp.plot(column=0)
```

After necessary module imports in the code snippet above, a custom quantile function was defined which calls NumPy's quantile routine notably with axis parameter set to one. Statistics module of this library has also a quantile function implementation but for the sake of demonstration it was defined here directly. Lastly, after an object of the `SampDist` class has been instantiated, estimate method of the class was called to compute the sampling distribution, standard error and BCa confidence interval. Data slice of shape n x 2
was passed to the estimate method and as the statistic is one-dimensional (requires input data in format n x 1 to produce one result) it makes the estimation simultaneously for both of the two attributes but of course the results do not have any interdependence.

Following figure is the result of the plot call. In addition to the histogram it shows the observed value (value of the statistic in original data) pointed to by the black arrow, standard error and BCa confidence interval highlighted at the upper right corner of the figure and pointed to by red arrows on x-axis.

![](docs/boostrap_distribution_quantile.png)

For other example, let's consider the sampling distribution estimation process for a multidimensional statistic, e.g. Pearson's linear correlation. Keeping the mentioned assumptions regarding data X, following code estimates the sampling distribution and in the final row, renders a histogram plot similarly to the figure above. Compared to previous example, notice the difference in estimation process of the chosen statistic. Here the multidimensional statistic, Pearson's correlation, requires two attributes of data X as input (or more precisely a data slice of shape n x 2) and produces a single output which is the value of correlation.

```python
from sampdist import SampDist

# Import custom implementation of Pearson's correlation
from sampdist import corr_pearson

samp = SampDist(corr_pearson)

# It's now mandatory to set multid to True
samp.estimate(X[:, :2], multid=True)

samp.plot()
```

![](docs/bootstrap_distribution_corr.png)

To wrap up this section, for one-dimensional statistics (that take one column/attribute of the data as input and produce one output) this library is quite convenient to use but for multidimensional statistics like correlation (which take k columns as input and produce one output) usage is bit trickier as it is required to implement these functions to take data input in higher-dimensional format (3d for correlation in particular). For further information and examples on this, please take a look at the statistics module.
