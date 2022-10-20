# Bootstrap sampling distribution #

[![main](https://github.com/elmomoilanen/Bootstrap-sampling-distribution/actions/workflows/main.yml/badge.svg)](https://github.com/elmomoilanen/Bootstrap-sampling-distribution/actions/workflows/main.yml)

Library that employs the statistical resampling method bootstrap to estimate a sampling distribution of a specific statistic from the provided sample of data. Estimates of standard error and confidence interval of the statistic can subsequently be determined from the obtained distribution, confidence interval being adjusted for both bias and skewness.

Generally speaking, in statistical inference the primary interest is to quantify an effect size of a measurement and as a secondary but yet important thing would be to evaluate uncertainty of the measurement. This library provides computational tools for the latter with an assumption that the given data sample is a representative sample of the unknown population. This enables to use the sample to generate new samples by the bootstrap resampling method.

Sampling distribution obtained by the bootstrap resampling process makes it possible to compute the standard error and confidence interval for the statistic, both quantifying statistical accuracy of the measurement. Standard error is the standard deviation of obtained values of the statistic whereas confidence intervals are constructed using the bias-corrected and accelerated bootstrap approach (BCa) which makes adjustments for bias and skewness.

At the moment, SciPy's API *stats.bootstrap* resembles this library but e.g. does not have integrated plotting.

## Install ##

Poetry is the recommended tool for installation and the following short guide uses it.

After cloning and navigating to the target folder, running the following command creates a virtual environment within this project directory and installs non-development dependencies inside it

```bash
poetry install --without dev
```

In-project virtual environment setup is controlled in *poetry.toml*. As the *--without dev* option skips installation of the development dependencies, do not include it in the command above if e.g. you want to be able to run the unit tests (pytest is needed for that).

For the plotting to work correctly it might be required to set the backend for Matplotlib. One way to do this is to set the MPLBACKEND environment variable (overrides any matplotlibrc configuration) for the current shell.

## Use ##

This section provides two short examples for usage of the library. To see the API docs, you can render the [documentation as HTML](#docs) or read the docstrings directly.

Start a new Python shell within the virtual environment e.g. as follows

```bash
MPLBACKEND= poetry run python
```

with a proper backend (e.g. macosx or qt5agg) after the equal sign. If the backend has been set correctly earlier, just drop this setting.

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

# samp.b_stats has the sampling distribution of the quantile for both columns
# Standard error (samp.se) and BCa CI (samp.ci) are also available

# Plot the sampling distribution for the first column
samp.plot(column=0)
```

After necessary module imports in the code snippet above, a custom quantile function was defined which calls NumPy's quantile routine notably with the axis parameter set to one. Statistics module of this library has also a quantile function implementation but for the sake of demonstration it was defined here directly. Lastly, after an object of the `SampDist` class was instantiated, its estimate method was called to compute the sampling distribution, standard error and BCa confidence interval. Data slice of shape n x 2
was passed to the estimate method and as the quantile statistic is one-dimensional (maps n x 1 input to single result and n x p input to p results) it ran the estimation simultaneously for both of the two attributes (columns 0 and 2 in X).

Following figure represents a possible result of the plot call. In addition to the histogram it shows the observed value (value of the statistic in original data sample) pointed to by the black arrow, standard error and BCa confidence interval pointed to by red arrows on x-axis.

![](docs/boostrap_distribution_quantile.png)

For the second example, let's consider the sampling distribution estimation process for a multidimensional statistic, e.g. Pearson's linear correlation. Keeping the mentioned assumptions regarding data X, following code estimates the sampling distribution and in the final row of the snippet, renders a histogram plot similarly to the figure above. Compared to previous example, notice the difference in estimation process of the chosen statistic. Here the multidimensional statistic, Pearson's correlation, requires two attributes (columns) of the data X as input (a data slice of shape n x 2) and produces a single output which is the value of correlation.

```python
from sampdist import SampDist

# Import custom implementation of Pearson's correlation
from sampdist import corr_pearson

samp = SampDist(corr_pearson)

# Run estimation for columns 0 and 1. It's now mandatory to set multid to True
samp.estimate(X[:, :2], multid=True)

samp.plot()
```

![](docs/bootstrap_distribution_corr.png)

Notice that validity of the statistic is checked when calling the estimate method. If the check fails, a `StatisticError` exception will be raised. Furthermore, if the estimated sampling distribution is degenerate (e.g. data almost identical), a `BcaError` exception gets raised (in this case you may try to use True for smooth_bootstrap parameter). Both exceptions inherit from class `SampDistError` which can be imported directly from sampdist namespace.

## Docs ##

Make sure that you included the *docs* dependency group in the installation step.

Render the documentation as HTML with the following command

```bash
sphinx-build -b html docs/source/ docs/build/html
```

and open the starting page docs/build/html/index.html in a browser.
