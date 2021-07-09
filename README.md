# Bootstrap sampling distribution #

A library that uses the statistical resampling method bootstrap to estimate a sampling distribution of a specific statistic from the original data that is used as a model of the unknown population. Estimates of standard error and confidence interval of the statistic can subsequently be determined from the obtained distribution, confidence interval being adjusted for both bias and skewness.

Generally speaking, in statistical inference the primary aspect is to quantify the effect size or practical significance of a measurement and as a secondary but yet important thing would be to evaluate uncertainty of the measurement. This library provides computational tools for the latter thing, by making an assumption that the original data is a representative sample of the unknown population and using this sample to generate new samples by the bootstrap resampling method to finally obtain a sampling distribution of the statistic under consideration. Sampling distribution enables one to compute standard error and confidence interval for the statistic, both quantifying precision of the measurement. The library uses the bias-corrected and accelerated (BCa) bootstrap approach to construct confidence intervals (CIs), meaning that the intervals are adjusted for bias and skewness and thus they usually differ e.g. from the intervals computed with the naive percentile method.

## Installation ##

To be added.

## Usage ##

This section provides two short examples for usage of the library and by any means they should not be considered as giving a complete guide of inner parts of the library. Instead one is adviced to read the docstring of `SampDist` class prior usage.

Consider first the following example, where we assume X to be a numerical data with shape N x P (N observations and P attributes) and 10-quantile be the statistic of interest. Let's further assume that for the data P is equal to or larger than three.

```python
import numpy as np
from sampdist import SampDist

def quantile(x): return np.quantile(x, q=0.1, axis=1) # library requires one-dimensional statistics to be defined with axis=1

samp = SampDist(quantile, alpha=99, smooth_bootstrap=True) # provide values for kwargs alpha and smooth_bootstrap
samp.estimate(X[:, [0,2]]) # estimate sampling distribution simultaneously for columns 0 and 2 of the data
# now samp.b_stats, samp.se and samp.ci are available for usage

samp.plot(column=0) # plot the sampling distribution for first column (se and ci will be included)
samp.plot(column=1)
```

After necessary module imports, a custom quantile function was defined which calls NumPy's quantile routine notably with axis parameter set to one. Lastly, after an object of the `SampDist` class has been instantiated, the estimate method was called to compute the sampling distribution, standard error and BCa confidence interval that can then be plotted by the plot method.



