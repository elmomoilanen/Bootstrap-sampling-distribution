# Bootstrap sampling distribution #

A library that uses the statistical resampling method bootstrap to estimate a sampling distribution of a specific statistic from the original data that is used as a model of the unknown population. Estimates of standard error and confidence interval of the statistic can subsequently be determined from the obtained distribution, confidence interval being adjusted for both bias and skewness.

Generally speaking, in statistical inference the primary aspect is to quantify the effect size or practical significance of a measurement and as a secondary but yet important thing would be to evaluate uncertainty of the measurement. This library provides computational tools for the latter thing, by making an assumption that the original data is a representative sample of the unknown population and using this sample to generate new samples by the bootstrap resampling method to finally obtain a sampling distribution of the statistic under consideration. Sampling distribution enables one to compute standard error and confidence interval for the statistic, both quantifying precision of the measurement. The library uses the bias-corrected and accelerated (BCa) bootstrap approach to construct confidence intervals (CIs), meaning that the intervals are adjusted for bias and skewness and thus they usually differ e.g. from the intervals computed with the naive percentile method.

## Installation ##

To be added.

## Usage ##

To be added.
