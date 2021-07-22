"""Quick guide to sampdist package.

Package contains the following modules:
- sampling
- plotting
- statistics

`SampDist` class from the sampling modules implements
the main functionality of this package and can be interacted
via its public interfaces `estimate` and `plot`.

Some common statistical functions have been implemented in the
statistics module for convenience.
"""
from sampdist.sampling import SampDist

from sampdist.statistics import (
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
