"""Quick guide to Bootstrap-sampling-distribution package.

This package contains the following modules

- errors
- plotting
- sampling
- statistics

`SampDist` class from the `sampling` module implements the main functionality
of this package and can be interacted via its public interfaces `estimate`
and `plot`.

For convenience some common statistical functions have been implemented library
compatibly in the `statistics` module.
"""

from bootstrap_sampling_distribution.errors import SampDistError
from bootstrap_sampling_distribution.sampling import SampDist

from bootstrap_sampling_distribution.statistics import (
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
