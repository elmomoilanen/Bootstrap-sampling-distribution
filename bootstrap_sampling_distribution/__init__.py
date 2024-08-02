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

from bootstrap_sampling_distribution.errors import SampDistError as SampDistError
from bootstrap_sampling_distribution.sampling import SampDist as SampDist

from bootstrap_sampling_distribution.statistics import (
    mean as mean,
    geometric_mean as geometric_mean,
    harmonic_mean as harmonic_mean,
    trimmed_mean_factory as trimmed_mean_factory,
    median as median,
    maximum as maximum,
    minimum as minimum,
    skew as skew,
    kurtosis as kurtosis,
    interquartile_range as interquartile_range,
    standard_deviation as standard_deviation,
    median_absolute_deviation as median_absolute_deviation,
    quantile_factory as quantile_factory,
    corr_pearson as corr_pearson,
    corr_spearman as corr_spearman,
)
