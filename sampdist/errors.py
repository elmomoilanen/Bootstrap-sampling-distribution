"""Implements custom errors."""


class SampDistError(Exception):
    """Base class for `sampdist` package errors."""


class StatisticError(SampDistError):
    """Statistical function failed the validity check."""


class BcaError(SampDistError):
    """Sampling distribution is degenerate and cannot compute BCa CIs."""
