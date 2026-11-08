"""Chronological split rules shared by every product (see docs/PRD.md)."""

import pandas as pd

VALID_START = "2023-06-01"
TEST_START = "2023-09-01"
HORIZON_DAYS = 30


def assign_demand_split(origin, end_date, valid_start=VALID_START, test_start=TEST_START):
    """Label each prediction origin; its t+1..t+30 target must end inside one period."""
    origin = pd.to_datetime(pd.Series(origin))
    target_end = origin + pd.Timedelta(days=HORIZON_DAYS)
    return _label(origin, target_end, end_date, valid_start, test_start)


def assign_expiry_split(as_of, outcome_date, end_date, valid_start=VALID_START,
                        test_start=TEST_START):
    """Label each snapshot; its observed outcome must be known before the next boundary.

    Unknown (censored) outcomes are always excluded, never treated as negatives.
    """
    return _label(pd.to_datetime(pd.Series(as_of)),
                  pd.to_datetime(pd.Series(outcome_date)), end_date, valid_start, test_start)


def _label(start, finish, end_date, valid_start, test_start):
    valid_start, test_start, end_date = map(pd.Timestamp, (valid_start, test_start, end_date))
    observed = finish <= end_date  # also False for unknown (NaT) outcomes
    split = pd.Series("excluded", index=start.index)
    split[observed & (finish < valid_start)] = "train"
    split[observed & (start >= valid_start) & (finish < test_start)] = "validation"
    split[observed & (start >= test_start)] = "test"
    return split
