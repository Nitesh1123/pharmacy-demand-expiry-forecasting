import pandas as pd

from src.splits import assign_demand_split, assign_expiry_split

END = "2023-12-31"


def test_demand_windows_never_cross_a_boundary():
    origins = pd.Series(["2023-04-30", "2023-05-01", "2023-05-07", "2023-06-04",
                         "2023-08-06", "2023-09-03", "2023-12-01", "2023-12-03"])
    # Target is origin+1 .. origin+30; the window must finish before the next boundary.
    assert assign_demand_split(origins, END).tolist() == [
        "train", "train", "excluded", "validation",
        "excluded", "test", "test", "excluded",
    ]


def test_expiry_labels_must_be_observed_before_the_next_boundary():
    as_of = pd.Series(["2023-05-07", "2023-05-07", "2023-06-04", "2023-07-30",
                       "2023-09-03", "2023-10-01"])
    outcome = pd.Series(["2023-05-30", "2023-06-02", "2023-07-01", "2023-09-05",
                         "2023-10-01", None])
    assert assign_expiry_split(as_of, outcome, END).tolist() == [
        "train", "excluded", "validation", "excluded", "test", "excluded",
    ]


def test_target_running_past_the_data_is_never_training():
    split = assign_demand_split(pd.Series(["2022-04-24"]), "2022-04-30")
    assert split.tolist() == ["excluded"]


def test_unknown_outcome_is_never_a_test_example():
    split = assign_expiry_split(pd.Series(["2023-09-03"]), pd.Series([float("nan")]), END)
    assert split.tolist() == ["excluded"]
