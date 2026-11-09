# Reproduction and failure-case checks — November 8, 2026 (Day 3)

Planned for Day 8 and completed early. These checks changed how bad inputs are handled.
They did not change any model, split or result.

## Final fresh-clone run, November 9, 2026

The finished project was cloned from GitHub at commit `e4f8fca` into an empty folder with a
new Python 3.13 virtual environment, and the README's Windows commands were run in order.

- All steps passed: 96 tests, `pip check`, and the app's health endpoint returned `ok`.
- The hashes in `data/manifest.json` matched all five regenerated CSVs.
- The manifest and the quality, split, validation and test reports matched the checked-in
  copies. Only the re-executed notebook and its two exported EDA images differed.
- Commits after `e4f8fca` change documentation only.

## First fresh-clone reproduction

The project was cloned from GitHub at commit `fa3a8d8` into an empty folder with a new
Python 3.13 virtual environment. The Windows commands were read from the README and run
exactly, in order.

| Step | Result |
|---|---|
| Install pinned dependencies | Passed; `pip check` clean |
| Generate data, validate, build features | Passed |
| Execute the EDA notebook | Passed |
| Train, select and run the test evaluation | Passed |
| Save the model and build figures | Passed |
| Test suite at that commit | 78 passed |
| Start the Streamlit app | Passed; health endpoint returned `ok` |

These outputs were byte-for-byte identical to the checked-in copies:

- `data/manifest.json`, whose SHA-256 hashes also matched the regenerated CSVs
- `reports/day2-data-quality.json`
- `reports/day3-splits.json`
- `reports/day3-validation.json`
- `reports/test-evaluation.json`

The only tracked files that differed were the re-executed notebook and the two EDA images
exported from it, which are re-rendered on each run.

CI separately installs the pinned stack and runs the pipeline on Windows and Linux.

## Failure cases

`tests/test_failure_cases.py` has 18 cases. Thirteen failed before fixes, and all now pass.

| Situation | Before | Now |
|---|---|---|
| Damaged or empty model file | Crash | Refused; the app shows how to rebuild |
| Model file of the wrong type | Crash | Refused |
| Model built with another scikit-learn version | Loaded anyway | Refused, with a rebuild message |
| Data with missing product-days | **Forecast from it silently** | Refused by the quality checks |
| Table missing a required column | Failed later or unclear | Refused, naming the columns |
| Empty data file | App crash | Message with rebuild commands |
| History too short for any review date | App crash | Message explaining 56 days are needed |
| Missing input files on the command line | Traceback | One clear error line, exit code 2 |

These cases already behaved sensibly and are now covered by tests:

| Situation | Behaviour |
|---|---|
| Product category not seen in training | Finite, non-negative forecast |
| Sales 1,000 times larger than usual | Finite, non-negative forecast |
| Product with no sales | Forecast below the median of other products |
| Product with no stock | Flagged for reorder |
| Date with no near-expiry batches | Empty table, no alerts |

## Results on other datasets

The whole pipeline was re-run on three other random seeds. Each seed made its own selection
on its own validation rows and scored its own test rows once. The default seed 42 results
were not changed by this.

| Seed | Test batches | Expiry selection | Demand WAPE: served / seasonal naive | Expiry F1: rule | Best other F1 |
|---:|---:|---|---|---:|---:|
| 42 (default) | 103 | Rule | 4.39% / 4.52% | 0.937 | 0.950 |
| 7 | 72 | Rule | 4.17% / 4.25% | 0.949 | 0.961 |
| 123 | 104 | Rule | 4.32% / 4.07% | 0.936 | 0.903 |
| 2024 | 86 | Rule | 3.89% / 3.98% | 0.945 | 0.935 |

- No candidate cleared the selection bar on any seed, so the rule was selected on all four.
- Gradient boosting beat seasonal naive on WAPE on three of four seeds, always by a small
  margin. The two remain close.
- On F1, the rule was best on two seeds, and a candidate was ahead on the other two. Sample
  sizes are too small to separate them.
- All seeds share the same simulator, so this shows stability across random draws, not
  performance on different or real data.

## Verification of the fixes

- 96 tests pass, including all failure cases, and `pip check` is clean.
- After the fixes, regenerating the default pipeline left every tracked result unchanged.
- The fresh-clone run used the commit before these fixes. CI re-runs the full pipeline on this
  change.
