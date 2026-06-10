# Scripts

The top-level scripts are the intended public entry points. Helper modules used by these entry points live in `scripts/internal/` and are not meant to be run directly.

## Entry Points

| Script | Purpose |
| --- | --- |
| `run_study.py` | Basic feature extraction and model-selection pipeline for local retargeted motion data. |
| `run_carepd_benchmark_compatible_lodo.py` | CARE-PD benchmark-compatible leave-one-dataset-out evaluation. |
| `run_inhouse_target_adaptation_experiment.py` | Target-centre adaptation using in-house labelled data. |
| `run_inhouse_subject_level_external_test.py` | Subject-level aggregation and external testing on the in-house cohort. |
| `make_feature_dictionary.py` | Generate the feature dictionary used for manuscript reporting. |

## Internal Helpers

`scripts/internal/` contains model components, gating utilities and older intermediate experiment modules that are imported by the public entry points. They are kept for reproducibility but hidden from the main script list to keep the repository readable.
