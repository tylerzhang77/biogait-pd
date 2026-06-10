# Reproducibility Notes

## Main Experiments

The release keeps the scripts needed for the main manuscript analyses:

- `scripts/run_carepd_benchmark_compatible_lodo.py`: CAREPD-compatible leave-one-dataset-out evaluation.
- `scripts/run_inhouse_target_adaptation_experiment.py`: target-centre adaptation using in-house labels.
- `scripts/run_inhouse_subject_level_external_test.py`: subject-level aggregation and evaluation.
- `scripts/make_feature_dictionary.py`: feature dictionary generation.

Helper modules required by these entry points are retained in `scripts/internal/`.

## Randomness

The main scripts use fixed random seeds where applicable. For exact reproduction, record:

- Python version.
- Package versions.
- Dataset release and preprocessing version.
- Train/test split files.
- Random seeds used by each script.

## Outputs

Most scripts write to `outputs/` or `paper/res/`. These directories are excluded from Git because they can contain large intermediate files and patient-level predictions.
