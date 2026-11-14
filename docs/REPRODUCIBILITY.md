# Reproducibility Notes

The public code release is versioned separately from the manuscript files. The current release
identifier is `biogait-pd-paper-revision-0.2.0`, published at Git tag `v0.2.0` (commit `e38b54c`).
The final submission should also cite a permanent archive record rather than only an unpinned
branch.

## Main Experiments

The release keeps the scripts needed for the main manuscript analyses:

- `scripts/run_carepd_benchmark_compatible_lodo.py`: CAREPD-compatible leave-one-dataset-out evaluation.
- `scripts/run_inhouse_target_adaptation_experiment.py`: target-centre adaptation using in-house labels.
- `scripts/run_inhouse_subject_level_external_test.py`: subject-level aggregation and evaluation.
- `scripts/make_feature_dictionary.py`: feature dictionary generation.

Helper modules required by these entry points are retained in `scripts/internal/`.

The manuscript protocol is frozen in `release/configs/paper_final_config.json`. Machine-readable
companions to the Supplementary tables and aggregate audit tables are in
`release/machine_readable/`. Their filenames follow the Supplementary table numbering used in the
manuscript.

## Randomness

The main scripts use fixed random seeds where applicable. For exact reproduction, record:

- Python version.
- Package versions.
- Dataset release and preprocessing version.
- Train/test split files.
- Random seeds used by each script.

The target-centre adaptation uses the participant-disjoint split seeds listed in the release
configuration. Public-cohort uncertainty uses participant-clustered bootstrap resampling; the
bootstrap unit and replicate counts are also recorded in that configuration.

## Outputs

Most scripts write to `outputs/` or `paper/res/`. These directories are excluded from Git because
they can contain large intermediate files and patient-level predictions. The aggregate tables
needed to inspect the reported results are included under `release/machine_readable/`.

## Restricted reproducibility materials

The public repository intentionally excludes raw videos, in-house clinical records, direct
participant identifiers, large motion files and unrestricted patient-level feature tables. If the
data-use approval permits controlled sharing, the frozen split manifests, per-unit predictions,
selected-feature inventories and trained model files should be deposited in a confidential or
controlled archive and referenced by its permanent identifier in the response letter. The public
repository must not imply that these restricted materials are available when they have not been
archived.
