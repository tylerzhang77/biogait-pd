# Data Preparation

This repository contains executable code, aggregate machine-readable results and an authorized
set of derived participant-level reproducibility assets. It intentionally excludes raw videos,
in-house clinical records, the full feature tables, trained checkpoints and large cached motion
files.

## Public CARE-PD Data

CARE-PD should be downloaded from the official release rather than redistributed in this repository:

- Project page: <https://neurips2025.care-pd.ca/>
- Dataset DOI: <https://doi.org/10.5683/SP3/TWIKMK>
- Official benchmark code: <https://github.com/TaatiTeam/CARE-PD>

Use of CARE-PD data must follow the terms specified by the original data provider.

## Suggested Local Layout

```text
data/
  carepd/
    features/                      # CAREPD-derived feature tables, if available
    metadata/                      # Dataset, subject and label metadata
  inhouse/
    labels.csv
    retargeted_motion/
      <segment_id>.npz
    keypoints/
      <segment_id>/
        keypoints_body.npz
        meta.json
  models/
    MS-Human-700_skeleton.xml
```

The `data/` directory is ignored by Git.

## Label Table

The basic in-house pipeline expects a CSV with these columns:

- `patient_id`: participant identifier.
- `gait_label`: ordinal gait-severity label.
- `retarget_npz`: path to a retargeted motion `.npz` file.
- Optional cohort or segment columns can be retained for downstream analysis.

For multicentre CAREPD experiments, the paper scripts expect a feature table containing metadata columns such as dataset, subject and gait label, followed by numeric feature columns. The exact feature-table schema depends on the preprocessing pipeline used to export CAREPD and in-house motion features.

## Motion Files

Each retargeted `.npz` file should contain:

- `qpos`: an array of shape `(frames, generalized_coordinates)`.
- `fps`: optional scalar frame rate. If absent, the pipeline assumes 30 Hz.

## Released derived assets and restricted data

The release includes author-authorized derived predictions, adaptation split assignments and
selected-feature inventories under `release/reproducibility_assets/`. These files contain only
analysis identifiers and derived values; they do not include raw videos, clinical records or
contact information. Their release remains subject to the governing ethics approval and data-use
agreements.

The in-house clinical cohort itself cannot be redistributed because of participant privacy and
institutional data-use restrictions. Raw videos, clinical records, full feature tables and trained
model checkpoints are therefore not included in the public repository. The public CARE-PD data
should be obtained from the official provider under its applicable terms.
