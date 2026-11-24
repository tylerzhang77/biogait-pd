# BioGait-PD

BioGait-PD is an interpretable kinematic modelling pipeline for cross-cohort gait-impairment rating prediction. The code in this release supports feature extraction from retargeted skeletal motion, CARE-PD-style leave-one-dataset-out evaluation, in-house target-centre adaptation, subject-level evaluation and feature-dictionary generation.

This repository does not include raw videos, in-house clinical records, direct participant
identifiers or large intermediate motion files. It includes a de-identified, segment-level
processed feature table for the in-house cohort under
`release/reproducibility_assets/data/`, together with derived predictions, adaptation splits,
selected-feature inventories and reconstructed verified model bundles. Public CARE-PD data should
be obtained from the official CARE-PD release.

## Reproducibility release

The public release manifest, frozen paper protocol, machine-readable Supplementary tables,
author-authorized participant-level reproducibility assets and reconstructed verified checkpoints
are in `release/`. The release identifier for this revision is
`biogait-pd-paper-revision-0.2.2`, with the final checkpoint, feature-table and verification
assets included. The checkpoint manifests retain the
qualification that the historical fitted objects were not preserved and that the exported models
were refitted from the frozen protocol and verified against the archived predictions.

## CARE-PD Data Access

CARE-PD is available for non-commercial research through the official release:

- Project page: <https://neurips2025.care-pd.ca/>
- Dataset DOI: <https://doi.org/10.5683/SP3/TWIKMK>
- Official benchmark code: <https://github.com/TaatiTeam/CARE-PD>

## Repository Structure

```text
biogait-pd-github/
  configs/                         # Local configuration templates
  docs/                            # Data and reproducibility notes
  examples/                        # Minimal input table schemas
  scripts/                         # Public experiment entry points
  scripts/internal/                # Helper modules imported by entry points
  src/parkinson_gait/              # Reusable feature extraction and modelling package
  pyproject.toml
  requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
pip install -r requirements.txt
```

## Quick Start

Copy and edit the local configuration:

```bash
cp configs/study_config.local.example.json configs/study_config.local.json
```

Run the basic in-house feature extraction and model-selection pipeline:

```bash
python scripts/run_study.py --config configs/study_config.local.json
```

Run CAREPD benchmark-compatible leave-one-dataset-out evaluation:

```bash
python scripts/run_carepd_benchmark_compatible_lodo.py
```

Run in-house target-centre adaptation:

```bash
python scripts/run_inhouse_target_adaptation_experiment.py
```

Run subject-level external testing on the in-house cohort:

```bash
python scripts/run_inhouse_subject_level_external_test.py
```

Generate the feature dictionary used for reporting:

```bash
python scripts/make_feature_dictionary.py
```

## Expected Inputs

At minimum, the core pipeline expects:

- A label table with participant identifiers, severity labels and paths to retargeted motion files.
- Retargeted motion files in `.npz` format containing `qpos` and optionally `fps`.
- The musculoskeletal model XML used to map generalized coordinates.
- Optional keypoint files for keypoint-fusion features.

See `examples/labels_schema.csv`, `examples/feature_table_schema.csv` and `docs/DATA.md`.

## Outputs

Scripts write results under `outputs/` or `paper/res/` by default. These folders are ignored by Git because they may contain large derived data, predictions and cached features.

## Citation

If you use this code, please cite the associated manuscript once available.

```bibtex
@software{zhang_biogaitpd_2026,
  title = {BioGait-PD: interpretable kinematic modelling of gait-impairment ratings across clinical cohorts},
  author = {Zhang, Tianyao and others},
  year = {2026},
  version = {0.2.2},
  url = {https://github.com/tylerzhang77/biogait-pd/tree/v0.2.2}
}
```

## Data and Privacy

The in-house videos and clinical records are not shared because they contain sensitive health
information. The released in-house feature table is a de-identified processed artifact and does
not contain direct identifiers or source paths. Public CARE-PD data should be accessed from the
official CARE-PD data provider under its applicable terms.
