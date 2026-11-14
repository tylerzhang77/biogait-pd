# Reproducibility assets

This directory contains the participant-level and audit-level assets authorized for the public
BioGait-PD release. The files are organized by role:

- `predictions/`: identity-matched recording/state predictions, baseline comparisons and the
  in-house local-only control outputs.
- `splits/`: adaptation split assignments, split-level metrics and paired comparisons.
- `feature_selection/`: per-fold/per-expert selected-feature inventories and selector audit
  manifests.
- `configs/`: provenance, protocol and prediction-audit manifests.

The identifiers in these files are analysis identifiers used by the study pipeline. No raw video,
clinical record, contact information or directly identifying metadata is included in this release.
The release of participant-level derived files follows the author-confirmed data-release decision;
the final GitHub release should still be checked against the governing ethics and data-use terms.

The aggregate machine-readable companions to Supplementary Tables 1–27 remain in
`../machine_readable/`. The source files were copied from the revision audit package and retain
their original column names so that table and figure lineage can be checked.

## Model files

Reconstructed verified fold-specific model bundles are provided under `models/reconstructed/`.
Their manifests record the refitting protocol, software versions, selected features and hashes.
The original fitted estimators were not preserved in the historical revision audit package; the
released bundles were refitted from the frozen v3 FPS-fix protocol and verified against the
archived predictions. This qualification is retained in the release documentation and response
letter.
