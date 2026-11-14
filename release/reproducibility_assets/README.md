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

No serialized trained-model checkpoint was present in the source revision audit package used to
assemble this release. The executable scripts and frozen protocol are included, but the final
release must not claim checkpoint availability until the fitted models have been exported and
added under a versioned `models/` directory or an attached GitHub/Zenodo release asset.
