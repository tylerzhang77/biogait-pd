# BioGait-PD reproducibility release manifest

Release identifier: `biogait-pd-paper-revision-0.2.4`
Git tag: `v0.2.4`
Release commit: the commit referenced by tag `v0.2.4`
Permanent archive DOI: <https://doi.org/10.5281/zenodo.24505794>

This directory contains the public, machine-readable result tables and the frozen protocol
configuration used to organize the manuscript revision. It is designed to be paired with a
versioned Git tag and a permanent archive record before publication.

## Included in this public release

- `configs/paper_final_config.json`: label mapping, descriptor-selection budgets, target-data
  access, decoding and bootstrap settings.
- `machine_readable/table_S1.csv` through `table_S27.csv`, including panel-specific refit files:
  machine-readable companions to the Supplementary Information tables.
- `machine_readable/source_*.csv`: aggregate audit tables used to generate selected manuscript
  and Supplementary results. These files contain summary statistics, not raw videos or clinical
  records.
- `reproducibility_assets/`: author-authorized participant-level predictions, adaptation split
  assignments, selected-feature inventories and provenance manifests.
- `reproducibility_assets/data/inhouse_features_segment_level_1069.csv`: de-identified,
  segment-level processed features for 1,069 segments from 150 in-house participants, with 1,258
  model-derived features, generated anonymous patient codes and mapped gait labels.
- `reproducibility_assets/data/segment_feature_table_manifest.json`: source provenance, dimensions,
  removed metadata fields and checksums for the released feature table.

The released feature-table CSV has SHA-256
`452589c03e8024772189a1d0eb1b3f72ef4d291d2220eaca1a07d62634a0b714` and contains 1,069 rows by
1,261 columns. The manifest records the restricted source-table hash
`94c94f4c739aae47d771f0a26d00a38b60dc1bedffdd2dd521a0f5dec5aed11c`.
- The executable Python entry points under `scripts/` and reusable package code under `src/`.
- Input schemas under `examples/` and data-placement instructions under `docs/DATA.md`.

## Materials requiring a controlled or confidential archive

The public repository does not contain raw videos, in-house clinical records, direct participant
identifiers or large motion files. The released feature table is processed and de-identified. The serialized reconstructed model bundles and their checksums
are included under `release/reproducibility_assets/models/reconstructed/`. The following item
still requires a controlled or confidential archive:

- the exact restricted input tables needed to regenerate participant-level predictions.

The checkpoint manifests identify the models as `reconstructed_verified`, because the historical
fitted estimators were not preserved. They were refitted from the frozen protocol and verified
against the archived predictions.

The final response letter must name the actual archive or release identifier used for these
materials. This manifest does not itself constitute a confidential archive.

## Verification before publication

Before creating the permanent release, verify the clean-environment installation, exact
commit/tag, package versions, data-release version, file checksums, software license and the
permitted access route for restricted materials. Do not claim that restricted items are publicly
available unless the approved archive has been created.
