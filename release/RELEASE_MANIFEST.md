# BioGait-PD reproducibility release manifest

Release identifier: `biogait-pd-paper-revision-0.2.1`
Git tag: `v0.2.1`
Release commit: the commit referenced by tag `v0.2.1`

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
- The executable Python entry points under `scripts/` and reusable package code under `src/`.
- Input schemas under `examples/` and data-placement instructions under `docs/DATA.md`.

## Materials requiring a controlled or confidential archive

The public repository does not contain raw videos, in-house clinical records, direct participant
identifiers or large motion files. The serialized reconstructed model bundles and their checksums
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
