# Controlled reproducibility archive manifest

This file specifies the remaining materials that must accompany the public GitHub release if the
Editor or reviewers are given access through a confidential or controlled archive. Public copies
of the split assignments, per-unit predictions, selected-feature inventories and reconstructed
model bundles are now under `reproducibility_assets/`. This file remains a manifest for restricted
input materials and does not itself constitute a confidential archive.

## Required files

| Item | Required content | Privacy / release note |
|---|---|---|
| `frozen_configs/` | Exact JSON/YAML configuration consumed by each reported experiment | Remove local absolute paths |
| `split_manifests/` | Participant-disjoint adaptation splits and held-out-cohort unit manifests | Public copies are in `reproducibility_assets/splits/`; add any missing manifests here |
| `selected_features/` | Per-fold and per-expert selected-feature lists, including gate candidate columns | Public copies are in `reproducibility_assets/feature_selection/` |
| `predictions/` | Per-unit labels, probabilities, decoded classes and model/config identifiers | Public copies are in `reproducibility_assets/predictions/` |
| `model_checkpoints/` | Serialized reconstructed fitted experts/checkpoints and version metadata | Public copies are in `reproducibility_assets/models/reconstructed/` |
| `source_tables/` | Machine-readable rows used to generate every manuscript and Supplementary table/figure | Public aggregate tables are already in `release/machine_readable/` |
| `checksums/` | SHA-256 hashes for every archive member | Generate after the archive is frozen |
| `README.md` | Exact commands, environment lockfile, data-release versions and access restrictions | Must name the permanent archive identifier |

## Archive completion fields

- Archive provider: `[TO BE COMPLETED]`
- Permanent identifier or DOI: `[TO BE COMPLETED]`
- Version/tag: `[TO BE COMPLETED]`
- Access route for the Editor and reviewers: `[TO BE COMPLETED]`
- Ethics/data-use approval for controlled sharing: `[TO BE COMPLETED]`
- Software license: `[TO BE COMPLETED]`
