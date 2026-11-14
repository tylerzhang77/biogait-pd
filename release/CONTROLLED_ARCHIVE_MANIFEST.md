# Controlled reproducibility archive manifest

This file specifies the materials that must accompany the public GitHub release if the Editor or
reviewers are given access through a confidential or controlled archive. It is a manifest template,
not evidence that the archive has already been created.

## Required files

| Item | Required content | Privacy / release note |
|---|---|---|
| `frozen_configs/` | Exact JSON/YAML configuration consumed by each reported experiment | Remove local absolute paths |
| `split_manifests/` | Participant-disjoint adaptation splits and held-out-cohort unit manifests | Use approved pseudonymous identifiers |
| `selected_features/` | Per-fold and per-expert selected-feature lists, including gate candidate columns | Feature names only where possible |
| `predictions/` | Per-unit labels, probabilities, decoded classes and model/config identifiers | Controlled access if participant linkage remains possible |
| `model_checkpoints/` | Serialized fitted experts/checkpoints and version metadata | Controlled access if derived from restricted cohort data |
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
