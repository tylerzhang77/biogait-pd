# GitHub Release Checklist

Before making the repository public, confirm:

- [ ] Add a license file selected by all contributors and institutions.
- [x] Add a release identifier and frozen paper protocol under `release/`.
- [x] Add machine-readable Supplementary table companions and aggregate audit tables.
- [x] Add the permanent Git tag `v0.2.4` after the checkpoint-containing commit is pushed.
- [x] Add author-authorized split manifests, per-unit predictions and selected-feature inventories to `release/reproducibility_assets/`.
- [x] Export and add reconstructed verified fold-specific model checkpoints, with manifests and hashes.
- [x] Add the de-identified in-house segment-level feature table and provenance manifest.
- [x] Add the permanent software-archive DOI to the release documentation.
- [x] Add the official CARE-PD data-access URL, DOI and benchmark-code link.
- [x] Confirm that the de-identified segment-level feature table, splits and predictions can be shared.
- [x] Confirm that no raw videos, patient identifiers, clinical records or cached `.npz` files are committed; serialized checkpoints are intentionally included under `release/reproducibility_assets/models/reconstructed/`.
- [ ] Run `rg "C:\\\\Users|D:\\\\|\\.xlsx|\\.npz|\\.pkl|\\.pt|\\.pth"` before release.
- [ ] Test a fresh installation in a clean environment.
