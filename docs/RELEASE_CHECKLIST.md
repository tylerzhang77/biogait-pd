# GitHub Release Checklist

Before making the repository public, confirm:

- [ ] Add a license file selected by all contributors and institutions.
- [ ] Replace the provisional BibTeX entry in `README.md` after the manuscript DOI is available.
- [x] Add the official CARE-PD data-access URL, DOI and benchmark-code link.
- [ ] Confirm whether de-identified derived features, splits and predictions can be shared.
- [ ] Confirm that no raw videos, patient identifiers, clinical records, model checkpoints or cached `.npz` files are committed.
- [ ] Run `rg "C:\\\\Users|D:\\\\|\\.xlsx|\\.npz|\\.pkl|\\.pt|\\.pth"` before release.
- [ ] Test a fresh installation in a clean environment.
