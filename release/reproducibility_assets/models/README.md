# Trained model files

This directory contains reconstructed fold-specific model bundles for the eight CARE-PD
leave-one-dataset-out analyses used in the revision: four `CAREPD-only` folds and four
`five-cohort` folds. Each `experts.joblib` file contains the six fitted implementation-level
pipelines used to construct the three role-level experts:

- additive EBM (`k=150`);
- two calibrated linear SVC margin models (`k=320`, `C=0.1` and `C=1.0`);
- logistic and ordinal cumulative models (`k=320`);
- concept-sparse ordinal model.

The accompanying JSON files record feature columns, selected features, concept columns, model
parameters, software versions and SHA-256 provenance. `final_prediction_verification.csv` records
the comparison between predictions generated from the serialized model bundles and the archived
manuscript prediction file. Across all eight folds, the predicted classes were identical and the
maximum absolute probability difference was `1.11e-16` (floating-point round-off).

The historical revision package did not contain the original serialized estimators. These files
were refitted from the frozen v3 FPS-fix feature table and protocol, and are therefore labelled
`reconstructed_verified`. The package should retain this qualification in the manuscript and
response letter. The input feature table itself is not redistributed; its SHA-256 identifier and
the required input schema are recorded in each fold manifest.

The top-level `CHECKPOINT_RELEASE_MANIFEST.json` lists all 36 serialized bundles and their hashes.
The bundles can be loaded with `joblib.load` after adding `scripts/internal/` to `PYTHONPATH`,
which provides the custom estimator classes referenced by the serialized pipelines.
