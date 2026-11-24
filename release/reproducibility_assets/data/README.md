# In-house segment-level feature table

This directory contains the de-identified processed feature table used for the in-house
(`main150`) cohort in the paper's final v3 FPS-corrected analysis.

`inhouse_features_segment_level_1069.csv` contains 1,069 gait-segment rows from 150 participants,
1,258 model-derived feature columns, an anonymous `patient_id` coded from 1 to 150, a generated
`segment_index`, and the mapped three-class `gait_label`. The same anonymous patient code is used
for all segments from one participant. The released codes cannot be mapped back to the original
identifiers.

The table is a processed data artifact. Raw videos, clinical records, source paths and direct
participant identifiers remain unavailable because of privacy and institutional data-use
restrictions. The model operates on segment rows first and aggregates predictions at the
recording/state or participant level according to the evaluation protocol.

The CSV uses a simplified public schema. The internal five-cohort loader also expects cohort and
recording metadata, so this file requires a schema adapter before it can be used as a drop-in
replacement for the restricted internal feature table. The provenance, dimensions and SHA-256
values are recorded in `segment_feature_table_manifest.json`.
