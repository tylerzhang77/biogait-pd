"""Verify reconstructed checkpoint role probabilities and final predictions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "internal"))

from run_biogait_moe_final_ablation_report import (  # noqa: E402
    FINAL,
    build_probabilities,
)
from optimize_biogait_moe_elegant_advanced import entropy_alpha, source_weights  # noqa: E402
from run_carepd_hierarchical_source_role_moe import source_subject_counts  # noqa: E402
from run_carepd_similarity_scope_role_moe import source_similarity_ranking  # noqa: E402
from run_paper_lodo_v2_branch_hier_moe import (  # noqa: E402
    CAREPD_SCOPES,
    concept_groups_v2,
    load_v2_features,
)


CHECKPOINT_ROOT = Path(os.environ.get("BIOGAIT_CHECKPOINT_ROOT", str(ROOT / "paper" / "res" / "reconstructed_checkpoints")))
SAVED_PREDICTIONS = Path(
    os.environ.get(
        "BIOGAIT_SAVED_PREDICTIONS",
        str(ROOT / "paper" / "res" / "final_moe_ablation" / "final_moe_ablation_segment_predictions.csv"),
    )
)
VARIANT = "final_similarity_gated_biogait_moe"


def load_roles(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def verify_one(full: pd.DataFrame, groups: dict[str, list[str]], setting: str, scope: str, saved: pd.DataFrame) -> dict:
    include_main150 = setting == "five-cohort"
    raw_train = full[full["dataset"] != scope].copy()
    if not include_main150:
        raw_train = raw_train[raw_train["dataset"] != "main150"].copy()
    raw_test = full[full["dataset"] == scope].copy()
    ranking = source_similarity_ranking(raw_train, raw_test, groups)
    counts = source_subject_counts(raw_train)
    pop_dir = CHECKPOINT_ROOT / setting / scope / "population"
    population_roles = load_roles(pop_dir / "role_probabilities.npz")
    source_roles = {}
    for source, _shift in ranking:
        source_roles[source] = load_roles(CHECKPOINT_ROOT / setting / scope / f"source_{source.replace('-', '_')}" / "role_probabilities.npz")
    weights = source_weights(
        ranking,
        counts,
        scheme=FINAL["source_weight_scheme"],
        kappa=FINAL["kappa"],
        tau=FINAL["tau"],
        reliability_power=FINAL["reliability_power"],
        uniform_blend=FINAL["uniform_blend"],
    )
    alpha = entropy_alpha(weights, FINAL["alpha_low"], FINAL["alpha_high"], FINAL["alpha_mode"])
    proba, decision = build_probabilities(population_roles, source_roles, weights, counts, VARIANT, alpha)
    pred = np.argmax(proba, axis=1) if decision == "argmax" else np.where((proba[:, 1] + 2 * proba[:, 2]) < FINAL["threshold_01"], 0, np.where((proba[:, 1] + 2 * proba[:, 2]) < FINAL["threshold_12"], 1, 2))
    target = saved[(saved["train_setting"] == setting) & (saved["test_center"] == scope) & (saved["variant"] == VARIANT)].copy()
    target = target.set_index("sample_id").loc[raw_test["sample_id"].astype(str)]
    max_prob_diff = float(np.max(np.abs(proba - target[["proba_0", "proba_1", "proba_2"]].to_numpy(float))))
    pred_equal = bool(np.array_equal(pred, target["pred"].to_numpy(int)))
    return {"train_setting": setting, "test_center": scope, "n": int(len(proba)), "max_abs_probability_difference": max_prob_diff, "predictions_equal": pred_equal, "decision": decision}


def main() -> None:
    full, _cols = load_v2_features()
    groups = concept_groups_v2(_cols)
    saved = pd.read_csv(SAVED_PREDICTIONS, low_memory=False)
    rows = []
    for setting in ["CAREPD-only", "five-cohort"]:
        for scope in CAREPD_SCOPES:
            rows.append(verify_one(full, groups, setting, scope, saved))
    out = pd.DataFrame(rows)
    out.to_csv(CHECKPOINT_ROOT / "final_prediction_verification.csv", index=False)
    print(out.to_string(index=False))
    print(json.dumps({"all_predictions_equal": bool(out["predictions_equal"].all()), "max_probability_difference": float(out["max_abs_probability_difference"].max())}, indent=2))


if __name__ == "__main__":
    main()
