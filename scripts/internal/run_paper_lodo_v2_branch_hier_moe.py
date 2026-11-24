from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "internal"))

from run_carepd_branch_hier_moe_segment import (  # noqa: E402
    local_source_gate_branch,
    normalized_entropy,
    population_local_branch,
)
from run_carepd_fivecohort_main150_domainnorm_moe import build_experts, domain_robust_normalize  # noqa: E402
from run_carepd_hierarchical_source_role_moe import (  # noqa: E402
    get_role_dict,
    similarity_reliability_weights,
    source_subject_counts,
)
from run_carepd_or_da_ebm_plus import concept_groups as base_concept_groups  # noqa: E402
from run_carepd_paper_baseline import LABELS, patient_aggregate  # noqa: E402
from run_carepd_similarity_scope_role_moe import source_similarity_ranking  # noqa: E402


FEATURE_ROOT = ROOT / "outputs" / "axis_corrected_features_v3_fpsfix"
FEATURE_TABLE = Path(
    os.environ.get(
        "BIOGAIT_FEATURE_TABLE",
        str(FEATURE_ROOT / "fivecohort_axis_corrected_common_feature_table.csv"),
    )
)
OUT_DIR = ROOT / "paper" / "res" / "lodo_v2_branch_hier_moe"
PRED_DIR = OUT_DIR / "predictions"

CAREPD_SCOPES = ["3DGait", "BMCLab", "PD-GaM", "T-SDU-PD"]
ALL_SCOPES = ["3DGait", "BMCLab", "PD-GaM", "T-SDU-PD", "main150"]
BENCHMARK = {"3DGait": 0.41, "BMCLab": 0.50, "PD-GaM": 0.51, "T-SDU-PD": 0.49}
META_COLS = [
    "dataset",
    "sample_id",
    "subject",
    "group_id",
    "subject_label_id",
    "trial",
    "retarget_npz",
    "updrs_gait_raw",
    "gait_label",
    "medication",
    "other",
    "frames",
    "fps",
]

FINAL_SOURCE_CFG = {"kappa": 25.0, "tau": 1.0, "uniform_blend": 0.25, "reliability_power": 1.0}
FINAL_PL_GATE_CFG = {
    "top_k": 3,
    "gap": 0.25,
    "risk_temperature": 0.1,
    "confidence_bonus": 0.08,
    "ebm_prior": 0.25,
    "ordinal_prior": 0.0,
    "coverage_prior": 0.0,
    "margin_shift_bonus": 0.5,
}
FINAL_LOCAL_GATE_CFG = {
    "top_k": 1,
    "gap": 0.50,
    "risk_temperature": 0.1,
    "confidence_bonus": 0.08,
    "ebm_prior": 0.50,
    "ordinal_prior": 0.25,
    "coverage_prior": 0.0,
    "margin_shift_bonus": 2.25,
    "base_width": 0.05,
    "uncertainty_scale": 0.5,
}
FINAL_ALPHA_LOW = 0.6
FINAL_ALPHA_HIGH = 0.95
FINAL_ALPHA_MODE = "source_when_uncertain"
# Kept for older component-ablation helpers that import this symbol.
FINAL_ALPHA_LOCAL = 0.625
FINAL_MODEL_NAME = "BranchHierMoE_v2_k25_tau1_blend0.25_rp1_PL0_HLG0_alphaentropy_0.6_0.95"


def markdown_table(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def metric_dict(y_true, y_pred, prefix: str) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}acceptable_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        f"{prefix}weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in set(META_COLS)]


def load_v2_features() -> tuple[pd.DataFrame, list[str]]:
    full = pd.read_csv(FEATURE_TABLE, low_memory=False)
    full["gait_label"] = full["gait_label"].astype(int)
    for col in META_COLS:
        if col not in full.columns:
            full[col] = ""
    cols = feature_cols(full)
    cols = [c for c in cols if full[c].notna().any()]
    return full, cols


def concept_groups_v2(cols: list[str]) -> dict[str, list[str]]:
    groups = base_concept_groups(cols)
    if "amplitude_rom" in groups:
        extra = [c for c in cols if "cycle_horizontal_rom" in c or c == "root_acc_rms"]
        groups["amplitude_rom"] = sorted(set(groups["amplitude_rom"]) | set(extra))
    return groups


def evaluate_predictions(
    train_setting: str,
    scope: str,
    meta: pd.DataFrame,
    proba: np.ndarray,
    pred_dir: Path,
) -> dict:
    pred = np.argmax(proba, axis=1)
    pred_df = meta.copy()
    pred_df["pred"] = pred
    for c in LABELS:
        pred_df[f"proba_{c}"] = proba[:, c]

    unit_df = patient_aggregate(pred_df)
    unit_df = unit_df[unit_df["gait_label"] >= 0].copy()

    row = {
        "train_setting": train_setting,
        "test_scope": scope,
        "model": FINAL_MODEL_NAME,
        "n_test_segments": int(len(meta)),
        "n_subject_label_units": int(len(unit_df)),
        **metric_dict(meta["gait_label"].to_numpy(int), pred, "segment_"),
        **metric_dict(unit_df["gait_label"].to_numpy(int), unit_df["pred"].to_numpy(int), "subject_label_"),
    }
    row["segment_benchmark_macro_f1"] = BENCHMARK.get(scope, np.nan)
    row["segment_delta_vs_benchmark"] = (
        row["segment_macro_f1"] - row["segment_benchmark_macro_f1"] if scope in BENCHMARK else np.nan
    )

    setting_dir = pred_dir / train_setting
    setting_dir.mkdir(parents=True, exist_ok=True)
    safe = f"{scope}__{FINAL_MODEL_NAME}".replace("/", "_")
    pred_df.to_csv(setting_dir / f"{safe}_segment_predictions.csv", index=False, encoding="utf-8-sig")
    unit_df.to_csv(setting_dir / f"{safe}_subject_label_predictions.csv", index=False, encoding="utf-8-sig")
    seg_cm = confusion_matrix(meta["gait_label"].to_numpy(int), pred, labels=LABELS)
    pd.DataFrame(seg_cm, index=[f"true_{c}" for c in LABELS], columns=[f"pred_{c}" for c in LABELS]).to_csv(
        setting_dir / f"{safe}_segment_confusion.csv", encoding="utf-8-sig"
    )
    unit_cm = confusion_matrix(unit_df["gait_label"].to_numpy(int), unit_df["pred"].to_numpy(int), labels=LABELS)
    pd.DataFrame(unit_cm, index=[f"true_{c}" for c in LABELS], columns=[f"pred_{c}" for c in LABELS]).to_csv(
        setting_dir / f"{safe}_subject_label_confusion.csv", encoding="utf-8-sig"
    )
    return row


def run_one_scope(
    full: pd.DataFrame,
    cols: list[str],
    groups: dict[str, list[str]],
    train_setting: str,
    scope: str,
    include_main150: bool,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    raw_train = full[full["dataset"] != scope].copy()
    if not include_main150:
        raw_train = raw_train[raw_train["dataset"] != "main150"].copy()
    raw_test = full[full["dataset"] == scope].copy()
    meta = raw_test[META_COLS].copy()

    counts = source_subject_counts(raw_train)
    ranking = source_similarity_ranking(raw_train, raw_test, groups)
    print(f"[paper-lodo-v2] {train_setting} held-out={scope} ranking={ranking}", flush=True)

    train_z, test_z = domain_robust_normalize(raw_train, raw_test, cols, "domain_z")
    population_roles = get_role_dict("population", build_experts(train_z, test_z, cols, groups))

    source_roles = {}
    for source, _shift in ranking:
        source_train = raw_train[raw_train["dataset"] == source].copy()
        print(
            f"  source expert {source}: n_segments={len(source_train)}, n_units={counts[source]}",
            flush=True,
        )
        source_train_z, source_test_z = domain_robust_normalize(source_train, raw_test, cols, "domain_z")
        source_roles[source] = get_role_dict(
            f"{source.replace('-', '_')}_source",
            build_experts(source_train_z, source_test_z, cols, groups),
        )

    source_weights = similarity_reliability_weights(ranking, counts, **FINAL_SOURCE_CFG)
    entropy = normalized_entropy(source_weights)
    alpha = FINAL_ALPHA_LOW + (FINAL_ALPHA_HIGH - FINAL_ALPHA_LOW) * entropy

    p_pl, info_pl = population_local_branch(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        FINAL_PL_GATE_CFG,
    )
    p_local, info_local = local_source_gate_branch(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        FINAL_LOCAL_GATE_CFG,
    )
    p_final = (1.0 - alpha) * p_pl + alpha * p_local

    result = evaluate_predictions(train_setting, scope, meta, p_final, PRED_DIR)
    diag = pd.DataFrame(
        [
            {
                "train_setting": train_setting,
                "test_scope": scope,
                "model": FINAL_MODEL_NAME,
                "alpha_local_branch": alpha,
                "source_weight_entropy": entropy,
                "pl_selected_expert": info_pl.get("selected_expert", ""),
                "local_selected_expert": info_local.get("local_selected_expert", ""),
                **{f"pl_{k}": v for k, v in info_pl.items() if isinstance(v, (int, float, str, bool))},
                **{f"local_{k}": v for k, v in info_local.items() if isinstance(v, (int, float, str, bool))},
            }
        ]
    )
    weight_rows = []
    shift_map = dict(ranking)
    for ds, w in source_weights.items():
        weight_rows.append(
            {
                "train_setting": train_setting,
                "test_scope": scope,
                "source_dataset": ds,
                "source_weight": w,
                "n_units": counts[ds],
                "concept_shift": shift_map.get(ds, np.nan),
                "source_weight_entropy": entropy,
            }
        )
    return result, diag, pd.DataFrame(weight_rows)


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting, sub in results.groupby("train_setting", sort=False):
        carepd = sub[sub["test_scope"].isin(CAREPD_SCOPES)].copy()
        rows.append(
            {
                "train_setting": setting,
                "n_scopes_total": int(len(sub)),
                "mean_segment_macro_f1_all_tested": float(sub["segment_macro_f1"].mean()),
                "mean_segment_macro_f1_carepd": float(carepd["segment_macro_f1"].mean()),
                "min_segment_macro_f1_carepd": float(carepd["segment_macro_f1"].min()),
                "mean_segment_delta_vs_benchmark_carepd": float(carepd["segment_delta_vs_benchmark"].mean()),
                "min_segment_delta_vs_benchmark_carepd": float(carepd["segment_delta_vs_benchmark"].min()),
                "pass_all_carepd_benchmarks": bool((carepd["segment_delta_vs_benchmark"] >= 0).all()),
                "mean_segment_accuracy_carepd": float(carepd["segment_accuracy"].mean()),
                "mean_segment_acceptable_accuracy_carepd": float(carepd["segment_acceptable_accuracy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    full, cols = load_v2_features()
    groups = concept_groups_v2(cols)

    pd.DataFrame({"feature": cols}).to_csv(OUT_DIR / "v2_common_feature_columns.csv", index=False, encoding="utf-8-sig")
    with (OUT_DIR / "fixed_model_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": FINAL_MODEL_NAME,
                "feature_table": str(FEATURE_TABLE),
                "n_features": len(cols),
                "source_cfg": FINAL_SOURCE_CFG,
                "pl_gate_cfg": FINAL_PL_GATE_CFG,
                "local_gate_cfg": FINAL_LOCAL_GATE_CFG,
                "alpha_low": FINAL_ALPHA_LOW,
                "alpha_high": FINAL_ALPHA_HIGH,
                "alpha_mode": FINAL_ALPHA_MODE,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    experiment_plan = []
    for scope in CAREPD_SCOPES:
        experiment_plan.append(("CAREPD-only", scope, False))
    for scope in ALL_SCOPES:
        experiment_plan.append(("five-cohort", scope, True))

    rows = []
    diag_rows = []
    weight_rows = []
    for train_setting, scope, include_main150 in experiment_plan:
        result, diag, weights = run_one_scope(full, cols, groups, train_setting, scope, include_main150)
        rows.append(result)
        diag_rows.append(diag)
        weight_rows.append(weights)
        pd.DataFrame(rows).to_csv(OUT_DIR / "paper_lodo_v2_results_partial.csv", index=False, encoding="utf-8-sig")

    results = pd.DataFrame(rows)
    diagnostics = pd.concat(diag_rows, ignore_index=True)
    source_weights = pd.concat(weight_rows, ignore_index=True)
    aggregate = aggregate_results(results)

    results.to_csv(OUT_DIR / "paper_lodo_v2_results.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(OUT_DIR / "paper_lodo_v2_aggregate.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(OUT_DIR / "paper_lodo_v2_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    source_weights.to_csv(OUT_DIR / "paper_lodo_v2_source_weights.csv", index=False, encoding="utf-8-sig")

    report = [
        "# Paper LODO Results with Axis-Corrected v2 Features",
        "",
        f"Fixed model: `{FINAL_MODEL_NAME}`",
        "",
        f"Feature table: `{FEATURE_TABLE}`",
        "",
        f"Number of common features: `{len(cols)}`",
        "",
        "## Aggregate",
        "",
        markdown_table(aggregate),
        "",
        "## Per-Dataset Segment-Level Results",
        "",
        markdown_table(
            results[
                [
                    "train_setting",
                    "test_scope",
                    "n_test_segments",
                    "segment_macro_f1",
                    "segment_accuracy",
                    "segment_acceptable_accuracy",
                    "segment_benchmark_macro_f1",
                    "segment_delta_vs_benchmark",
                    "subject_label_macro_f1",
                ]
            ]
        ),
        "",
        "## Gate Diagnostics",
        "",
        markdown_table(
            diagnostics[
                [
                    "train_setting",
                    "test_scope",
                    "alpha_local_branch",
                    "source_weight_entropy",
                    "pl_selected_expert",
                    "local_selected_expert",
                ]
            ]
        ),
    ]
    (OUT_DIR / "PAPER_LODO_V2_BRANCH_HIER_MOE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\nAggregate:")
    print(aggregate.to_string(index=False))
    print("\nPer-dataset:")
    print(results[["train_setting", "test_scope", "segment_macro_f1", "segment_accuracy", "segment_acceptable_accuracy", "segment_delta_vs_benchmark"]].to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
