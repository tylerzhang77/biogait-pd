from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "internal"))

from optimize_biogait_moe_elegant_advanced import entropy_alpha, fuse, normalize_rows, pred_from_thresholds, source_weights  # noqa: E402
from run_carepd_fivecohort_main150_domainnorm_moe import build_experts, domain_robust_normalize  # noqa: E402
from run_carepd_hierarchical_local_gate_role_moe import weighted_probability_sum  # noqa: E402
from run_carepd_hierarchical_source_role_moe import get_role_dict, source_subject_counts  # noqa: E402
from run_carepd_similarity_scope_role_moe import source_similarity_ranking  # noqa: E402
from run_paper_lodo_v2_branch_hier_moe import CAREPD_SCOPES, META_COLS, concept_groups_v2, load_v2_features  # noqa: E402


OUT_DIR = ROOT / "paper" / "res" / "inhouse_external_subject_level"
LABELS = [0, 1, 2]
FINAL = {
    "beta": 0.2,
    "source_weight_scheme": "similarity_only",
    "kappa": 25.0,
    "tau": 1.0,
    "reliability_power": 1.0,
    "uniform_blend": 0.25,
    "fusion_mode": "arithmetic",
    "temperature": 0.8,
    "alpha_mode": "source_when_uncertain",
    "alpha_low": 0.6,
    "alpha_high": 0.95,
    "threshold_01": 0.65,
    "threshold_12": 1.45,
}


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "acceptable_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, *, n_boot: int = 1000, seed: int = 20260604) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: dict[str, list[float]] = {k: [] for k in metric_values(y_true, y_pred)}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        row = metric_values(y_true[idx], y_pred[idx])
        for key, value in row.items():
            values[key].append(value)
    return {key: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) for key, vals in values.items()}


def source_ebm_shrunk(
    population_roles: dict[str, np.ndarray],
    source_roles: dict[str, dict[str, np.ndarray]],
    weights: dict[str, float],
    counts: dict[str, int],
    kappa: float,
) -> np.ndarray:
    items = {}
    for source, roles in source_roles.items():
        eta = counts[source] / (counts[source] + kappa)
        items[source] = eta * roles["nonlinear_ebm"] + (1.0 - eta) * population_roles["nonlinear_ebm"]
    return weighted_probability_sum(items, weights)


def aggregate_subjects(segment_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, g in segment_pred.groupby("subject", sort=False):
        p = g[[f"proba_{c}" for c in LABELS]].to_numpy(float).mean(axis=0)
        labels = sorted(g["gait_label"].astype(int).unique().tolist())
        rows.append(
            {
                "dataset": "main150",
                "subject": subject,
                "gait_label": int(labels[0]) if len(labels) == 1 else -1,
                "n_segments": int(len(g)),
                **{f"proba_{c}": float(p[c]) for c in LABELS},
            }
        )
    out = pd.DataFrame(rows)
    proba = out[[f"proba_{c}" for c in LABELS]].to_numpy(float)
    out["pred_ordinal_risk"] = pred_from_thresholds(proba, FINAL["threshold_01"], FINAL["threshold_12"])
    out["pred_argmax"] = np.argmax(proba, axis=1)
    return out


def add_ci(row: dict, y_true: np.ndarray, y_pred: np.ndarray, seed_key: str) -> dict:
    cis = bootstrap_ci(y_true, y_pred, seed=20260604 + zlib.adler32(seed_key.encode("utf-8")) % 100000)
    for key, (lo, hi) in cis.items():
        row[f"{key}_ci_low"] = lo
        row[f"{key}_ci_high"] = hi
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full, cols = load_v2_features()
    groups = concept_groups_v2(cols)

    raw_train = full[full["dataset"].isin(CAREPD_SCOPES)].copy()
    raw_test = full[full["dataset"] == "main150"].copy()
    meta = raw_test[META_COLS].copy()

    counts = source_subject_counts(raw_train)
    ranking = source_similarity_ranking(raw_train, raw_test, groups)
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
    print(f"[inhouse-external] ranking={ranking}", flush=True)
    print(f"[inhouse-external] weights={weights} alpha={alpha:.4f}", flush=True)

    train_z, test_z = domain_robust_normalize(raw_train, raw_test, cols, "domain_z")
    population_roles = get_role_dict("population", build_experts(train_z, test_z, cols, groups))

    source_roles = {}
    for source, _ in ranking:
        source_train = raw_train[raw_train["dataset"] == source].copy()
        print(
            f"  source {source}: n_segments={len(source_train)}, labels={source_train['gait_label'].value_counts().sort_index().to_dict()}",
            flush=True,
        )
        source_train_z, source_test_z = domain_robust_normalize(source_train, raw_test, cols, "domain_z")
        source_roles[source] = get_role_dict(f"{source.replace('-', '_')}_source", build_experts(source_train_z, source_test_z, cols, groups))

    p_pop = normalize_rows((1.0 - FINAL["beta"]) * population_roles["nonlinear_ebm"] + FINAL["beta"] * population_roles["linear_margin_correction"])
    p_source = source_ebm_shrunk(population_roles, source_roles, weights, counts, FINAL["kappa"])
    p_final = fuse(p_pop, p_source, alpha, FINAL["fusion_mode"], FINAL["temperature"])

    segment_pred = meta.copy()
    for c in LABELS:
        segment_pred[f"proba_{c}"] = p_final[:, c]
    segment_pred["pred_ordinal_risk"] = pred_from_thresholds(p_final, FINAL["threshold_01"], FINAL["threshold_12"])
    segment_pred["pred_argmax"] = np.argmax(p_final, axis=1)
    segment_pred["correct_ordinal_risk"] = segment_pred["pred_ordinal_risk"].astype(int) == segment_pred["gait_label"].astype(int)
    segment_pred["correct_argmax"] = segment_pred["pred_argmax"].astype(int) == segment_pred["gait_label"].astype(int)

    subject_pred = aggregate_subjects(segment_pred)
    subject_eval = subject_pred[subject_pred["gait_label"] >= 0].copy()

    rows = []
    for level, df, label_col in [
        ("segment", segment_pred, "gait_label"),
        ("subject", subject_eval, "gait_label"),
    ]:
        y_true = df[label_col].to_numpy(int)
        for pred_col in ["pred_ordinal_risk", "pred_argmax"]:
            y_pred = df[pred_col].to_numpy(int)
            row = {
                "train_scope": "CAREPD_four_public_cohorts",
                "test_scope": "main150_inhouse",
                "level": level,
                "decision_rule": pred_col.replace("pred_", ""),
                "n_units": int(len(df)),
                **metric_values(y_true, y_pred),
            }
            rows.append(add_ci(row, y_true, y_pred, f"{level}|{pred_col}"))

    metrics = pd.DataFrame(rows)
    label_audit = {
        "train_label_distribution": raw_train.groupby(["dataset", "gait_label"]).size().unstack(fill_value=0).to_dict(),
        "test_segment_label_distribution": raw_test["gait_label"].value_counts().sort_index().to_dict(),
        "test_subject_label_distribution": subject_eval["gait_label"].value_counts().sort_index().to_dict(),
        "source_ranking": ranking,
        "source_weights": weights,
        "alpha": alpha,
        "final_config": FINAL,
    }

    segment_pred.to_csv(OUT_DIR / "inhouse_segment_predictions_carepd_train.csv", index=False, encoding="utf-8-sig")
    subject_pred.to_csv(OUT_DIR / "inhouse_subject_predictions_carepd_train.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT_DIR / "inhouse_external_metrics.csv", index=False, encoding="utf-8-sig")
    with (OUT_DIR / "inhouse_external_label_and_gate_audit.json").open("w", encoding="utf-8") as f:
        json.dump(label_audit, f, ensure_ascii=False, indent=2)

    print(json.dumps({"out_dir": str(OUT_DIR), "n_segments": len(segment_pred), "n_subjects": len(subject_eval)}, ensure_ascii=False, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
