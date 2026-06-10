from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "internal"))

from optimize_biogait_moe_elegant_advanced import entropy_alpha, fuse, normalize_rows  # noqa: E402
from run_carepd_bo_da_moe import make_linear_svc_expert  # noqa: E402
from run_carepd_fivecohort_main150_domainnorm_moe import domain_robust_normalize, make_ebm_expert  # noqa: E402
from run_carepd_hierarchical_local_gate_role_moe import weighted_probability_sum  # noqa: E402
from run_carepd_hierarchical_source_role_moe import source_subject_counts  # noqa: E402
from run_carepd_similarity_scope_role_moe import source_similarity_ranking  # noqa: E402
from run_paper_lodo_v2_branch_hier_moe import CAREPD_SCOPES, META_COLS, concept_groups_v2, load_v2_features  # noqa: E402


OUT_DIR = ROOT / "paper" / "res" / "carepd_benchmark_compatible_lodo"
PRED_DIR = OUT_DIR / "predictions"
BENCHMARK = {"3DGait": 0.41, "BMCLab": 0.50, "PD-GaM": 0.51, "T-SDU-PD": 0.49}
TRAIN_LABELS = [0, 1, 2, 3]
METRIC_LABELS = [0, 1, 2]
CAREPD_DATASETS = set(CAREPD_SCOPES)
EPS = 1e-9


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
    "threshold_23": 2.45,
}


def restore_carepd_raw_labels(full: pd.DataFrame) -> pd.DataFrame:
    """Use raw CAREPD score 3 for training; keep main150's existing 0/1/2 labels."""
    out = full.copy()
    raw = pd.to_numeric(out["updrs_gait_raw"], errors="coerce")
    carepd_mask = out["dataset"].isin(CAREPD_DATASETS) & raw.notna()
    out["raw_gait_label"] = out["gait_label"].astype(int)
    out.loc[carepd_mask, "raw_gait_label"] = raw.loc[carepd_mask].astype(int)
    out["gait_label_collapsed_2_3"] = out["gait_label"].astype(int)
    out["gait_label"] = out["raw_gait_label"].astype(int)
    out["subject_label_id"] = (
        out["dataset"].astype(str) + "::" + out["subject"].astype(str) + "::rawlabel" + out["gait_label"].astype(str)
    )
    return out


def align_proba(model, x_test: np.ndarray, labels: list[int]) -> np.ndarray:
    p = np.asarray(model.predict_proba(x_test), dtype=np.float64)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps["clf"], "classes_", None)
    if classes is None:
        if p.shape[1] == len(labels):
            return normalize_rows(p)
        raise RuntimeError("Cannot infer model classes for probability alignment.")
    classes = [int(c) for c in classes]
    out = np.zeros((p.shape[0], len(labels)), dtype=np.float64)
    for j, cls in enumerate(classes):
        if cls in labels:
            out[:, labels.index(cls)] = p[:, j]
    return normalize_rows(out)


def fit_predict_core_experts(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    x_train = train[cols].to_numpy(np.float64)
    y_train = train["gait_label"].to_numpy(int)
    x_test = test[cols].to_numpy(np.float64)
    experts = {
        "nonlinear_ebm": make_ebm_expert(k=150),
        "linear_margin_correction_c0p1": make_linear_svc_expert(C=0.1),
        "linear_margin_correction_c1": make_linear_svc_expert(C=1.0),
    }
    out: dict[str, np.ndarray] = {}
    for name, model in experts.items():
        m = clone(model)
        m.fit(x_train, y_train)
        out[name] = align_proba(m, x_test, TRAIN_LABELS)
    out["linear_margin_correction"] = normalize_rows(
        0.5 * out["linear_margin_correction_c0p1"] + 0.5 * out["linear_margin_correction_c1"]
    )
    return {
        "nonlinear_ebm": out["nonlinear_ebm"],
        "linear_margin_correction": out["linear_margin_correction"],
    }


def source_weights(
    ranking: list[tuple[str, float]],
    counts: dict[str, int],
    *,
    scheme: str,
    kappa: float,
    tau: float,
    reliability_power: float,
    uniform_blend: float,
) -> dict[str, float]:
    names = [name for name, _ in ranking]
    shifts = np.asarray([shift for _, shift in ranking], dtype=float)
    shifts = shifts - float(np.nanmin(shifts))
    rel = np.asarray([counts[n] / (counts[n] + kappa) for n in names], dtype=float)
    sim = np.exp(-shifts / max(tau, EPS))
    if scheme == "similarity_only":
        score = sim
    elif scheme == "similarity_reliability":
        score = sim * np.power(rel, reliability_power)
    elif scheme == "uniform":
        score = np.ones(len(names))
    else:
        raise ValueError(scheme)
    if not np.isfinite(score).all() or score.sum() <= 0:
        score = np.ones_like(score)
    score = score / score.sum()
    uniform = np.ones_like(score) / len(score)
    w = (1.0 - uniform_blend) * score + uniform_blend * uniform
    w = w / w.sum()
    return {name: float(value) for name, value in zip(names, w)}


def source_ebm(
    population_roles: dict[str, np.ndarray],
    source_roles: dict[str, dict[str, np.ndarray]],
    weights: dict[str, float],
    counts: dict[str, int],
    kappa: float,
) -> np.ndarray:
    items = {}
    for source, roles in source_roles.items():
        lam = counts[source] / (counts[source] + kappa)
        items[source] = lam * roles["nonlinear_ebm"] + (1.0 - lam) * population_roles["nonlinear_ebm"]
    return weighted_probability_sum(items, weights)


def pred_from_thresholds_4class(p: np.ndarray) -> np.ndarray:
    risk = p[:, 1] + 2.0 * p[:, 2] + 3.0 * p[:, 3]
    pred = np.zeros(len(risk), dtype=int)
    pred[risk >= FINAL["threshold_01"]] = 1
    pred[risk >= FINAL["threshold_12"]] = 2
    pred[risk >= FINAL["threshold_23"]] = 3
    return pred


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isin(y_true, METRIC_LABELS)
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "n_eval_segments": int(len(yt)),
        "n_excluded_score3_segments": int((y_true == 3).sum()),
        "macro_f1": float(f1_score(yt, yp, labels=METRIC_LABELS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_precision": float(precision_score(yt, yp, labels=METRIC_LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(yt, yp, labels=METRIC_LABELS, average="macro", zero_division=0)),
        "acceptable_accuracy": float(np.mean(np.abs(yt - yp) <= 1)),
        "predicted_score3_rate_on_eval": float(np.mean(yp == 3)),
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, *, n_boot: int = 1000, seed: int = 20260602) -> dict[str, tuple[float, float]]:
    mask = np.isin(y_true, METRIC_LABELS)
    yt = y_true[mask]
    yp = y_pred[mask]
    rng = np.random.default_rng(seed)
    n = len(yt)
    keys = ["macro_f1", "accuracy", "macro_precision", "macro_recall", "acceptable_accuracy"]
    values = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        row = metric_values(yt[idx], yp[idx])
        for key in keys:
            values[key].append(row[key])
    return {key: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) for key, vals in values.items()}


def run_one_fold(full: pd.DataFrame, cols: list[str], groups: dict[str, list[str]], train_setting: str, scope: str, include_main150: bool):
    raw_train = full[full["dataset"] != scope].copy()
    if not include_main150:
        raw_train = raw_train[raw_train["dataset"] != "main150"].copy()
    raw_test = full[full["dataset"] == scope].copy()
    meta = raw_test[META_COLS + ["raw_gait_label", "gait_label_collapsed_2_3"]].copy()
    y_true = raw_test["gait_label"].to_numpy(int)
    counts = source_subject_counts(raw_train)
    ranking = source_similarity_ranking(raw_train, raw_test, groups)
    print(f"[carepd-compatible-lodo] {train_setting} held-out={scope} ranking={ranking}", flush=True)

    train_z, test_z = domain_robust_normalize(raw_train, raw_test, cols, "domain_z")
    population_roles = fit_predict_core_experts(train_z, test_z, cols)
    source_roles = {}
    for source, _shift in ranking:
        source_train = raw_train[raw_train["dataset"] == source].copy()
        source_train_z, source_test_z = domain_robust_normalize(source_train, raw_test, cols, "domain_z")
        print(
            f"  source={source} n_segments={len(source_train)} labels={source_train['gait_label'].value_counts().sort_index().to_dict()}",
            flush=True,
        )
        source_roles[source] = fit_predict_core_experts(source_train_z, source_test_z, cols)

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
    p_pop = normalize_rows(
        (1.0 - FINAL["beta"]) * population_roles["nonlinear_ebm"]
        + FINAL["beta"] * population_roles["linear_margin_correction"]
    )
    p_source = source_ebm(population_roles, source_roles, weights, counts, FINAL["kappa"])
    p_final = fuse(p_pop, p_source, alpha, FINAL["fusion_mode"], FINAL["temperature"])

    variants = {
        "final_4class_ordinal_threshold": pred_from_thresholds_4class(p_final),
        "final_4class_argmax": np.argmax(p_final, axis=1),
    }
    rows = []
    pred_frames = []
    for variant, pred in variants.items():
        metrics = metric_values(y_true, pred)
        seed_key = f"{train_setting}|{scope}|{variant}".encode("utf-8")
        cis = bootstrap_ci(y_true, pred, seed=20260602 + zlib.adler32(seed_key) % 100000)
        row = {
            "train_setting": train_setting,
            "test_center": scope,
            "variant": variant,
            "n_test_segments_total": int(len(y_true)),
            "benchmark_macro_f1": BENCHMARK.get(scope, np.nan),
            "alpha": float(alpha),
            "source_weights": json.dumps(weights, ensure_ascii=False, sort_keys=True),
            **metrics,
        }
        for key, (lo, hi) in cis.items():
            row[f"{key}_ci_low"] = lo
            row[f"{key}_ci_high"] = hi
        row["delta_vs_benchmark_macro_f1"] = row["macro_f1"] - row["benchmark_macro_f1"]
        rows.append(row)

        pred_df = meta.copy()
        pred_df["train_setting"] = train_setting
        pred_df["test_center"] = scope
        pred_df["variant"] = variant
        pred_df["eval_in_macro_f1"] = np.isin(y_true, METRIC_LABELS)
        pred_df["pred"] = pred
        for c in TRAIN_LABELS:
            pred_df[f"proba_{c}"] = p_final[:, c]
        pred_frames.append(pred_df)
    return rows, pred_frames


def fmt_ci(value: float, lo: float, hi: float) -> str:
    return f"{value:.3f} ({lo:.3f}-{hi:.3f})"


def make_report(results: pd.DataFrame) -> str:
    lines = [
        "# CARE-PD Benchmark-Compatible LODO Results",
        "",
        "Score-3 samples were restored from `updrs_gait_raw` and retained in training folds. Metrics were computed only on held-out samples with raw labels 0, 1, or 2, matching the CARE-PD benchmark convention.",
        "",
        "## Label audit",
        "",
    ]
    audit_cols = ["train_setting", "test_center", "variant", "n_test_segments_total", "n_eval_segments", "n_excluded_score3_segments"]
    lines.append(results[audit_cols].to_markdown(index=False))
    lines.extend(["", "## Macro-F1 by held-out center", ""])
    for setting in ["CAREPD-only", "five-cohort"]:
        lines.extend([f"### {setting}", ""])
        sub = results[results["train_setting"] == setting].copy()
        table = sub[
            [
                "test_center",
                "variant",
                "macro_f1",
                "macro_f1_ci_low",
                "macro_f1_ci_high",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "acceptable_accuracy",
                "predicted_score3_rate_on_eval",
                "benchmark_macro_f1",
                "delta_vs_benchmark_macro_f1",
            ]
        ].copy()
        table["Macro-F1 (95% CI)"] = table.apply(lambda r: fmt_ci(r.macro_f1, r.macro_f1_ci_low, r.macro_f1_ci_high), axis=1)
        table = table.rename(
            columns={
                "test_center": "Held-out center",
                "variant": "Variant",
                "accuracy": "Accuracy",
                "macro_precision": "Macro precision",
                "macro_recall": "Macro recall",
                "acceptable_accuracy": "Acceptable accuracy",
                "predicted_score3_rate_on_eval": "Predicted score-3 rate on 0/1/2",
                "benchmark_macro_f1": "Benchmark macro-F1",
                "delta_vs_benchmark_macro_f1": "Delta vs benchmark",
            }
        )
        lines.append(
            table[
                [
                    "Held-out center",
                    "Variant",
                    "Macro-F1 (95% CI)",
                    "Accuracy",
                    "Macro precision",
                    "Macro recall",
                    "Acceptable accuracy",
                    "Predicted score-3 rate on 0/1/2",
                    "Benchmark macro-F1",
                    "Delta vs benchmark",
                ]
            ].to_markdown(index=False)
        )
        lines.append("")
    summary = (
        results.groupby(["train_setting", "variant"])
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            min_macro_f1=("macro_f1", "min"),
            mean_delta_vs_benchmark=("delta_vs_benchmark_macro_f1", "mean"),
            min_delta_vs_benchmark=("delta_vs_benchmark_macro_f1", "min"),
            mean_accuracy=("accuracy", "mean"),
            mean_acceptable_accuracy=("acceptable_accuracy", "mean"),
        )
        .reset_index()
    )
    lines.extend(["## Mean summary", "", summary.to_markdown(index=False)])
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    full, cols = load_v2_features()
    full = restore_carepd_raw_labels(full)
    groups = concept_groups_v2(cols)

    label_audit = (
        full.groupby(["dataset", "gait_label"]).size().unstack(fill_value=0).reindex(columns=TRAIN_LABELS, fill_value=0)
    )
    label_audit.to_csv(OUT_DIR / "raw_label_distribution_used_for_training.csv", encoding="utf-8-sig")

    rows = []
    pred_frames = []
    for train_setting, include_main150 in [("CAREPD-only", False), ("five-cohort", True)]:
        for scope in CAREPD_SCOPES:
            fold_rows, fold_preds = run_one_fold(full, cols, groups, train_setting, scope, include_main150)
            rows.extend(fold_rows)
            pred_frames.extend(fold_preds)
            pd.DataFrame(rows).to_csv(OUT_DIR / "carepd_compatible_lodo_metrics_partial.csv", index=False, encoding="utf-8-sig")

    results = pd.DataFrame(rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    summary = (
        results.groupby(["train_setting", "variant"], sort=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            min_macro_f1=("macro_f1", "min"),
            mean_delta_vs_benchmark=("delta_vs_benchmark_macro_f1", "mean"),
            min_delta_vs_benchmark=("delta_vs_benchmark_macro_f1", "min"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_precision=("macro_precision", "mean"),
            mean_macro_recall=("macro_recall", "mean"),
            mean_acceptable_accuracy=("acceptable_accuracy", "mean"),
        )
        .reset_index()
    )
    wide = results.pivot_table(index=["train_setting", "variant"], columns="test_center", values="macro_f1", aggfunc="first").reset_index()

    results.to_csv(OUT_DIR / "carepd_compatible_lodo_metrics_by_center.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "carepd_compatible_lodo_summary.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUT_DIR / "carepd_compatible_lodo_f1_wide.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "carepd_compatible_lodo_segment_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "CAREPD_COMPATIBLE_LODO_REPORT.md").write_text(make_report(results), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR), "n_metric_rows": len(results), "n_prediction_rows": len(predictions)}, ensure_ascii=False, indent=2))
    print(results[["train_setting", "test_center", "variant", "n_eval_segments", "n_excluded_score3_segments", "macro_f1", "delta_vs_benchmark_macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
