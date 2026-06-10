from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "internal"))

from optimize_biogait_moe_elegant_advanced import entropy_alpha, fuse, normalize_rows, pred_from_thresholds, source_weights  # noqa: E402
from run_carepd_bo_da_moe import make_linear_svc_expert  # noqa: E402
from run_carepd_fivecohort_main150_domainnorm_moe import domain_robust_normalize, make_ebm_expert  # noqa: E402
from run_carepd_hierarchical_local_gate_role_moe import weighted_probability_sum  # noqa: E402
from run_carepd_hierarchical_source_role_moe import source_subject_counts  # noqa: E402
from run_carepd_similarity_scope_role_moe import source_similarity_ranking  # noqa: E402
from run_paper_lodo_v2_branch_hier_moe import CAREPD_SCOPES, META_COLS, concept_groups_v2, load_v2_features  # noqa: E402


OUT_DIR = ROOT / "paper" / "res" / "inhouse_target_adaptation"
LABELS = [0, 1, 2]
SEEDS = [42, 2024, 3407, 7, 99]
ADAPT_RATIOS = [0.10, 0.20, 0.40, 0.60]
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
}


def align_proba(model, x_test: np.ndarray) -> np.ndarray:
    p = np.asarray(model.predict_proba(x_test), dtype=np.float64)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps["clf"], "classes_", None)
    if classes is None:
        if p.shape[1] == len(LABELS):
            return normalize_rows(p)
        raise RuntimeError("Cannot infer classes for probability alignment.")
    out = np.zeros((p.shape[0], len(LABELS)), dtype=np.float64)
    for j, cls in enumerate([int(c) for c in classes]):
        if cls in LABELS:
            out[:, LABELS.index(cls)] = p[:, j]
    return normalize_rows(out)


def fit_predict_core_roles(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    x_train = train[cols].to_numpy(np.float64)
    y_train = train["gait_label"].to_numpy(int)
    x_test = test[cols].to_numpy(np.float64)
    models = {
        "nonlinear_ebm": make_ebm_expert(k=150),
        "linear_c0p1": make_linear_svc_expert(C=0.1),
        "linear_c1": make_linear_svc_expert(C=1.0),
    }
    probs: dict[str, np.ndarray] = {}
    for name, model in models.items():
        m = clone(model)
        m.fit(x_train, y_train)
        probs[name] = align_proba(m, x_test)
    probs["linear_margin_correction"] = normalize_rows(0.5 * probs["linear_c0p1"] + 0.5 * probs["linear_c1"])
    return {
        "nonlinear_ebm": probs["nonlinear_ebm"],
        "linear_margin_correction": probs["linear_margin_correction"],
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
    vals = {k: [] for k in metric_values(y_true, y_pred)}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        row = metric_values(y_true[idx], y_pred[idx])
        for key, value in row.items():
            vals[key].append(value)
    return {key: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for key, v in vals.items()}


def stratified_subject_split(main: pd.DataFrame, ratio: float, seed: int) -> tuple[list[str], list[str], pd.DataFrame]:
    subject_df = (
        main.groupby("subject", as_index=False)
        .agg(gait_label=("gait_label", lambda s: int(pd.Series(s).mode().iloc[0])), n_segments=("sample_id", "count"))
        .sort_values("subject")
    )
    adapt_subjects, test_subjects = train_test_split(
        subject_df["subject"].astype(str).to_numpy(),
        train_size=ratio,
        random_state=seed,
        stratify=subject_df["gait_label"].to_numpy(int),
    )
    split_df = subject_df.copy()
    split_df["split"] = np.where(split_df["subject"].astype(str).isin(set(adapt_subjects)), "adaptation", "test")
    return list(map(str, adapt_subjects)), list(map(str, test_subjects)), split_df


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


def build_population_roles(raw_train: pd.DataFrame, raw_test: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    train_z, test_z = domain_robust_normalize(raw_train, raw_test, cols, "domain_z")
    return fit_predict_core_roles(train_z, test_z, cols)


def build_source_roles(raw_train: pd.DataFrame, raw_test: pd.DataFrame, ranking: list[tuple[str, float]], cols: list[str]) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for source, _shift in ranking:
        source_train = raw_train[raw_train["dataset"] == source].copy()
        source_train_z, source_test_z = domain_robust_normalize(source_train, raw_test, cols, "domain_z")
        out[source] = fit_predict_core_roles(source_train_z, source_test_z, cols)
    return out


def predict_variant(
    raw_train_population: pd.DataFrame,
    raw_train_sources: pd.DataFrame,
    raw_test: pd.DataFrame,
    cols: list[str],
    groups: dict[str, list[str]],
) -> tuple[np.ndarray, dict]:
    population_roles = build_population_roles(raw_train_population, raw_test, cols)
    counts = source_subject_counts(raw_train_sources)
    ranking = source_similarity_ranking(raw_train_sources, raw_test, groups)
    weights = source_weights(
        ranking,
        counts,
        scheme=FINAL["source_weight_scheme"],
        kappa=FINAL["kappa"],
        tau=FINAL["tau"],
        reliability_power=FINAL["reliability_power"],
        uniform_blend=FINAL["uniform_blend"],
    )
    source_roles = build_source_roles(raw_train_sources, raw_test, ranking, cols)
    alpha = entropy_alpha(weights, FINAL["alpha_low"], FINAL["alpha_high"], FINAL["alpha_mode"])
    p_pop = normalize_rows(
        (1.0 - FINAL["beta"]) * population_roles["nonlinear_ebm"]
        + FINAL["beta"] * population_roles["linear_margin_correction"]
    )
    p_source = source_ebm_shrunk(population_roles, source_roles, weights, counts, FINAL["kappa"])
    p_final = fuse(p_pop, p_source, alpha, FINAL["fusion_mode"], FINAL["temperature"])
    return p_final, {"source_ranking": ranking, "source_weights": weights, "alpha": alpha, "source_counts": counts}


def aggregate_subjects(segment_pred: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for subject, g in segment_pred.groupby("subject", sort=False):
        p = g[[f"proba_{c}" for c in LABELS]].to_numpy(float).mean(axis=0)
        labels = sorted(g["gait_label"].astype(int).unique().tolist())
        rows.append(
            {
                "subject": subject,
                "gait_label": int(labels[0]) if len(labels) == 1 else -1,
                "n_segments": int(len(g)),
                **{f"proba_{c}": float(p[c]) for c in LABELS},
            }
        )
    out = pd.DataFrame(rows)
    proba = out[[f"proba_{c}" for c in LABELS]].to_numpy(float)
    if pred_col == "pred_ordinal_risk":
        out[pred_col] = pred_from_thresholds(proba, FINAL["threshold_01"], FINAL["threshold_12"])
    elif pred_col == "pred_argmax":
        out[pred_col] = np.argmax(proba, axis=1)
    else:
        raise ValueError(pred_col)
    return out


def add_metric_row(
    rows: list[dict],
    *,
    ratio: float,
    seed: int,
    variant: str,
    level: str,
    decision_rule: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_adapt_subjects: int,
    n_test_subjects: int,
    n_adapt_segments: int,
    n_test_segments: int,
) -> None:
    row = {
        "adapt_ratio": ratio,
        "seed": seed,
        "variant": variant,
        "level": level,
        "decision_rule": decision_rule,
        "n_adapt_subjects": n_adapt_subjects,
        "n_test_subjects": n_test_subjects,
        "n_adapt_segments": n_adapt_segments,
        "n_test_segments": n_test_segments,
        "n_units": int(len(y_true)),
        **metric_values(y_true, y_pred),
    }
    ci = bootstrap_ci(
        y_true,
        y_pred,
        seed=20260604 + zlib.adler32(f"{ratio}|{seed}|{variant}|{level}|{decision_rule}".encode("utf-8")) % 100000,
    )
    for key, (lo, hi) in ci.items():
        row[f"{key}_ci_low"] = lo
        row[f"{key}_ci_high"] = hi
    rows.append(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full, cols = load_v2_features()
    groups = concept_groups_v2(cols)
    carepd = full[full["dataset"].isin(CAREPD_SCOPES)].copy()
    main = full[full["dataset"] == "main150"].copy()

    metric_rows: list[dict] = []
    diag_rows: list[dict] = []
    split_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []

    for ratio in ADAPT_RATIOS:
        for seed in SEEDS:
            adapt_subjects, test_subjects, split_df = stratified_subject_split(main, ratio, seed)
            split_df["adapt_ratio"] = ratio
            split_df["seed"] = seed
            split_rows.append(split_df)
            adapt = main[main["subject"].astype(str).isin(set(adapt_subjects))].copy()
            test = main[main["subject"].astype(str).isin(set(test_subjects))].copy()
            print(
                f"[target-adapt] ratio={ratio:.2f} seed={seed} "
                f"adapt_subjects={len(adapt_subjects)} test_subjects={len(test_subjects)} "
                f"adapt_segments={len(adapt)} test_segments={len(test)}",
                flush=True,
            )

            train_carepd = carepd.copy()
            train_full = pd.concat([carepd, adapt], ignore_index=True, sort=False)

            variant_specs = {
                "zero_shot_carepd_only": (train_carepd, train_carepd),
                "pooled_adaptation_only": (train_full, train_carepd),
                "target_source_only": (train_carepd, train_full),
                "full_target_adaptive": (train_full, train_full),
            }
            for variant, (pop_train, source_train) in variant_specs.items():
                p, diag = predict_variant(pop_train, source_train, test, cols, groups)
                seg = test[META_COLS].copy()
                seg["adapt_ratio"] = ratio
                seg["seed"] = seed
                seg["variant"] = variant
                for c in LABELS:
                    seg[f"proba_{c}"] = p[:, c]
                seg["pred_ordinal_risk"] = pred_from_thresholds(p, FINAL["threshold_01"], FINAL["threshold_12"])
                seg["pred_argmax"] = np.argmax(p, axis=1)
                prediction_rows.append(seg)

                diag_rows.append(
                    {
                        "adapt_ratio": ratio,
                        "seed": seed,
                        "variant": variant,
                        "source_ranking": json.dumps(diag["source_ranking"], ensure_ascii=False),
                        "source_weights": json.dumps(diag["source_weights"], ensure_ascii=False, sort_keys=True),
                        "source_counts": json.dumps(diag["source_counts"], ensure_ascii=False, sort_keys=True),
                        "alpha": diag["alpha"],
                    }
                )
                for pred_col in ["pred_ordinal_risk", "pred_argmax"]:
                    y_true_seg = seg["gait_label"].to_numpy(int)
                    y_pred_seg = seg[pred_col].to_numpy(int)
                    add_metric_row(
                        metric_rows,
                        ratio=ratio,
                        seed=seed,
                        variant=variant,
                        level="segment",
                        decision_rule=pred_col.replace("pred_", ""),
                        y_true=y_true_seg,
                        y_pred=y_pred_seg,
                        n_adapt_subjects=len(adapt_subjects),
                        n_test_subjects=len(test_subjects),
                        n_adapt_segments=len(adapt),
                        n_test_segments=len(test),
                    )
                    subj = aggregate_subjects(seg, pred_col)
                    subj = subj[subj["gait_label"] >= 0].copy()
                    add_metric_row(
                        metric_rows,
                        ratio=ratio,
                        seed=seed,
                        variant=variant,
                        level="subject",
                        decision_rule=pred_col.replace("pred_", ""),
                        y_true=subj["gait_label"].to_numpy(int),
                        y_pred=subj[pred_col].to_numpy(int),
                        n_adapt_subjects=len(adapt_subjects),
                        n_test_subjects=len(test_subjects),
                        n_adapt_segments=len(adapt),
                        n_test_segments=len(test),
                    )
            pd.DataFrame(metric_rows).to_csv(OUT_DIR / "target_adaptation_metrics_partial.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(["adapt_ratio", "variant", "level", "decision_rule"], as_index=False)
        .agg(
            n_splits=("seed", "nunique"),
            mean_macro_f1=("macro_f1", "mean"),
            sd_macro_f1=("macro_f1", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_precision=("macro_precision", "mean"),
            mean_macro_recall=("macro_recall", "mean"),
            mean_acceptable_accuracy=("acceptable_accuracy", "mean"),
        )
        .sort_values(["level", "decision_rule", "adapt_ratio", "mean_macro_f1"], ascending=[True, True, True, False])
    )
    best = (
        summary.sort_values("mean_macro_f1", ascending=False)
        .groupby(["adapt_ratio", "level", "decision_rule"], as_index=False)
        .head(1)
        .sort_values(["level", "decision_rule", "adapt_ratio"])
    )

    metrics.to_csv(OUT_DIR / "target_adaptation_metrics_by_split.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "target_adaptation_summary.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT_DIR / "target_adaptation_best_by_ratio.csv", index=False, encoding="utf-8-sig")
    pd.concat(split_rows, ignore_index=True).to_csv(OUT_DIR / "target_adaptation_subject_splits.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_rows, ignore_index=True).to_csv(OUT_DIR / "target_adaptation_segment_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diag_rows).to_csv(OUT_DIR / "target_adaptation_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    with (OUT_DIR / "target_adaptation_config.json").open("w", encoding="utf-8") as f:
        json.dump({"final": FINAL, "adapt_ratios": ADAPT_RATIOS, "seeds": SEEDS}, f, ensure_ascii=False, indent=2)

    print(json.dumps({"out_dir": str(OUT_DIR), "n_metric_rows": len(metrics)}, ensure_ascii=False, indent=2))
    print(best[["adapt_ratio", "level", "decision_rule", "variant", "mean_macro_f1", "sd_macro_f1", "mean_accuracy", "mean_acceptable_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
