from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "internal"))

from run_biogait_moe_component_ablation import (  # noqa: E402
    CACHE_DIR,
    population_branch_custom,
    source_branch_custom,
    load_roles,
)
from run_paper_lodo_v2_branch_hier_moe import (  # noqa: E402
    CAREPD_SCOPES,
    FINAL_LOCAL_GATE_CFG,
    FINAL_PL_GATE_CFG,
    FINAL_SOURCE_CFG,
    META_COLS,
    concept_groups_v2,
    load_v2_features,
)
from run_carepd_hierarchical_local_gate_role_moe import weighted_probability_sum  # noqa: E402
from run_carepd_hierarchical_source_role_moe import (  # noqa: E402
    similarity_reliability_weights,
    source_subject_counts,
)
from run_carepd_similarity_scope_role_moe import source_similarity_ranking  # noqa: E402


OUT_DIR = ROOT / "paper" / "res" / "elegant_moe_optimization"
PRED_DIR = OUT_DIR / "predictions"
BENCHMARK = {"3DGait": 0.41, "BMCLab": 0.50, "PD-GaM": 0.51, "T-SDU-PD": 0.49}


def metric_row(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = np.argmax(proba, axis=1)
    return {
        "segment_macro_f1": float(f1_score(y_true, pred, average="macro")),
        "segment_accuracy": float(accuracy_score(y_true, pred)),
        "segment_acceptable_accuracy": float(np.mean(np.abs(y_true - pred) <= 1)),
    }


def save_prediction(path: Path, meta: pd.DataFrame, proba: np.ndarray) -> None:
    out = meta.copy()
    out["pred"] = np.argmax(proba, axis=1)
    for i in range(proba.shape[1]):
        out[f"proba_{i}"] = proba[:, i]
    out.to_csv(path, index=False, encoding="utf-8-sig")


def load_cached_fold_roles(train_setting: str, scope: str, ranking: list[tuple[str, float]]):
    fold_cache = CACHE_DIR / train_setting / scope
    population_roles = load_roles(fold_cache / "population_roles.npz")
    if population_roles is None:
        raise FileNotFoundError(f"Missing population role cache: {fold_cache}")
    source_roles = {}
    for source, _ in ranking:
        roles = load_roles(fold_cache / f"source_{source.replace('-', '_')}_roles.npz")
        if roles is None:
            raise FileNotFoundError(f"Missing source role cache: {fold_cache / source}")
        source_roles[source] = roles
    return population_roles, source_roles


def source_weighted_single_role(
    population_roles: dict[str, np.ndarray],
    source_roles: dict[str, dict[str, np.ndarray]],
    source_weights: dict[str, float],
    counts: dict[str, int],
    role: str,
    kappa: float,
) -> np.ndarray:
    items = {}
    for source, roles in source_roles.items():
        lam = counts[source] / (counts[source] + kappa)
        items[source] = lam * roles[role] + (1.0 - lam) * population_roles[role]
    return weighted_probability_sum(items, source_weights)


def source_weighted_uniform_roles(
    population_roles: dict[str, np.ndarray],
    source_roles: dict[str, dict[str, np.ndarray]],
    source_weights: dict[str, float],
    counts: dict[str, int],
    roles: list[str],
    kappa: float,
) -> np.ndarray:
    role_preds = [
        source_weighted_single_role(population_roles, source_roles, source_weights, counts, role, kappa)
        for role in roles
    ]
    return sum(role_preds) / float(len(role_preds))


def evaluate_variant(
    rows: list[dict],
    train_setting: str,
    scope: str,
    variant: str,
    proba: np.ndarray,
    y_true: np.ndarray,
    meta: pd.DataFrame,
) -> None:
    row = {
        "train_setting": train_setting,
        "test_scope": scope,
        "variant": variant,
        **metric_row(y_true, proba),
        "benchmark_macro_f1": BENCHMARK.get(scope, np.nan),
    }
    row["delta_vs_benchmark"] = row["segment_macro_f1"] - row["benchmark_macro_f1"]
    rows.append(row)
    safe = f"{train_setting}__{scope}__{variant}".replace("/", "_")
    save_prediction(PRED_DIR / f"{safe}.csv", meta, proba)


def run_one_fold(
    full: pd.DataFrame,
    groups: dict[str, list[str]],
    train_setting: str,
    scope: str,
    include_main150: bool,
) -> list[dict]:
    raw_train = full[full["dataset"] != scope].copy()
    if not include_main150:
        raw_train = raw_train[raw_train["dataset"] != "main150"].copy()
    raw_test = full[full["dataset"] == scope].copy()
    meta = raw_test[META_COLS].copy()
    y_true = meta["gait_label"].to_numpy(int)
    counts = source_subject_counts(raw_train)
    ranking = source_similarity_ranking(raw_train, raw_test, groups)
    source_weights = similarity_reliability_weights(ranking, counts, **FINAL_SOURCE_CFG)
    population_roles, source_roles = load_cached_fold_roles(train_setting, scope, ranking)

    rows: list[dict] = []

    # Reference variants.
    p_pl_full, _ = population_branch_custom(
        raw_train, population_roles, source_roles, source_weights, counts, FINAL_SOURCE_CFG, FINAL_PL_GATE_CFG
    )
    p_source_full, _ = source_branch_custom(
        raw_train, population_roles, source_roles, source_weights, counts, FINAL_SOURCE_CFG, FINAL_LOCAL_GATE_CFG
    )
    p_full = 0.375 * p_pl_full + 0.625 * p_source_full
    evaluate_variant(rows, train_setting, scope, "reference_full_current", p_full, y_true, meta)

    # Elegant population route: only EBM + linear margin roles.
    p_pop_ebm_linear, _ = population_branch_custom(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        FINAL_PL_GATE_CFG,
        role_subset=["nonlinear_ebm", "linear_margin_correction"],
    )
    p_pop_ebm_only, _ = population_branch_custom(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        FINAL_PL_GATE_CFG,
        role_subset=["nonlinear_ebm"],
    )

    # Elegant source route: no role gate, source-weighted shrunk EBM or EBM+linear.
    p_source_ebm = source_weighted_single_role(
        population_roles,
        source_roles,
        source_weights,
        counts,
        "nonlinear_ebm",
        FINAL_SOURCE_CFG["kappa"],
    )
    p_source_ebm_linear_uniform = source_weighted_uniform_roles(
        population_roles,
        source_roles,
        source_weights,
        counts,
        ["nonlinear_ebm", "linear_margin_correction"],
        FINAL_SOURCE_CFG["kappa"],
    )

    evaluate_variant(rows, train_setting, scope, "simple_population_ebm_linear_only", p_pop_ebm_linear, y_true, meta)
    evaluate_variant(rows, train_setting, scope, "simple_source_weighted_ebm_only", p_source_ebm, y_true, meta)
    evaluate_variant(rows, train_setting, scope, "simple_source_weighted_ebm_linear_uniform", p_source_ebm_linear_uniform, y_true, meta)

    for alpha in [0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]:
        p = (1.0 - alpha) * p_pop_ebm_linear + alpha * p_source_ebm
        evaluate_variant(rows, train_setting, scope, f"elegant_ebm_linear_pop__source_ebm_alpha{alpha:g}", p, y_true, meta)
        p2 = (1.0 - alpha) * p_pop_ebm_only + alpha * p_source_ebm
        evaluate_variant(rows, train_setting, scope, f"elegant_ebm_pop__source_ebm_alpha{alpha:g}", p2, y_true, meta)
        p3 = (1.0 - alpha) * p_pop_ebm_linear + alpha * p_source_ebm_linear_uniform
        evaluate_variant(rows, train_setting, scope, f"elegant_ebm_linear_both_routes_alpha{alpha:g}", p3, y_true, meta)

    return rows


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        results.groupby(["train_setting", "variant"], sort=False)
        .agg(
            mean_segment_macro_f1=("segment_macro_f1", "mean"),
            min_segment_macro_f1=("segment_macro_f1", "min"),
            mean_segment_accuracy=("segment_accuracy", "mean"),
            mean_segment_acceptable_accuracy=("segment_acceptable_accuracy", "mean"),
            mean_delta_vs_benchmark=("delta_vs_benchmark", "mean"),
            min_delta_vs_benchmark=("delta_vs_benchmark", "min"),
        )
        .reset_index()
        .sort_values(["train_setting", "mean_segment_macro_f1"], ascending=[True, False])
    )
    wide = results.pivot_table(
        index=["train_setting", "variant"],
        columns="test_scope",
        values="segment_macro_f1",
        aggfunc="first",
    ).reset_index()
    centers = [c for c in ["3DGait", "BMCLab", "PD-GaM", "T-SDU-PD"] if c in wide.columns]
    wide["mean_carepd"] = wide[centers].mean(axis=1)
    return summary, wide


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    full, cols = load_v2_features()
    groups = concept_groups_v2(cols)
    del cols

    rows: list[dict] = []
    for train_setting, include_main150 in [("CAREPD-only", False), ("five-cohort", True)]:
        for scope in CAREPD_SCOPES:
            print(f"[elegant-moe] {train_setting} held-out={scope}", flush=True)
            rows.extend(run_one_fold(full, groups, train_setting, scope, include_main150))

    results = pd.DataFrame(rows)
    summary, wide = summarize(results)
    results.to_csv(OUT_DIR / "elegant_moe_optimization_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "elegant_moe_optimization_summary.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUT_DIR / "elegant_moe_optimization_f1_by_test_center.csv", index=False, encoding="utf-8-sig")
    report = [
        "# Elegant BioGait-MoE optimization",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## F1 by held-out CAREPD center",
        "",
        wide.to_markdown(index=False),
    ]
    (OUT_DIR / "ELEGANT_MOE_OPTIMIZATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR), "n_rows": len(results)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
