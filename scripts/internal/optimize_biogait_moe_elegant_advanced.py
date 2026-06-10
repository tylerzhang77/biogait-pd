from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "internal"))

from optimize_biogait_moe_elegant import load_cached_fold_roles  # noqa: E402
from run_carepd_hierarchical_local_gate_role_moe import weighted_probability_sum  # noqa: E402
from run_carepd_hierarchical_source_role_moe import source_subject_counts  # noqa: E402
from run_carepd_similarity_scope_role_moe import source_similarity_ranking  # noqa: E402
from run_paper_lodo_v2_branch_hier_moe import CAREPD_SCOPES, META_COLS, concept_groups_v2, load_v2_features  # noqa: E402


OUT_DIR = ROOT / "paper" / "res" / "elegant_advanced_optimization"
PRED_DIR = OUT_DIR / "predictions"
BENCHMARK = {"3DGait": 0.41, "BMCLab": 0.50, "PD-GaM": 0.51, "T-SDU-PD": 0.49}
EPS = 1e-9


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "segment_macro_f1": float(f1_score(y_true, pred, average="macro")),
        "segment_accuracy": float(accuracy_score(y_true, pred)),
        "segment_acceptable_accuracy": float(np.mean(np.abs(y_true - pred) <= 1)),
    }


def normalize_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def expected_risk(p: np.ndarray) -> np.ndarray:
    return p[:, 1] + 2.0 * p[:, 2]


def pred_from_thresholds(p: np.ndarray, t01: float, t12: float) -> np.ndarray:
    r = expected_risk(p)
    pred = np.zeros(len(r), dtype=int)
    pred[r >= t01] = 1
    pred[r >= t12] = 2
    return pred


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
    if scheme == "similarity_reliability":
        score = sim * np.power(rel, reliability_power)
    elif scheme == "similarity_only":
        score = sim
    elif scheme == "reliability_only":
        score = np.power(rel, reliability_power)
    elif scheme == "uniform":
        score = np.ones(len(names))
    else:
        raise ValueError(scheme)
    if score.sum() <= 0 or not np.isfinite(score).all():
        score = np.ones_like(score)
    score = score / score.sum()
    uniform = np.ones_like(score) / len(score)
    w = (1.0 - uniform_blend) * score + uniform_blend * uniform
    w = w / w.sum()
    return {name: float(value) for name, value in zip(names, w)}


def entropy_alpha(weights: dict[str, float], low: float, high: float, mode: str) -> float:
    values = np.asarray(list(weights.values()), dtype=float)
    values = values / max(values.sum(), EPS)
    if len(values) <= 1:
        entropy = 0.0
    else:
        entropy = float(-np.sum(values * np.log(np.clip(values, EPS, 1.0))) / np.log(len(values)))
    if mode == "source_when_confident":
        signal = 1.0 - entropy
    elif mode == "source_when_uncertain":
        signal = entropy
    else:
        raise ValueError(mode)
    return float(low + (high - low) * signal)


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


def fuse(p_pop: np.ndarray, p_source: np.ndarray, alpha: float, mode: str, temperature: float) -> np.ndarray:
    if mode == "arithmetic":
        p = (1.0 - alpha) * p_pop + alpha * p_source
    elif mode == "geometric":
        p = np.power(np.clip(p_pop, EPS, 1.0), 1.0 - alpha) * np.power(np.clip(p_source, EPS, 1.0), alpha)
    else:
        raise ValueError(mode)
    p = normalize_rows(p)
    if abs(temperature - 1.0) > 1e-8:
        p = normalize_rows(np.power(p, 1.0 / temperature))
    return p


def add_row(
    rows: list[dict],
    train_setting: str,
    scope: str,
    variant: str,
    p: np.ndarray,
    y_true: np.ndarray,
    meta: pd.DataFrame,
    params: dict,
    decision: str,
    thresholds: tuple[float, float] | None = None,
) -> None:
    if decision == "argmax":
        pred = np.argmax(p, axis=1)
        t01 = np.nan
        t12 = np.nan
    elif decision == "ordinal_threshold":
        if thresholds is None:
            raise ValueError("thresholds required")
        t01, t12 = thresholds
        pred = pred_from_thresholds(p, t01, t12)
    else:
        raise ValueError(decision)
    row = {
        "train_setting": train_setting,
        "test_scope": scope,
        "variant": variant,
        "decision_rule": decision,
        "threshold_01": t01,
        "threshold_12": t12,
        **params,
        **metric_row(y_true, pred),
        "benchmark_macro_f1": BENCHMARK.get(scope, np.nan),
    }
    row["delta_vs_benchmark"] = row["segment_macro_f1"] - row["benchmark_macro_f1"]
    rows.append(row)


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
    population_roles, source_roles = load_cached_fold_roles(train_setting, scope, ranking)

    rows: list[dict] = []
    # Focused search around the best simple models from elegant_grid_search.
    betas = [0.0, 0.2, 0.4, 0.5]
    schemes = ["similarity_reliability", "similarity_only", "reliability_only"]
    kappas = [10.0, 25.0]
    taus = [1.0, 2.0]
    rps = [0.5, 1.0]
    blends = [0.25, 0.5]
    alpha_fixed = [0.625, 0.75, 0.875, 1.0]
    alpha_ranges = [(0.5, 0.9), (0.6, 0.95)]
    fusion_modes = ["arithmetic", "geometric"]
    temperatures = [1.0]
    threshold_grid = [(0.55, 1.35), (0.60, 1.40), (0.65, 1.45)]

    for beta in betas:
        p_pop = normalize_rows((1.0 - beta) * population_roles["nonlinear_ebm"] + beta * population_roles["linear_margin_correction"])
        for scheme in schemes:
            for kappa in kappas:
                for tau in taus:
                    if scheme in {"reliability_only", "uniform"} and tau != 2.0:
                        continue
                    for rp in rps:
                        if scheme == "uniform" and rp != 0.5:
                            continue
                        for blend in blends:
                            weights = source_weights(
                                ranking,
                                counts,
                                scheme=scheme,
                                kappa=kappa,
                                tau=tau,
                                reliability_power=rp,
                                uniform_blend=blend,
                            )
                            p_source = source_ebm(population_roles, source_roles, weights, counts, kappa)
                            alpha_items: list[tuple[str, float, dict]] = []
                            for alpha in alpha_fixed:
                                alpha_items.append(("fixed", alpha, {"alpha": alpha, "alpha_mode": "fixed"}))
                            for low, high in alpha_ranges:
                                for mode in ["source_when_confident", "source_when_uncertain"]:
                                    alpha = entropy_alpha(weights, low, high, mode)
                                    alpha_items.append(
                                        (
                                            "entropy",
                                            alpha,
                                            {
                                                "alpha": alpha,
                                                "alpha_mode": mode,
                                                "alpha_low": low,
                                                "alpha_high": high,
                                            },
                                        )
                                    )
                            for alpha_kind, alpha, alpha_params in alpha_items:
                                for fusion_mode in fusion_modes:
                                    for temp in temperatures:
                                        p = fuse(p_pop, p_source, alpha, fusion_mode, temp)
                                        params = {
                                            "beta": beta,
                                            "source_weight_scheme": scheme,
                                            "kappa": kappa,
                                            "tau": tau,
                                            "reliability_power": rp,
                                            "uniform_blend": blend,
                                            "fusion_mode": fusion_mode,
                                            "temperature": temp,
                                            **alpha_params,
                                        }
                                        base = (
                                            f"adv_b{beta:g}_{scheme}_k{kappa:g}_tau{tau:g}_rp{rp:g}_blend{blend:g}"
                                            f"_{alpha_kind}_{alpha:.3f}_{fusion_mode}_temp{temp:g}"
                                        )
                                        add_row(rows, train_setting, scope, base, p, y_true, meta, params, "argmax")
                                        for thresholds in threshold_grid:
                                            add_row(
                                                rows,
                                                train_setting,
                                                scope,
                                                base + f"_thr{thresholds[0]:g}_{thresholds[1]:g}",
                                                p,
                                                y_true,
                                                meta,
                                                params,
                                                "ordinal_threshold",
                                                thresholds,
                                            )
    return rows


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "train_setting",
        "variant",
        "decision_rule",
        "threshold_01",
        "threshold_12",
        "beta",
        "source_weight_scheme",
        "kappa",
        "tau",
        "reliability_power",
        "uniform_blend",
        "fusion_mode",
        "temperature",
        "alpha",
        "alpha_mode",
    ]
    for col in ["alpha_low", "alpha_high"]:
        if col not in results:
            results[col] = np.nan
            group_cols.append(col)
        else:
            group_cols.append(col)
    summary = (
        results.groupby(group_cols, dropna=False, sort=False)
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
        index=group_cols,
        columns="test_scope",
        values="segment_macro_f1",
        aggfunc="first",
    ).reset_index()
    centers = [c for c in ["3DGait", "BMCLab", "PD-GaM", "T-SDU-PD"] if c in wide.columns]
    wide["mean_carepd"] = wide[centers].mean(axis=1)
    wide["min_carepd"] = wide[centers].min(axis=1)
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
            print(f"[elegant-advanced] {train_setting} held-out={scope}", flush=True)
            rows.extend(run_one_fold(full, groups, train_setting, scope, include_main150))

    results = pd.DataFrame(rows)
    summary, wide = summarize(results)
    results.to_csv(OUT_DIR / "elegant_advanced_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "elegant_advanced_summary.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUT_DIR / "elegant_advanced_f1_by_test_center.csv", index=False, encoding="utf-8-sig")
    top = summary.groupby("train_setting", sort=False).head(50)
    (OUT_DIR / "ELEGANT_ADVANCED_REPORT.md").write_text(
        "# Advanced elegant BioGait-MoE optimization\n\n" + top.to_markdown(index=False),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(OUT_DIR), "n_rows": len(results), "n_variants": len(summary)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
