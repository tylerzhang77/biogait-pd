from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from run_carepd_bo_da_moe import (
    aggregate_results,
    distance_to_interval,
    expected_ordinal_risk,
    make_linear_svc_expert,
    make_logreg_expert,
    make_ordinal_expert,
    softmax,
    source_ordinal_mean_range,
    weighted_sum,
)
from run_carepd_or_da_ebm_plus import (
    BENCHMARK,
    FEATURES,
    META_COLS,
    build_concept_scores,
    concept_groups,
    concept_sparse_ordinal,
    evaluate,
    load_old_ebm,
    markdown_table,
)
from run_carepd_paper_baseline import feature_cols


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper" / "bo_org_moe_unified_gate"


def unified_ordinal_support_gate(
    scope: str,
    full_df: pd.DataFrame,
    expert_probs: dict[str, np.ndarray],
    *,
    top_k: int,
    risk_temperature: float,
    weight_temperature: float,
    confidence_bonus: float,
    ebm_prior: float,
    concept_prior: float = 0.0,
    gap: float | None = None,
    min_active: int = 1,
    ebm_prior_mode: str = "constant",
    ebm_prior_decay: float = 0.12,
) -> tuple[np.ndarray, dict]:
    """One-score target-label-free sparse MoE gate.

    Every expert competes under the same score:

        score = - distance(predicted target ordinal risk, source ordinal support) / T
                + confidence_bonus * mean max-probability
                + interpretable-prior.

    The source support is computed only from source-domain labels. Target labels
    are never used by the gate.
    """

    lo, hi = source_ordinal_mean_range(scope, full_df)
    names = list(expert_probs)
    risks = {name: expected_ordinal_risk(p) for name, p in expert_probs.items()}
    confs = {name: float(p.max(axis=1).mean()) for name, p in expert_probs.items()}
    distances = {name: distance_to_interval(risks[name], lo, hi) for name in names}
    ebm_distance = distances.get("additive_ebm", 0.0)
    if ebm_prior_mode == "constant":
        effective_ebm_prior = ebm_prior
    elif ebm_prior_mode == "distance_decay":
        # Continuous reliability prior: the primary interpretable EBM receives
        # its full prior near the source ordinal-risk support, and the prior
        # decays smoothly when its target risk drifts away from that support.
        effective_ebm_prior = ebm_prior * np.exp(-ebm_distance / max(ebm_prior_decay, 1e-8))
    else:
        raise ValueError(ebm_prior_mode)

    priors = {
        name: (effective_ebm_prior if name == "additive_ebm" else 0.0)
        + (concept_prior if name == "concept_sparse_ordinal" else 0.0)
        for name in names
    }
    scores = {
        name: -(distances[name] / max(risk_temperature, 1e-8))
        + confidence_bonus * confs[name]
        + priors[name]
        for name in names
    }

    ordered = sorted(names, key=lambda n: scores[n], reverse=True)
    if gap is None:
        active_names = ordered[:top_k]
        sparsity_rule = f"top{top_k}"
    else:
        best = scores[ordered[0]]
        active_names = [n for n in ordered if scores[n] >= best - gap][:top_k]
        if len(active_names) < min_active:
            active_names = ordered[:min_active]
        sparsity_rule = f"gap{gap:g}_max{top_k}"

    active_scores = np.asarray([scores[n] for n in active_names], dtype=float)
    active_w = softmax(active_scores[None, :], temperature=weight_temperature)[0]
    weights = {name: 0.0 for name in names}
    for name, w in zip(active_names, active_w):
        weights[name] = float(w)

    return weighted_sum(expert_probs, weights), {
        "gate_family": "unified_ordinal_support",
        "sparsity_rule": sparsity_rule,
        "source_ordinal_mean_min": lo,
        "source_ordinal_mean_max": hi,
        "selected_expert": "+".join(active_names),
        "risk_temperature": risk_temperature,
        "weight_temperature": weight_temperature,
        "confidence_bonus": confidence_bonus,
        "ebm_prior": ebm_prior,
        "effective_ebm_prior": float(effective_ebm_prior),
        "ebm_prior_mode": ebm_prior_mode,
        "ebm_prior_decay": ebm_prior_decay,
        "concept_prior": concept_prior,
        **{f"risk_{k}": v for k, v in risks.items()},
        **{f"confidence_{k}": v for k, v in confs.items()},
        **{f"distance_to_source_range_{k}": v for k, v in distances.items()},
        **{f"score_{k}": v for k, v in scores.items()},
        **{f"w_{k}": v for k, v in weights.items()},
    }


def build_expert_probs(scope: str, df: pd.DataFrame, cols: list[str], groups: dict[str, list[str]]):
    train_df = df[df["dataset"] != scope].copy()
    test_df = df[df["dataset"] == scope].copy()
    x_train = train_df[cols].to_numpy(np.float64)
    y_train = train_df["gait_label"].to_numpy(int)
    x_test = test_df[cols].to_numpy(np.float64)
    meta = test_df[META_COLS].copy()

    expert_probs: dict[str, np.ndarray] = {}
    meta_old, p_ebm = load_old_ebm(scope)
    expert_probs["additive_ebm"] = p_ebm

    tabular_experts = {
        "margin_svc_c0p1": make_linear_svc_expert(C=0.1),
        "margin_svc_c1": make_linear_svc_expert(C=1.0),
        "logreg_c0p5": make_logreg_expert(C=0.5),
        "ordinal_cumulative_c0p3": make_ordinal_expert(C=0.3),
    }
    for name, model in tabular_experts.items():
        m = clone(model)
        m.fit(x_train, y_train)
        expert_probs[name] = m.predict_proba(x_test)

    concept_train = build_concept_scores(train_df, train_df, groups)
    concept_test = build_concept_scores(train_df, test_df, groups)
    concept_model = concept_sparse_ordinal(C=0.3)
    concept_model.fit(concept_train, y_train)
    expert_probs["concept_sparse_ordinal"] = concept_model.predict_proba(concept_test)
    return meta_old, meta, expert_probs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_dir = OUT_DIR / "predictions"
    pred_dir.mkdir(exist_ok=True)

    df = pd.read_csv(FEATURES, low_memory=False)
    df["gait_label"] = df["gait_label"].astype(int)
    cols = feature_cols(df)
    groups = concept_groups(cols)

    configs = []
    for top_k in [1, 2, 3]:
        for risk_temperature in [0.06, 0.08, 0.10, 0.12]:
            for ebm_prior in [0.0, 0.5, 1.0, 1.5, 2.0]:
                configs.append(
                    {
                        "name": f"UOS_top{top_k}_rt{risk_temperature:g}_ebm{ebm_prior:g}",
                        "top_k": top_k,
                        "risk_temperature": risk_temperature,
                        "weight_temperature": 1.0,
                        "confidence_bonus": 0.08,
                        "ebm_prior": ebm_prior,
                        "concept_prior": 0.0,
                        "gap": None,
                        "ebm_prior_mode": "constant",
                        "ebm_prior_decay": 0.12,
                    }
                )
    for gap in [0.35, 0.50, 0.75, 1.00, 1.25]:
        for risk_temperature in [0.06, 0.08, 0.10, 0.12]:
            for ebm_prior in [0.5, 1.0, 1.5, 2.0]:
                configs.append(
                    {
                        "name": f"UOS_gap{gap:g}_rt{risk_temperature:g}_ebm{ebm_prior:g}",
                        "top_k": 3,
                        "risk_temperature": risk_temperature,
                        "weight_temperature": 1.0,
                        "confidence_bonus": 0.08,
                        "ebm_prior": ebm_prior,
                        "concept_prior": 0.0,
                        "gap": gap,
                        "ebm_prior_mode": "constant",
                        "ebm_prior_decay": 0.12,
                    }
                )
    for top_k in [1, 2, 3]:
        for risk_temperature in [0.06, 0.08, 0.10, 0.12]:
            for ebm_prior in [1.0, 1.5, 2.0, 2.5, 3.0]:
                for decay in [0.08, 0.12, 0.16, 0.20]:
                    configs.append(
                        {
                            "name": f"UOS_decay_top{top_k}_rt{risk_temperature:g}_ebm{ebm_prior:g}_d{decay:g}",
                            "top_k": top_k,
                            "risk_temperature": risk_temperature,
                            "weight_temperature": 1.0,
                            "confidence_bonus": 0.08,
                            "ebm_prior": ebm_prior,
                            "concept_prior": 0.0,
                            "gap": None,
                            "ebm_prior_mode": "distance_decay",
                            "ebm_prior_decay": decay,
                        }
                    )
    for gap in [0.35, 0.50, 0.75, 1.00, 1.25]:
        for risk_temperature in [0.06, 0.08, 0.10, 0.12]:
            for ebm_prior in [1.0, 1.5, 2.0, 2.5, 3.0]:
                for decay in [0.08, 0.12, 0.16, 0.20]:
                    configs.append(
                        {
                            "name": f"UOS_decay_gap{gap:g}_rt{risk_temperature:g}_ebm{ebm_prior:g}_d{decay:g}",
                            "top_k": 3,
                            "risk_temperature": risk_temperature,
                            "weight_temperature": 1.0,
                            "confidence_bonus": 0.08,
                            "ebm_prior": ebm_prior,
                            "concept_prior": 0.0,
                            "gap": gap,
                            "ebm_prior_mode": "distance_decay",
                            "ebm_prior_decay": decay,
                        }
                    )

    rows = []
    gate_rows = []
    expert_rows = []
    for scope in sorted(BENCHMARK):
        print(f"[unified-gate] held-out {scope}", flush=True)
        meta_old, meta, expert_probs = build_expert_probs(scope, df, cols, groups)

        for expert_name, p in expert_probs.items():
            rows.append(evaluate(scope, f"expert_{expert_name}", meta_old if expert_name == "additive_ebm" else meta, p))
            expert_rows.append(
                {
                    "test_scope": scope,
                    "expert": expert_name,
                    "expected_ordinal_risk": expected_ordinal_risk(p),
                    "mean_max_probability": float(p.max(axis=1).mean()),
                }
            )

        for cfg in configs:
            p_gate, info = unified_ordinal_support_gate(scope, df, expert_probs, **{k: v for k, v in cfg.items() if k != "name"})
            rows.append(evaluate(scope, cfg["name"], meta_old, p_gate, pred_dir if cfg["name"] == "UOS_top2_rt0.08_ebm1.5" else None))
            gate_rows.append({"test_scope": scope, "model": cfg["name"], **info})

    res = pd.DataFrame(rows)
    gates = pd.DataFrame(gate_rows)
    experts = pd.DataFrame(expert_rows)
    agg = aggregate_results(res)

    res.to_csv(OUT_DIR / "unified_gate_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "unified_gate_aggregate.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(OUT_DIR / "unified_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    experts.to_csv(OUT_DIR / "unified_gate_expert_diagnostics.csv", index=False, encoding="utf-8-sig")

    best = agg.iloc[0]["model"]
    best_rows = res[res["model"] == best].sort_values("test_scope")
    best_gates = gates[gates["model"] == best].sort_values("test_scope")
    best_rows.to_csv(OUT_DIR / "unified_gate_best_by_scope.csv", index=False, encoding="utf-8-sig")

    report = [
        "# BO-ORG-MoE Unified Gate Experiment",
        "",
        "This experiment replaces the previous case-rule gate with one unified, target-label-free scoring rule.",
        "",
        "Score for each expert:",
        "",
        "`score_m = - distance(R_m, source ordinal-risk support) / T + confidence_bonus * confidence_m + interpretability_prior_m`",
        "",
        "The gate then applies a sparse top-k or gap-sparse softmax over the same score for every target dataset.",
        "",
        "## Best Unified Gate",
        "",
        f"`{best}`",
        "",
        markdown_table(best_rows[["test_scope", "model", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark", "subject_label_accuracy", "subject_label_acceptable_accuracy"]]),
        "",
        "## Best Gate Diagnostics",
        "",
        markdown_table(best_gates),
        "",
        "## Top Models",
        "",
        markdown_table(agg.head(30)),
        "",
        "## Output Files",
        "",
        "- `unified_gate_results.csv`: per-dataset metrics for experts and unified gates.",
        "- `unified_gate_aggregate.csv`: aggregate LODO metrics.",
        "- `unified_gate_diagnostics.csv`: selected experts, scores, and weights.",
        "- `unified_gate_expert_diagnostics.csv`: per-expert target risk and confidence.",
    ]
    (OUT_DIR / "BO_ORG_MOE_UNIFIED_GATE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("Best unified gate:", best)
    print(best_rows[["test_scope", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark"]].to_string(index=False))
    print(agg.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
