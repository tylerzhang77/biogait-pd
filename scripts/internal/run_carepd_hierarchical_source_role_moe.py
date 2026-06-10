from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_carepd_bo_da_moe import aggregate_results, expected_ordinal_risk
from run_carepd_fivecohort_main150_domainnorm_moe import (
    BENCHMARK,
    build_experts,
    concept_groups,
    domain_robust_normalize,
    load_fivecohort,
)
from run_carepd_or_da_ebm_plus import evaluate, markdown_table
from run_carepd_role_expert_moe import role_experts, role_gate
from run_carepd_similarity_scope_role_moe import source_similarity_ranking
from run_carepd_source_reweighted_role_moe import weighted_support


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper" / "hierarchical_source_role_moe"
ROLES = ["nonlinear_ebm", "linear_margin_correction", "ordinal_concept"]


def source_subject_counts(raw_train: pd.DataFrame) -> dict[str, int]:
    return raw_train.groupby("dataset")["subject_label_id"].nunique().astype(int).to_dict()


def similarity_reliability_weights(
    ranking: list[tuple[str, float]],
    counts: dict[str, int],
    *,
    tau: float,
    uniform_blend: float,
    reliability_power: float,
    kappa: float,
) -> dict[str, float]:
    names = [name for name, _ in ranking]
    shifts = np.asarray([shift for _, shift in ranking], dtype=float)
    shifts = shifts - float(np.nanmin(shifts))
    sim = np.exp(-shifts / max(tau, 1e-8))
    rel = np.asarray([counts[name] / (counts[name] + kappa) for name in names], dtype=float)
    score = sim * np.power(rel, reliability_power)
    if not np.isfinite(score).all() or score.sum() <= 0:
        score = np.ones_like(score)
    score = score / score.sum()
    uniform = np.ones_like(score) / len(score)
    w = (1.0 - uniform_blend) * score + uniform_blend * uniform
    w = w / w.sum()
    return {name: float(value) for name, value in zip(names, w)}


def get_role_dict(scope_name: str, raw_probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name.replace(f"{scope_name}_", ""): p for name, p in role_experts(scope_name, raw_probs).items()}


def hierarchical_role_probs(
    source_role_probs: dict[str, dict[str, np.ndarray]],
    global_roles: dict[str, np.ndarray],
    counts: dict[str, int],
    source_weights: dict[str, float],
    *,
    kappa: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    shrinkage = {source: counts[source] / (counts[source] + kappa) for source in source_weights}
    out: dict[str, np.ndarray] = {}
    for role in ROLES:
        acc = None
        total = 0.0
        for source, source_weight in source_weights.items():
            r = shrinkage[source]
            shrunk = r * source_role_probs[source][role] + (1.0 - r) * global_roles[role]
            acc = source_weight * shrunk if acc is None else acc + source_weight * shrunk
            total += source_weight
        if acc is None or total <= 0:
            raise RuntimeError(f"missing role {role}")
        out[f"hier_source_{role}"] = acc / total
    return out, shrinkage


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full, cols = load_fivecohort()
    groups = concept_groups(cols)

    source_configs = []
    for kappa in [25.0, 50.0, 100.0]:
        for tau in [0.3, 0.6, 1.2]:
            for uniform_blend in [0.25, 0.5]:
                for reliability_power in [0.5, 1.0]:
                    source_configs.append(
                        {
                            "kappa": kappa,
                            "tau": tau,
                            "uniform_blend": uniform_blend,
                            "reliability_power": reliability_power,
                        }
                    )

    gate_configs = []
    for top_k in [2, 3]:
        for gap in [0.5, 0.75]:
            for ebm_prior in [0.5]:
                for ordinal_prior in [0.0, 0.25]:
                    for margin_shift_bonus in [1.0, 2.25]:
                        gate_configs.append(
                            {
                                "top_k": top_k,
                                "gap": gap,
                                "risk_temperature": 0.1,
                                "confidence_bonus": 0.08,
                                "ebm_prior": ebm_prior,
                                "ordinal_prior": ordinal_prior,
                                "coverage_prior": 0.0,
                                "margin_shift_bonus": margin_shift_bonus,
                            }
                        )

    rows = []
    gate_rows = []
    expert_rows = []
    source_weight_rows = []
    for scope in sorted(BENCHMARK):
        print(f"[hierarchical-source-role-moe] held-out {scope}", flush=True)
        raw_train = full[full["dataset"] != scope].copy()
        raw_test = full[full["dataset"] == scope].copy()
        meta = raw_test[["dataset", "subject", "sample_id", "gait_label", "group_id", "subject_label_id"]].copy()
        counts = source_subject_counts(raw_train)
        ranking = source_similarity_ranking(raw_train, raw_test, groups)
        print("  ranking:", ranking, flush=True)
        print("  subject counts:", counts, flush=True)

        # Global pooled prior expert.
        train_z, test_z = domain_robust_normalize(raw_train, raw_test, cols, "domain_z")
        global_raw = build_experts(train_z, test_z, cols, groups)
        global_roles = get_role_dict("global_prior", global_raw)

        source_role_probs: dict[str, dict[str, np.ndarray]] = {}
        for source, _ in ranking:
            source_train = raw_train[raw_train["dataset"] == source].copy()
            print(f"  train source expert {source}: n_segments={len(source_train)}, n_subjects={counts[source]}", flush=True)
            source_train_z, source_test_z = domain_robust_normalize(source_train, raw_test, cols, "domain_z")
            source_raw = build_experts(source_train_z, source_test_z, cols, groups)
            safe = source.replace("-", "_")
            source_role_probs[source] = get_role_dict(f"{safe}_source", source_raw)
            for role, p in source_role_probs[source].items():
                rows.append(evaluate(scope, f"expert_{safe}_{role}", meta, p))
                expert_rows.append(
                    {
                        "test_scope": scope,
                        "source_dataset": source,
                        "role": role,
                        "n_subjects": counts[source],
                        "expected_ordinal_risk": expected_ordinal_risk(p),
                        "mean_max_probability": float(p.max(axis=1).mean()),
                    }
                )

        for source_cfg in source_configs:
            source_weights = similarity_reliability_weights(ranking, counts, **source_cfg)
            source_name = (
                f"HS_k{source_cfg['kappa']:g}_tau{source_cfg['tau']:g}"
                f"_blend{source_cfg['uniform_blend']:g}_rp{source_cfg['reliability_power']:g}"
            )
            role_probs, shrinkage = hierarchical_role_probs(
                source_role_probs,
                global_roles,
                counts,
                source_weights,
                kappa=source_cfg["kappa"],
            )
            support_min, support_max, support_mean, support_sd = weighted_support(raw_train, source_weights)
            expert_meta = {}
            for name, p in role_probs.items():
                role = name.replace("hier_source_", "")
                expert_meta[name] = {
                    "source_scope": "hier_source",
                    "role": role,
                    "source_names": "+".join(source_weights.keys()),
                    "n_sources": len(source_weights),
                    "coverage": 1.0,
                    "support_min": support_min,
                    "support_max": support_max,
                    "support_mean": support_mean,
                    "support_sd": support_sd,
                }
                rows.append(evaluate(scope, f"expert_{source_name}_{name}", meta, p))

            for ds, w in source_weights.items():
                source_weight_rows.append(
                    {
                        "test_scope": scope,
                        "source_config": source_name,
                        "source_dataset": ds,
                        "source_weight": w,
                        "shrinkage_to_source": shrinkage[ds],
                        "shrinkage_to_global": 1.0 - shrinkage[ds],
                        "n_subjects": counts[ds],
                        "concept_shift": dict(ranking)[ds],
                    }
                )

            for gate_cfg in gate_configs:
                model = (
                    f"HSRoleMoE_{source_name}_top{gate_cfg['top_k']}_gap{gate_cfg['gap']:g}"
                    f"_ebm{gate_cfg['ebm_prior']:g}_ord{gate_cfg['ordinal_prior']:g}_mb{gate_cfg['margin_shift_bonus']:g}"
                )
                p_gate, info = role_gate(role_probs, expert_meta, **gate_cfg)
                rows.append(evaluate(scope, model, meta, p_gate))
                gate_rows.append({"test_scope": scope, "model": model, "source_config": source_name, **info})

    res = pd.DataFrame(rows)
    agg = aggregate_results(res)
    gates = pd.DataFrame(gate_rows)
    experts = pd.DataFrame(expert_rows)
    source_weights_df = pd.DataFrame(source_weight_rows)
    gate_agg = agg[agg["model"].str.startswith("HSRoleMoE_")].copy()
    best_gate_model = gate_agg.iloc[0]["model"]
    best_gate_rows = res[res["model"] == best_gate_model].sort_values("test_scope")
    best_gate_diag = gates[gates["model"] == best_gate_model].sort_values("test_scope")

    res.to_csv(OUT_DIR / "hierarchical_source_role_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "hierarchical_source_role_aggregate.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(OUT_DIR / "hierarchical_source_role_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    experts.to_csv(OUT_DIR / "hierarchical_source_role_expert_diagnostics.csv", index=False, encoding="utf-8-sig")
    source_weights_df.to_csv(OUT_DIR / "hierarchical_source_weights.csv", index=False, encoding="utf-8-sig")
    best_gate_rows.to_csv(OUT_DIR / "hierarchical_source_role_best_gate_by_scope.csv", index=False, encoding="utf-8-sig")

    report = [
        "# Data-Size-Aware Hierarchical Source Role BO-OS-MoE",
        "",
        "This experiment removes fixed public/all source scopes. Each source cohort trains role-level experts, and a pooled all-source expert acts as an empirical-Bayes prior. Source experts are shrunk toward the pooled prior according to independent subject count, then continuously mixed using target-domain concept similarity and data-size reliability.",
        "",
        "## Best Gate Model",
        "",
        f"`{best_gate_model}`",
        "",
        markdown_table(best_gate_rows[["test_scope", "model", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark", "subject_label_accuracy", "subject_label_acceptable_accuracy"]]),
        "",
        "## Best Gate Diagnostics",
        "",
        markdown_table(best_gate_diag),
        "",
        "## Source Weights and Shrinkage",
        "",
        markdown_table(source_weights_df[source_weights_df["source_config"].isin(best_gate_diag["source_config"].unique())]),
        "",
        "## Top Gate Models",
        "",
        markdown_table(gate_agg.head(40)),
    ]
    (OUT_DIR / "HIERARCHICAL_SOURCE_ROLE_MOE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("Best gate:", best_gate_model)
    print(best_gate_rows[["test_scope", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark"]].to_string(index=False))
    print(gate_agg.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
