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
from run_carepd_role_expert_moe import role_gate
from run_carepd_similarity_scope_role_moe import source_similarity_ranking
from run_carepd_source_reweighted_role_moe import weighted_support
from run_carepd_hierarchical_source_role_moe import (
    ROLES,
    get_role_dict,
    similarity_reliability_weights,
    source_subject_counts,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper" / "hierarchical_local_gate_role_moe"


def source_ordinal_support(raw_train: pd.DataFrame, source: str, *, base_width: float, uncertainty_scale: float) -> tuple[float, float, float, float]:
    src = raw_train[raw_train["dataset"] == source].copy()
    unit = src.groupby("subject_label_id")["gait_label"].first().astype(float)
    mu = float(unit.mean())
    sd = float(unit.std(ddof=0)) if len(unit) else 0.0
    half_width = base_width + uncertainty_scale * sd / max(np.sqrt(max(len(unit), 1)), 1.0)
    return max(0.0, mu - half_width), min(2.0, mu + half_width), mu, half_width


def weighted_probability_sum(items: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out = None
    total = 0.0
    for name, p in items.items():
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        out = w * p if out is None else out + w * p
        total += w
    if out is None or total <= 0:
        raise RuntimeError("empty weighted probability sum")
    return out / total


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full, cols = load_fivecohort()
    groups = concept_groups(cols)

    source_configs = []
    for kappa in [25.0, 50.0, 100.0]:
        for tau in [0.6, 1.2]:
            for uniform_blend in [0.25]:
                for reliability_power in [0.5]:
                    source_configs.append(
                        {
                            "kappa": kappa,
                            "tau": tau,
                            "uniform_blend": uniform_blend,
                            "reliability_power": reliability_power,
                        }
                    )

    local_gate_configs = []
    for top_k in [1, 2]:
        for gap in [0.5, 0.75]:
            for ebm_prior in [0.0, 0.5]:
                for ordinal_prior in [0.0, 0.25]:
                    for margin_shift_bonus in [2.25, 4.0]:
                        for base_width in [0.05, 0.1]:
                            for uncertainty_scale in [0.5]:
                                local_gate_configs.append(
                                    {
                                        "top_k": top_k,
                                        "gap": gap,
                                        "risk_temperature": 0.1,
                                        "confidence_bonus": 0.08,
                                        "ebm_prior": ebm_prior,
                                        "ordinal_prior": ordinal_prior,
                                        "coverage_prior": 0.0,
                                        "margin_shift_bonus": margin_shift_bonus,
                                        "base_width": base_width,
                                        "uncertainty_scale": uncertainty_scale,
                                    }
                                )

    rows = []
    gate_rows = []
    source_weight_rows = []
    local_gate_rows = []
    for scope in sorted(BENCHMARK):
        print(f"[hierarchical-local-gate-role-moe] held-out {scope}", flush=True)
        raw_train = full[full["dataset"] != scope].copy()
        raw_test = full[full["dataset"] == scope].copy()
        meta = raw_test[["dataset", "subject", "sample_id", "gait_label", "group_id", "subject_label_id"]].copy()
        counts = source_subject_counts(raw_train)
        ranking = source_similarity_ranking(raw_train, raw_test, groups)
        print("  ranking:", ranking, flush=True)
        print("  subject counts:", counts, flush=True)

        train_z, test_z = domain_robust_normalize(raw_train, raw_test, cols, "domain_z")
        global_roles = get_role_dict("global_prior", build_experts(train_z, test_z, cols, groups))

        source_roles: dict[str, dict[str, np.ndarray]] = {}
        for source, _ in ranking:
            source_train = raw_train[raw_train["dataset"] == source].copy()
            print(f"  train source expert {source}: n_segments={len(source_train)}, n_subjects={counts[source]}", flush=True)
            source_train_z, source_test_z = domain_robust_normalize(source_train, raw_test, cols, "domain_z")
            safe = source.replace("-", "_")
            source_roles[source] = get_role_dict(f"{safe}_source", build_experts(source_train_z, source_test_z, cols, groups))

        for source_cfg in source_configs:
            source_weights = similarity_reliability_weights(ranking, counts, **source_cfg)
            source_name = (
                f"HLG_k{source_cfg['kappa']:g}_tau{source_cfg['tau']:g}"
                f"_blend{source_cfg['uniform_blend']:g}_rp{source_cfg['reliability_power']:g}"
            )
            shrinkage = {source: counts[source] / (counts[source] + source_cfg["kappa"]) for source in source_weights}
            for source, weight in source_weights.items():
                source_weight_rows.append(
                    {
                        "test_scope": scope,
                        "source_config": source_name,
                        "source_dataset": source,
                        "source_weight": weight,
                        "shrinkage_to_source": shrinkage[source],
                        "shrinkage_to_global": 1.0 - shrinkage[source],
                        "n_subjects": counts[source],
                        "concept_shift": dict(ranking)[source],
                    }
                )

            for gate_cfg in local_gate_configs:
                base_width = gate_cfg["base_width"]
                uncertainty_scale = gate_cfg["uncertainty_scale"]
                gate_name = (
                    f"top{gate_cfg['top_k']}_gap{gate_cfg['gap']:g}_ebm{gate_cfg['ebm_prior']:g}"
                    f"_ord{gate_cfg['ordinal_prior']:g}_mb{gate_cfg['margin_shift_bonus']:g}"
                    f"_bw{base_width:g}_us{uncertainty_scale:g}"
                )
                source_predictions: dict[str, np.ndarray] = {}
                selected_summary = []
                gate_kwargs = {k: v for k, v in gate_cfg.items() if k not in {"base_width", "uncertainty_scale"}}

                for source in source_weights:
                    safe = source.replace("-", "_")
                    r = shrinkage[source]
                    role_probs: dict[str, np.ndarray] = {}
                    expert_meta: dict[str, dict] = {}
                    lo, hi, support_mean, support_half_width = source_ordinal_support(
                        raw_train,
                        source,
                        base_width=base_width,
                        uncertainty_scale=uncertainty_scale,
                    )
                    for role in ROLES:
                        name = f"{safe}_source_{role}"
                        role_probs[name] = r * source_roles[source][role] + (1.0 - r) * global_roles[role]
                        expert_meta[name] = {
                            "source_scope": f"{safe}_source",
                            "role": role,
                            "source_names": source,
                            "n_sources": 1,
                            "coverage": 1.0,
                            "support_min": lo,
                            "support_max": hi,
                            "support_mean": support_mean,
                            "support_half_width": support_half_width,
                        }
                    p_source, info = role_gate(role_probs, expert_meta, **gate_kwargs)
                    source_predictions[source] = p_source
                    selected_summary.append(f"{source}:{info['selected_expert']}")
                    local_gate_rows.append(
                        {
                            "test_scope": scope,
                            "model": f"HLGRoleMoE_{source_name}_{gate_name}",
                            "source_dataset": source,
                            "source_weight": source_weights[source],
                            "shrinkage_to_source": shrinkage[source],
                            "selected_expert": info["selected_expert"],
                            "support_min": lo,
                            "support_max": hi,
                            "support_mean": support_mean,
                            "support_half_width": support_half_width,
                            **{k: v for k, v in info.items() if k.endswith("_ebm_excess") or k.endswith("_margin_shift_activation")},
                        }
                    )

                p_final = weighted_probability_sum(source_predictions, source_weights)
                model = f"HLGRoleMoE_{source_name}_{gate_name}"
                rows.append(evaluate(scope, model, meta, p_final))
                gate_rows.append(
                    {
                        "test_scope": scope,
                        "model": model,
                        "source_config": source_name,
                        "local_gate_config": gate_name,
                        "selected_expert": " | ".join(selected_summary),
                        "source_weight_entropy": float(-sum(w * np.log(max(w, 1e-12)) for w in source_weights.values())),
                    }
                )

    res = pd.DataFrame(rows)
    agg = aggregate_results(res)
    gates = pd.DataFrame(gate_rows)
    source_weights_df = pd.DataFrame(source_weight_rows)
    local_gates = pd.DataFrame(local_gate_rows)
    gate_agg = agg[agg["model"].str.startswith("HLGRoleMoE_")].copy()
    best_gate_model = gate_agg.iloc[0]["model"]
    best_gate_rows = res[res["model"] == best_gate_model].sort_values("test_scope")
    best_gate_diag = gates[gates["model"] == best_gate_model].sort_values("test_scope")

    res.to_csv(OUT_DIR / "hierarchical_local_gate_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "hierarchical_local_gate_aggregate.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(OUT_DIR / "hierarchical_local_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    local_gates.to_csv(OUT_DIR / "hierarchical_local_source_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    source_weights_df.to_csv(OUT_DIR / "hierarchical_local_source_weights.csv", index=False, encoding="utf-8-sig")
    best_gate_rows.to_csv(OUT_DIR / "hierarchical_local_gate_best_by_scope.csv", index=False, encoding="utf-8-sig")

    report = [
        "# Hierarchical Local-Gate Role BO-OS-MoE",
        "",
        "This variant keeps the data-size-aware empirical-Bayes shrinkage, but applies role gating inside each source before source-level mixture. This prevents source-specific nonlinear failures from being averaged away before margin correction can activate.",
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
        "## Source Weights",
        "",
        markdown_table(source_weights_df[source_weights_df["source_config"].isin(best_gate_diag["source_config"].unique())]),
        "",
        "## Top Gate Models",
        "",
        markdown_table(gate_agg.head(40)),
    ]
    (OUT_DIR / "HIERARCHICAL_LOCAL_GATE_ROLE_MOE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("Best gate:", best_gate_model)
    print(best_gate_rows[["test_scope", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark"]].to_string(index=False))
    print(gate_agg.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
