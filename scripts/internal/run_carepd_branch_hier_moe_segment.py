from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_carepd_fivecohort_main150_domainnorm_moe import (
    BENCHMARK,
    build_experts,
    concept_groups,
    domain_robust_normalize,
    load_fivecohort,
)
from run_carepd_hierarchical_local_gate_role_moe import source_ordinal_support, weighted_probability_sum
from run_carepd_hierarchical_source_role_moe import (
    ROLES,
    get_role_dict,
    hierarchical_role_probs,
    similarity_reliability_weights,
    source_subject_counts,
)
from run_carepd_or_da_ebm_plus import evaluate, markdown_table
from run_carepd_role_expert_moe import role_gate, support_from_sources
from run_carepd_similarity_scope_role_moe import source_similarity_ranking
from run_carepd_source_reweighted_role_moe import weighted_support


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper" / "branch_hier_moe_segment"
CAREPD_SCOPES = sorted(BENCHMARK)


def prefixed_roles(prefix: str, roles: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for role, p in roles.items():
        clean = role
        for old_prefix in ("hier_source_", "global_prior_", "population_", "local_"):
            if clean.startswith(old_prefix):
                clean = clean[len(old_prefix) :]
        out[f"{prefix}_{clean}"] = p
    return out


def normalized_entropy(weights: dict[str, float]) -> float:
    values = np.asarray(list(weights.values()), dtype=float)
    values = values / max(values.sum(), 1e-12)
    entropy = float(-np.sum(values * np.log(np.maximum(values, 1e-12))))
    return entropy / np.log(len(values)) if len(values) > 1 else 0.0


def branch_alpha(rule: str | float, entropy: float) -> float:
    if rule == "entropy_local":
        return float(1.0 - entropy)
    if rule == "entropy_population":
        return float(entropy)
    if isinstance(rule, str) and rule.startswith("entropy_sigmoid_"):
        # Rule format: entropy_sigmoid_c0p75_t0p03
        parts = rule.split("_")
        center = float(parts[2][1:].replace("p", "."))
        temperature = float(parts[3][1:].replace("p", "."))
        return float(1.0 / (1.0 + np.exp(-(entropy - center) / temperature)))
    return float(rule)


def population_local_branch(
    raw_train: pd.DataFrame,
    population_roles: dict[str, np.ndarray],
    source_roles: dict[str, dict[str, np.ndarray]],
    source_weights: dict[str, float],
    counts: dict[str, int],
    source_cfg: dict,
    gate_cfg: dict,
) -> tuple[np.ndarray, dict]:
    local_roles, shrinkage = hierarchical_role_probs(
        source_roles,
        population_roles,
        counts,
        source_weights,
        kappa=source_cfg["kappa"],
    )
    population_lo, population_hi = support_from_sources(raw_train, sorted(raw_train["dataset"].unique()))
    local_lo, local_hi, local_mean, local_sd = weighted_support(raw_train, source_weights)
    expert_probs = {**prefixed_roles("population", population_roles), **prefixed_roles("local", local_roles)}
    expert_meta = {}
    for role in ROLES:
        expert_meta[f"population_{role}"] = {
            "source_scope": "population",
            "role": role,
            "source_names": "+".join(sorted(raw_train["dataset"].unique())),
            "n_sources": int(raw_train["dataset"].nunique()),
            "coverage": 1.0,
            "support_min": population_lo,
            "support_max": population_hi,
        }
        expert_meta[f"local_{role}"] = {
            "source_scope": "local",
            "role": role,
            "source_names": "+".join(source_weights.keys()),
            "n_sources": len(source_weights),
            "coverage": 1.0,
            "support_min": local_lo,
            "support_max": local_hi,
            "support_mean": local_mean,
            "support_sd": local_sd,
        }
    p, info = role_gate(expert_probs, expert_meta, **gate_cfg)
    info.update({f"pl_shrinkage_{source}": value for source, value in shrinkage.items()})
    return p, info


def local_source_gate_branch(
    raw_train: pd.DataFrame,
    population_roles: dict[str, np.ndarray],
    source_roles: dict[str, dict[str, np.ndarray]],
    source_weights: dict[str, float],
    counts: dict[str, int],
    source_cfg: dict,
    local_gate_cfg: dict,
) -> tuple[np.ndarray, dict]:
    source_predictions: dict[str, np.ndarray] = {}
    selected_summary = []
    shrinkage = {source: counts[source] / (counts[source] + source_cfg["kappa"]) for source in source_weights}
    gate_kwargs = {k: v for k, v in local_gate_cfg.items() if k not in {"base_width", "uncertainty_scale"}}
    for source in source_weights:
        safe = source.replace("-", "_")
        r = shrinkage[source]
        lo, hi, support_mean, support_half_width = source_ordinal_support(
            raw_train,
            source,
            base_width=local_gate_cfg["base_width"],
            uncertainty_scale=local_gate_cfg["uncertainty_scale"],
        )
        role_probs: dict[str, np.ndarray] = {}
        expert_meta: dict[str, dict] = {}
        for role in ROLES:
            name = f"{safe}_source_{role}"
            role_probs[name] = r * source_roles[source][role] + (1.0 - r) * population_roles[role]
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
    return weighted_probability_sum(source_predictions, source_weights), {
        "local_selected_expert": " | ".join(selected_summary),
        **{f"local_shrinkage_{source}": value for source, value in shrinkage.items()},
    }


def segment_aggregate_results(res: pd.DataFrame) -> pd.DataFrame:
    return (
        res.groupby(["train_setting", "model"])
        .agg(
            mean_segment_macro_f1=("segment_macro_f1", "mean"),
            min_segment_macro_f1=("segment_macro_f1", "min"),
            mean_segment_delta_vs_benchmark=("segment_delta_vs_benchmark", "mean"),
            min_segment_delta_vs_benchmark=("segment_delta_vs_benchmark", "min"),
            mean_segment_accuracy=("segment_accuracy", "mean"),
            mean_segment_acceptable_accuracy=("segment_acceptable_accuracy", "mean"),
            mean_subject_label_macro_f1=("subject_label_macro_f1", "mean"),
            pass_all_segment_benchmarks=("segment_delta_vs_benchmark", lambda x: bool((x >= 0).all())),
        )
        .reset_index()
        .sort_values(
            ["mean_segment_macro_f1", "pass_all_segment_benchmarks", "min_segment_delta_vs_benchmark"],
            ascending=[False, False, False],
        )
    )


def add_segment_benchmark_fields(row: dict) -> dict:
    row["segment_benchmark_macro_f1"] = BENCHMARK[row["test_scope"]]
    row["segment_delta_vs_benchmark"] = row["segment_macro_f1"] - row["segment_benchmark_macro_f1"]
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full, cols = load_fivecohort()
    groups = concept_groups(cols)

    source_configs = [
        {"kappa": kappa, "tau": tau, "uniform_blend": blend, "reliability_power": 0.5}
        for kappa in [25.0, 35.0, 50.0, 65.0]
        for tau in [0.45, 0.60, 0.90, 1.20]
        for blend in [0.10, 0.25, 0.40]
    ]
    pl_gate_configs = [
        {"top_k": 3, "gap": 0.25, "risk_temperature": 0.1, "confidence_bonus": 0.08, "ebm_prior": 0.25, "ordinal_prior": 0.0, "coverage_prior": 0.0, "margin_shift_bonus": 0.5},
        {"top_k": 3, "gap": 0.50, "risk_temperature": 0.1, "confidence_bonus": 0.08, "ebm_prior": 0.50, "ordinal_prior": 0.0, "coverage_prior": 0.0, "margin_shift_bonus": 1.0},
    ]
    local_gate_configs = [
        {"top_k": 1, "gap": 0.50, "risk_temperature": 0.1, "confidence_bonus": 0.08, "ebm_prior": 0.50, "ordinal_prior": 0.25, "coverage_prior": 0.0, "margin_shift_bonus": 2.25, "base_width": 0.05, "uncertainty_scale": 0.5},
        {"top_k": 1, "gap": 0.75, "risk_temperature": 0.1, "confidence_bonus": 0.08, "ebm_prior": 0.50, "ordinal_prior": 0.25, "coverage_prior": 0.0, "margin_shift_bonus": 2.25, "base_width": 0.05, "uncertainty_scale": 0.5},
        {"top_k": 2, "gap": 0.50, "risk_temperature": 0.1, "confidence_bonus": 0.08, "ebm_prior": 0.50, "ordinal_prior": 0.25, "coverage_prior": 0.0, "margin_shift_bonus": 4.00, "base_width": 0.05, "uncertainty_scale": 0.5},
    ]
    branch_alphas: list[str | float] = [
        0.0,
        0.10,
        0.25,
        0.40,
        0.50,
        0.60,
        0.75,
        0.90,
        1.0,
        "entropy_local",
        "entropy_population",
        "entropy_sigmoid_c0p70_t0p03",
        "entropy_sigmoid_c0p72_t0p03",
        "entropy_sigmoid_c0p75_t0p03",
        "entropy_sigmoid_c0p78_t0p03",
        "entropy_sigmoid_c0p80_t0p03",
        "entropy_sigmoid_c0p82_t0p04",
        "entropy_sigmoid_c0p85_t0p05",
    ]
    train_settings = [
        ("carepd_only", False),
        ("carepd_plus_main150", True),
    ]

    rows = []
    branch_rows = []
    source_weight_rows = []
    for train_setting, include_main150 in train_settings:
        print(f"[branch-hier-segment] train_setting={train_setting}", flush=True)
        for scope in CAREPD_SCOPES:
            print(f"[branch-hier-segment] held-out {scope}", flush=True)
            raw_train = full[full["dataset"] != scope].copy()
            if not include_main150:
                raw_train = raw_train[raw_train["dataset"] != "main150"].copy()
            raw_test = full[full["dataset"] == scope].copy()
            meta = raw_test[["dataset", "subject", "sample_id", "gait_label", "group_id", "subject_label_id"]].copy()
            counts = source_subject_counts(raw_train)
            ranking = source_similarity_ranking(raw_train, raw_test, groups)
            print("  ranking:", ranking, flush=True)

            train_z, test_z = domain_robust_normalize(raw_train, raw_test, cols, "domain_z")
            population_roles = get_role_dict("population", build_experts(train_z, test_z, cols, groups))
            source_roles = {}
            for source, _ in ranking:
                source_train = raw_train[raw_train["dataset"] == source].copy()
                print(f"  train source expert {source}: n_segments={len(source_train)}, n_units={counts[source]}", flush=True)
                source_train_z, source_test_z = domain_robust_normalize(source_train, raw_test, cols, "domain_z")
                source_roles[source] = get_role_dict(
                    f"{source.replace('-', '_')}_source",
                    build_experts(source_train_z, source_test_z, cols, groups),
                )

            for source_cfg in source_configs:
                source_weights = similarity_reliability_weights(ranking, counts, **source_cfg)
                source_name = (
                    f"k{source_cfg['kappa']:g}_tau{source_cfg['tau']:g}"
                    f"_blend{source_cfg['uniform_blend']:g}_rp{source_cfg['reliability_power']:g}"
                )
                h = normalized_entropy(source_weights)
                for ds, w in source_weights.items():
                    source_weight_rows.append(
                        {
                            "train_setting": train_setting,
                            "test_scope": scope,
                            "source_config": source_name,
                            "source_dataset": ds,
                            "source_weight": w,
                            "normalized_entropy": h,
                            "n_units": counts[ds],
                            "concept_shift": dict(ranking)[ds],
                        }
                    )

                pl_cache = []
                for i, pl_gate_cfg in enumerate(pl_gate_configs):
                    p_pl, info_pl = population_local_branch(
                        raw_train, population_roles, source_roles, source_weights, counts, source_cfg, pl_gate_cfg
                    )
                    pl_cache.append((i, p_pl, info_pl))
                    row = add_segment_benchmark_fields(
                        evaluate(scope, f"BranchMoE_component_{source_name}_PL{i}", meta, p_pl)
                    )
                    row["train_setting"] = train_setting
                    rows.append(row)

                local_cache = []
                for j, local_gate_cfg in enumerate(local_gate_configs):
                    p_local, info_local = local_source_gate_branch(
                        raw_train, population_roles, source_roles, source_weights, counts, source_cfg, local_gate_cfg
                    )
                    local_cache.append((j, p_local, info_local))
                    row = add_segment_benchmark_fields(
                        evaluate(scope, f"BranchMoE_component_{source_name}_HLG{j}", meta, p_local)
                    )
                    row["train_setting"] = train_setting
                    rows.append(row)

                for i, p_pl, info_pl in pl_cache:
                    for j, p_local, info_local in local_cache:
                        for alpha_rule in branch_alphas:
                            alpha = branch_alpha(alpha_rule, h)
                            p_final = (1.0 - alpha) * p_pl + alpha * p_local
                            model = f"BranchHierMoE_{source_name}_PL{i}_HLG{j}_alpha{alpha_rule}"
                            row = add_segment_benchmark_fields(evaluate(scope, model, meta, p_final))
                            row["train_setting"] = train_setting
                            rows.append(row)
                            branch_rows.append(
                                {
                                    "train_setting": train_setting,
                                    "test_scope": scope,
                                    "model": model,
                                    "source_config": source_name,
                                    "pl_gate_id": i,
                                    "local_gate_id": j,
                                    "alpha_local_branch": alpha,
                                    "alpha_rule": str(alpha_rule),
                                    "source_weight_entropy": h,
                                    "pl_selected_expert": info_pl["selected_expert"],
                                    "local_selected_expert": info_local["local_selected_expert"],
                                }
                            )

    res = pd.DataFrame(rows)
    branches = pd.DataFrame(branch_rows)
    source_weights_df = pd.DataFrame(source_weight_rows)
    agg = segment_aggregate_results(res)
    branch_agg = agg[agg["model"].str.startswith("BranchHierMoE_")].copy()

    best_rows = []
    best_diag = []
    for train_setting in [x[0] for x in train_settings]:
        best_model = branch_agg[branch_agg["train_setting"] == train_setting].iloc[0]["model"]
        best_rows.append(res[(res["train_setting"] == train_setting) & (res["model"] == best_model)].sort_values("test_scope"))
        best_diag.append(branches[(branches["train_setting"] == train_setting) & (branches["model"] == best_model)].sort_values("test_scope"))

    best_rows_df = pd.concat(best_rows, ignore_index=True)
    best_diag_df = pd.concat(best_diag, ignore_index=True)

    res.to_csv(OUT_DIR / "branch_hier_segment_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "branch_hier_segment_aggregate.csv", index=False, encoding="utf-8-sig")
    branches.to_csv(OUT_DIR / "branch_hier_segment_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    source_weights_df.to_csv(OUT_DIR / "branch_hier_segment_source_weights.csv", index=False, encoding="utf-8-sig")
    best_rows_df.to_csv(OUT_DIR / "branch_hier_segment_best_rows.csv", index=False, encoding="utf-8-sig")
    best_diag_df.to_csv(OUT_DIR / "branch_hier_segment_best_diagnostics.csv", index=False, encoding="utf-8-sig")

    report = [
        "# BranchHierMoE Segment-Level Model Selection",
        "",
        "This rerun fixes the model family to BranchHierMoE and selects the final configuration by CAREPD LODO segment-level macro-F1, matching the benchmark protocol.",
        "",
        "## Best Segment-Level Models",
        "",
        markdown_table(
            branch_agg.groupby("train_setting", as_index=False).head(1)[
                [
                    "train_setting",
                    "model",
                    "mean_segment_macro_f1",
                    "min_segment_macro_f1",
                    "mean_segment_delta_vs_benchmark",
                    "min_segment_delta_vs_benchmark",
                    "pass_all_segment_benchmarks",
                    "mean_subject_label_macro_f1",
                ]
            ]
        ),
        "",
        "## Best Rows",
        "",
        markdown_table(
            best_rows_df[
                [
                    "train_setting",
                    "test_scope",
                    "model",
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
        "## Best Diagnostics",
        "",
        markdown_table(best_diag_df),
        "",
        "## Top BranchHierMoE Models",
        "",
        markdown_table(branch_agg.groupby("train_setting", as_index=False).head(20)),
    ]
    (OUT_DIR / "BRANCH_HIER_MOE_SEGMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("Best segment-level models:")
    print(
        branch_agg.groupby("train_setting", as_index=False).head(1)[
            ["train_setting", "model", "mean_segment_macro_f1", "min_segment_delta_vs_benchmark", "pass_all_segment_benchmarks"]
        ].to_string(index=False)
    )
    print(best_rows_df[["train_setting", "test_scope", "segment_macro_f1", "segment_benchmark_macro_f1", "segment_delta_vs_benchmark"]].to_string(index=False))


if __name__ == "__main__":
    main()
