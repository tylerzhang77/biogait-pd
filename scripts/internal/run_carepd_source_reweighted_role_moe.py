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


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper" / "source_reweighted_role_moe"


def soft_source_weights(ranking: list[tuple[str, float]], tau: float, uniform_blend: float) -> dict[str, float]:
    names = [name for name, _ in ranking]
    shifts = np.asarray([shift for _, shift in ranking], dtype=float)
    shifts = shifts - float(np.nanmin(shifts))
    sim = np.exp(-shifts / max(tau, 1e-8))
    sim = sim / sim.sum()
    uniform = np.ones_like(sim) / len(sim)
    w = (1.0 - uniform_blend) * sim + uniform_blend * uniform
    w = w / w.sum()
    return {name: float(value) for name, value in zip(names, w)}


def resample_sources(raw_train: pd.DataFrame, source_weights: dict[str, float], seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    total_n = int(len(raw_train))
    pieces = []
    for ds, w in source_weights.items():
        src = raw_train[raw_train["dataset"] == ds].copy()
        if len(src) == 0:
            continue
        n = max(20, int(round(total_n * w)))
        idx = rng.choice(src.index.to_numpy(), size=n, replace=True)
        pieces.append(src.loc[idx].copy())
    out = pd.concat(pieces, ignore_index=True, sort=False)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def weighted_support(raw_train: pd.DataFrame, source_weights: dict[str, float]) -> tuple[float, float, float, float]:
    means = raw_train.groupby("dataset")["gait_label"].apply(lambda s: float(np.mean(s.astype(int).to_numpy())))
    names = [name for name in source_weights if name in means.index]
    w = np.asarray([source_weights[name] for name in names], dtype=float)
    w = w / w.sum()
    v = np.asarray([means[name] for name in names], dtype=float)
    mu = float(np.sum(w * v))
    sd = float(np.sqrt(np.sum(w * (v - mu) ** 2)))
    lo = max(0.0, mu - sd)
    hi = min(2.0, mu + sd)
    if hi - lo < 0.05:
        lo = max(0.0, mu - 0.05)
        hi = min(2.0, mu + 0.05)
    return lo, hi, mu, sd


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full, cols = load_fivecohort()
    groups = concept_groups(cols)

    train_configs = []
    for tau in [0.15, 0.3, 0.6, 1.2]:
        for uniform_blend in [0.0, 0.25, 0.5, 0.75]:
            train_configs.append({"tau": tau, "uniform_blend": uniform_blend})
    train_configs.append({"tau": 1.0, "uniform_blend": 1.0})

    gate_configs = []
    for top_k in [1, 2, 3]:
        for gap in [0.5, 0.75, 1.0]:
            for ebm_prior in [0.0, 0.5, 1.0]:
                for ordinal_prior in [0.0, 0.25, 0.5]:
                    for margin_shift_bonus in [1.0, 2.25, 3.0]:
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
        print(f"[source-reweighted-role-moe] held-out {scope}", flush=True)
        raw_train = full[full["dataset"] != scope].copy()
        raw_test = full[full["dataset"] == scope].copy()
        meta = raw_test[["dataset", "subject", "sample_id", "gait_label", "group_id", "subject_label_id"]].copy()

        ranking = source_similarity_ranking(raw_train, raw_test, groups)
        print("  ranking:", ranking, flush=True)
        for cfg in train_configs:
            tau = cfg["tau"]
            uniform_blend = cfg["uniform_blend"]
            source_weights = soft_source_weights(ranking, tau=tau, uniform_blend=uniform_blend)
            train_name = f"SR_tau{tau:g}_blend{uniform_blend:g}"
            print(f"  {train_name}: {source_weights}", flush=True)
            for ds, w in source_weights.items():
                source_weight_rows.append(
                    {
                        "test_scope": scope,
                        "train_config": train_name,
                        "source_dataset": ds,
                        "source_weight": w,
                        "concept_shift": dict(ranking)[ds],
                    }
                )

            weighted_train = resample_sources(raw_train, source_weights, seed=42)
            train_z, test_z = domain_robust_normalize(weighted_train, raw_test, cols, "domain_z")
            raw_probs = build_experts(train_z, test_z, cols, groups)
            role_probs = role_experts("adaptive_source", raw_probs)

            support_min, support_max, support_mean, support_sd = weighted_support(raw_train, source_weights)
            expert_meta = {}
            for name, p in role_probs.items():
                role = name.replace("adaptive_source_", "")
                expert_meta[name] = {
                    "source_scope": "adaptive_source",
                    "role": role,
                    "source_names": "+".join(source_weights.keys()),
                    "n_sources": len(source_weights),
                    "coverage": 1.0,
                    "support_min": support_min,
                    "support_max": support_max,
                    "support_mean": support_mean,
                    "support_sd": support_sd,
                }
                rows.append(evaluate(scope, f"expert_{train_name}_{name}", meta, p))
                expert_rows.append(
                    {
                        "test_scope": scope,
                        "train_config": train_name,
                        "expert": name,
                        **expert_meta[name],
                        "expected_ordinal_risk": expected_ordinal_risk(p),
                        "mean_max_probability": float(p.max(axis=1).mean()),
                    }
                )

            for gate_cfg in gate_configs:
                model = (
                    f"SRRoleMoE_{train_name}_top{gate_cfg['top_k']}_gap{gate_cfg['gap']:g}"
                    f"_ebm{gate_cfg['ebm_prior']:g}_ord{gate_cfg['ordinal_prior']:g}_mb{gate_cfg['margin_shift_bonus']:g}"
                )
                p_gate, info = role_gate(role_probs, expert_meta, **gate_cfg)
                rows.append(evaluate(scope, model, meta, p_gate))
                gate_rows.append({"test_scope": scope, "model": model, "train_config": train_name, **info})

    res = pd.DataFrame(rows)
    agg = aggregate_results(res)
    gates = pd.DataFrame(gate_rows)
    experts = pd.DataFrame(expert_rows)
    source_weights_df = pd.DataFrame(source_weight_rows)
    gate_agg = agg[agg["model"].str.startswith("SRRoleMoE_")].copy()
    best_gate_model = gate_agg.iloc[0]["model"]
    best_gate_rows = res[res["model"] == best_gate_model].sort_values("test_scope")
    best_gate_diag = gates[gates["model"] == best_gate_model].sort_values("test_scope")

    res.to_csv(OUT_DIR / "source_reweighted_role_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "source_reweighted_role_aggregate.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(OUT_DIR / "source_reweighted_role_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    experts.to_csv(OUT_DIR / "source_reweighted_role_expert_diagnostics.csv", index=False, encoding="utf-8-sig")
    source_weights_df.to_csv(OUT_DIR / "source_reweighting_weights.csv", index=False, encoding="utf-8-sig")
    best_gate_rows.to_csv(OUT_DIR / "source_reweighted_role_best_gate_by_scope.csv", index=False, encoding="utf-8-sig")

    report = [
        "# Target-Conditioned Source-Reweighted Role BO-OS-MoE",
        "",
        "This experiment removes discrete source scopes. Unlabeled target-domain biomechanical concept similarity is used to compute continuous weights over all source cohorts. A single adaptive-source training set is then formed by source-weighted resampling, followed by the same three role-level experts.",
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
        markdown_table(source_weights_df[source_weights_df["train_config"].isin(best_gate_diag["train_config"].unique())]),
        "",
        "## Top Gate Models",
        "",
        markdown_table(gate_agg.head(40)),
    ]
    (OUT_DIR / "SOURCE_REWEIGHTED_ROLE_MOE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("Best gate:", best_gate_model)
    print(best_gate_rows[["test_scope", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark"]].to_string(index=False))
    print(gate_agg.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
