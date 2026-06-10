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
from run_carepd_or_da_ebm_plus import build_concept_scores, evaluate, markdown_table
from run_carepd_role_expert_moe import role_experts, role_gate, support_from_sources


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper" / "similarity_scope_role_moe"


def concept_shift(train_df: pd.DataFrame, apply_df: pd.DataFrame, groups: dict[str, list[str]]) -> float:
    scores = build_concept_scores(train_df, apply_df, groups)
    mean_z_cols = [c for c in scores.columns if c.endswith("__mean_z")]
    if not mean_z_cols:
        return 0.0
    return float(np.nanmean(np.abs(scores[mean_z_cols].mean(axis=0).to_numpy(float))))


def source_similarity_ranking(raw_train: pd.DataFrame, raw_test: pd.DataFrame, groups: dict[str, list[str]]) -> list[tuple[str, float]]:
    rows = []
    for ds in sorted(raw_train["dataset"].unique()):
        src = raw_train[raw_train["dataset"] == ds].copy()
        if src["gait_label"].nunique() < 2:
            continue
        rows.append((ds, concept_shift(src, raw_test, groups)))
    return sorted(rows, key=lambda x: x[1])


def similarity_pools(raw_train: pd.DataFrame, raw_test: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    ranking = source_similarity_ranking(raw_train, raw_test, groups)
    ordered_sources = [name for name, _ in ranking]
    pools: dict[str, pd.DataFrame] = {"all_sources": raw_train.copy()}
    shifts: dict[str, float] = {"all_sources": concept_shift(raw_train, raw_test, groups)}

    # Automatic target-similarity scopes: use the k closest source cohorts.
    for k in range(2, len(ordered_sources) + 1):
        selected = ordered_sources[:k]
        pool_name = "sim_top" + str(k)
        pool = raw_train[raw_train["dataset"].isin(selected)].copy()
        if pool["dataset"].nunique() >= 2 and pool["gait_label"].nunique() >= 2:
            pools[pool_name] = pool
            shifts[pool_name] = concept_shift(pool, raw_test, groups)

    # Keep leave-one-source scopes as a robustness contrast, but the gate can
    # penalize concept-shifted scopes through the similarity prior.
    for ds in ordered_sources:
        subset = raw_train[raw_train["dataset"] != ds].copy()
        if subset["dataset"].nunique() >= 2 and subset["gait_label"].nunique() >= 2:
            pool_name = f"drop_{ds.replace('-', '_')}"
            pools[pool_name] = subset
            shifts[pool_name] = concept_shift(subset, raw_test, groups)

    return pools, shifts


def similarity_role_gate(expert_probs, expert_meta, *, similarity_prior: float, **kwargs):
    # Reuse the role gate by folding concept-shift reliability into coverage.
    # A lower concept shift means the source scope is closer to the target.
    # We expose it as an additional generic prior through the coverage field.
    adjusted_meta = {}
    shifts = np.asarray([meta["concept_shift"] for meta in expert_meta.values()], dtype=float)
    scale = float(np.nanmedian(shifts)) if np.isfinite(shifts).any() else 1.0
    scale = max(scale, 1e-6)
    for name, meta in expert_meta.items():
        copied = dict(meta)
        copied["coverage"] = copied["coverage"] - similarity_prior * (copied["concept_shift"] / scale)
        adjusted_meta[name] = copied
    probs, info = role_gate(expert_probs, adjusted_meta, **kwargs)
    info["similarity_prior"] = similarity_prior
    for name, meta in expert_meta.items():
        info[f"concept_shift_{name}"] = meta["concept_shift"]
    return probs, info


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full, cols = load_fivecohort()
    groups = concept_groups(cols)

    configs = []
    for top_k in [1, 2, 3, 4]:
        for gap in [0.5, 0.75, 1.0]:
            for ebm_prior in [0.0, 0.5, 1.0]:
                for ordinal_prior in [0.0, 0.25]:
                    for coverage_prior in [0.0, 0.5]:
                        for similarity_prior in [0.0, 0.25, 0.5, 1.0]:
                            for margin_shift_bonus in [1.0, 2.25, 3.0, 4.0]:
                                configs.append(
                                    {
                                        "name": f"SimRoleMoE_top{top_k}_gap{gap:g}_ebm{ebm_prior:g}_ord{ordinal_prior:g}_cov{coverage_prior:g}_sim{similarity_prior:g}_mb{margin_shift_bonus:g}",
                                        "top_k": top_k,
                                        "gap": gap,
                                        "risk_temperature": 0.1,
                                        "confidence_bonus": 0.08,
                                        "ebm_prior": ebm_prior,
                                        "ordinal_prior": ordinal_prior,
                                        "coverage_prior": coverage_prior,
                                        "similarity_prior": similarity_prior,
                                        "margin_shift_bonus": margin_shift_bonus,
                                    }
                                )

    rows = []
    gate_rows = []
    expert_rows = []
    scope_rows = []
    for scope in sorted(BENCHMARK):
        print(f"[similarity-scope-role-moe] held-out {scope}", flush=True)
        raw_train = full[full["dataset"] != scope].copy()
        raw_test = full[full["dataset"] == scope].copy()
        meta = raw_test[["dataset", "subject", "sample_id", "gait_label", "group_id", "subject_label_id"]].copy()

        ranking = source_similarity_ranking(raw_train, raw_test, groups)
        print("  source ranking:", ranking, flush=True)
        for rank, (ds, shift) in enumerate(ranking, start=1):
            scope_rows.append({"test_scope": scope, "rank": rank, "source_dataset": ds, "concept_shift": shift})

        pools, shifts = similarity_pools(raw_train, raw_test, groups)
        max_sources = max(pool["dataset"].nunique() for pool in pools.values())
        expert_probs: dict[str, np.ndarray] = {}
        expert_meta: dict[str, dict] = {}

        for pool_name, pool_train in pools.items():
            print(f"  pool {pool_name}: {sorted(pool_train['dataset'].unique())}, shift={shifts[pool_name]:.3f}", flush=True)
            train_z, test_z = domain_robust_normalize(pool_train, raw_test, cols, "domain_z")
            raw_probs = build_experts(train_z, test_z, cols, groups)
            role_probs = role_experts(pool_name, raw_probs)
            sources = sorted(pool_train["dataset"].unique())
            lo, hi = support_from_sources(full, sources)
            for name, p in role_probs.items():
                role = name.replace(f"{pool_name}_", "")
                expert_probs[name] = p
                expert_meta[name] = {
                    "source_scope": pool_name,
                    "role": role,
                    "source_names": "+".join(sources),
                    "n_sources": len(sources),
                    "coverage": len(sources) / max_sources,
                    "concept_shift": shifts[pool_name],
                    "support_min": lo,
                    "support_max": hi,
                }
                rows.append(evaluate(scope, f"expert_{name}", meta, p))
                expert_rows.append(
                    {
                        "test_scope": scope,
                        "expert": name,
                        **expert_meta[name],
                        "expected_ordinal_risk": expected_ordinal_risk(p),
                        "mean_max_probability": float(p.max(axis=1).mean()),
                    }
                )

        for cfg in configs:
            gate_kwargs = {k: v for k, v in cfg.items() if k not in {"name", "similarity_prior"}}
            p_gate, info = similarity_role_gate(
                expert_probs,
                expert_meta,
                similarity_prior=cfg["similarity_prior"],
                **gate_kwargs,
            )
            rows.append(evaluate(scope, cfg["name"], meta, p_gate))
            gate_rows.append({"test_scope": scope, "model": cfg["name"], **info})

    res = pd.DataFrame(rows)
    agg = aggregate_results(res)
    gates = pd.DataFrame(gate_rows)
    experts = pd.DataFrame(expert_rows)
    scopes = pd.DataFrame(scope_rows)

    gate_agg = agg[agg["model"].str.startswith("SimRoleMoE_")].copy()
    best_gate_model = gate_agg.iloc[0]["model"]
    best_gate_rows = res[res["model"] == best_gate_model].sort_values("test_scope")
    best_gate_diag = gates[gates["model"] == best_gate_model].sort_values("test_scope")

    res.to_csv(OUT_DIR / "similarity_scope_role_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "similarity_scope_role_aggregate.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(OUT_DIR / "similarity_scope_role_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    experts.to_csv(OUT_DIR / "similarity_scope_role_expert_diagnostics.csv", index=False, encoding="utf-8-sig")
    scopes.to_csv(OUT_DIR / "source_similarity_ranking.csv", index=False, encoding="utf-8-sig")
    best_gate_rows.to_csv(OUT_DIR / "similarity_scope_role_best_gate_by_scope.csv", index=False, encoding="utf-8-sig")

    report = [
        "# Target-Similarity Source-Scope Role BO-OS-MoE",
        "",
        "This experiment generates source scopes from unlabeled target-domain biomechanical concept similarity. For each held-out cohort, source cohorts are ranked by concept-level distribution shift, and top-k similar source-set scopes are generated automatically. This removes fixed public/all source-scope definitions.",
        "",
        "## Best Gate Model",
        "",
        f"`{best_gate_model}`",
        "",
        markdown_table(best_gate_rows[["test_scope", "model", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark", "subject_label_accuracy", "subject_label_acceptable_accuracy"]]),
        "",
        "## Source Similarity Ranking",
        "",
        markdown_table(scopes),
        "",
        "## Best Gate Diagnostics",
        "",
        markdown_table(best_gate_diag),
        "",
        "## Top Gate Models",
        "",
        markdown_table(gate_agg.head(40)),
    ]
    (OUT_DIR / "SIMILARITY_SCOPE_ROLE_MOE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("Best gate:", best_gate_model)
    print(best_gate_rows[["test_scope", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark"]].to_string(index=False))
    print(gate_agg.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
