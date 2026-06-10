from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_carepd_bo_da_moe import aggregate_results, distance_to_interval, expected_ordinal_risk, softmax, weighted_sum
from run_carepd_bo_org_moe_unified_gate import build_expert_probs as build_carepd_experts
from run_carepd_fivecohort_main150_domainnorm_moe import (
    BENCHMARK,
    CAREPD_FEATURES,
    build_experts as build_fivecohort_experts,
    concept_groups,
    domain_robust_normalize,
    feature_cols,
    load_fivecohort,
)
from run_carepd_or_da_ebm_plus import evaluate, markdown_table


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper" / "role_expert_moe"


def support_from_sources(df: pd.DataFrame, datasets: list[str]) -> tuple[float, float]:
    src = df[df["dataset"].isin(datasets)].copy()
    means = src.groupby("dataset")["gait_label"].apply(lambda s: float(np.mean(s.astype(int).to_numpy())))
    return float(means.min()), float(means.max())


def mean_probs(items: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(items[0], dtype=float)
    for p in items:
        out += p
    out /= float(len(items))
    return out


def role_experts(scope_name: str, raw_probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Collapse implementation-level models into role-level experts."""
    return {
        f"{scope_name}_nonlinear_ebm": raw_probs["additive_ebm"],
        f"{scope_name}_linear_margin_correction": mean_probs(
            [raw_probs["margin_svc_c0p1"], raw_probs["margin_svc_c1"]]
        ),
        f"{scope_name}_ordinal_concept": mean_probs(
            [raw_probs["ordinal_cumulative_c0p3"], raw_probs["concept_sparse_ordinal"]]
        ),
    }


def role_gate(
    expert_probs: dict[str, np.ndarray],
    expert_meta: dict[str, dict],
    *,
    top_k: int,
    gap: float,
    risk_temperature: float,
    confidence_bonus: float,
    ebm_prior: float,
    ordinal_prior: float,
    coverage_prior: float,
    margin_shift_bonus: float,
    margin_shift_center: float = 0.05,
    margin_shift_temperature: float = 0.05,
) -> tuple[np.ndarray, dict]:
    names = list(expert_probs)
    risks = {name: expected_ordinal_risk(p) for name, p in expert_probs.items()}
    confs = {name: float(p.max(axis=1).mean()) for name, p in expert_probs.items()}
    distances = {
        name: distance_to_interval(risks[name], expert_meta[name]["support_min"], expert_meta[name]["support_max"])
        for name in names
    }

    source_scopes = sorted({expert_meta[name]["source_scope"] for name in names})
    scope_excess = {}
    scope_activation = {}
    for source_scope in source_scopes:
        primary = f"{source_scope}_nonlinear_ebm"
        excess = risks.get(primary, 0.0) - expert_meta.get(primary, {"support_max": 0.0})["support_max"]
        scope_excess[source_scope] = excess
        scope_activation[source_scope] = float(
            1.0 / (1.0 + np.exp(-(excess - margin_shift_center) / max(margin_shift_temperature, 1e-8)))
        )

    scores = {}
    for name in names:
        role = expert_meta[name]["role"]
        prior = coverage_prior * expert_meta[name]["coverage"]
        if role == "nonlinear_ebm":
            prior += ebm_prior
        elif role == "ordinal_concept":
            prior += ordinal_prior
        elif role == "linear_margin_correction":
            prior += margin_shift_bonus * scope_activation[expert_meta[name]["source_scope"]]
        scores[name] = -(distances[name] / max(risk_temperature, 1e-8)) + confidence_bonus * confs[name] + prior

    ordered = sorted(names, key=lambda n: scores[n], reverse=True)
    best = scores[ordered[0]]
    active = [n for n in ordered if scores[n] >= best - gap][:top_k]
    active_scores = np.asarray([scores[n] for n in active], dtype=float)
    active_w = softmax(active_scores[None, :], temperature=1.0)[0]
    weights = {name: 0.0 for name in names}
    for name, w in zip(active, active_w):
        weights[name] = float(w)

    w_arr = np.asarray([weights[n] for n in names if weights[n] > 0], dtype=float)
    return weighted_sum(expert_probs, weights), {
        "gate_family": "role_expert_own_support",
        "selected_expert": "+".join(active),
        "top_k": top_k,
        "gap": gap,
        "risk_temperature": risk_temperature,
        "confidence_bonus": confidence_bonus,
        "ebm_prior": ebm_prior,
        "ordinal_prior": ordinal_prior,
        "coverage_prior": coverage_prior,
        "margin_shift_bonus": margin_shift_bonus,
        "effective_n_experts": float(1.0 / np.sum(np.square(w_arr))),
        **{f"{scope}_ebm_excess": value for scope, value in scope_excess.items()},
        **{f"{scope}_margin_shift_activation": value for scope, value in scope_activation.items()},
        **{f"risk_{k}": v for k, v in risks.items()},
        **{f"confidence_{k}": v for k, v in confs.items()},
        **{f"distance_to_own_support_{k}": v for k, v in distances.items()},
        **{f"score_{k}": v for k, v in scores.items()},
        **{f"w_{k}": v for k, v in weights.items()},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    carepd_df = pd.read_csv(CAREPD_FEATURES, low_memory=False)
    carepd_df["gait_label"] = carepd_df["gait_label"].astype(int)
    carepd_cols = feature_cols(carepd_df)
    carepd_groups = concept_groups(carepd_cols)

    five_df, common_cols = load_fivecohort()
    five_groups = concept_groups(common_cols)

    configs = []
    for top_k in [1, 2, 3]:
        for gap in [0.25, 0.5, 0.75, 1.0]:
            for ebm_prior in [0.5, 1.0, 1.5]:
                for ordinal_prior in [0.0, 0.25, 0.5]:
                    for coverage_prior in [0.0, 0.5, 1.0]:
                        for margin_shift_bonus in [1.0, 2.25, 3.0, 4.0]:
                            configs.append(
                                {
                                    "name": f"RoleMoE_top{top_k}_gap{gap:g}_ebm{ebm_prior:g}_ord{ordinal_prior:g}_cov{coverage_prior:g}_mb{margin_shift_bonus:g}",
                                    "top_k": top_k,
                                    "gap": gap,
                                    "risk_temperature": 0.1,
                                    "confidence_bonus": 0.08,
                                    "ebm_prior": ebm_prior,
                                    "ordinal_prior": ordinal_prior,
                                    "coverage_prior": coverage_prior,
                                    "margin_shift_bonus": margin_shift_bonus,
                                }
                            )

    rows = []
    gate_rows = []
    expert_rows = []
    for scope in sorted(BENCHMARK):
        print(f"[role-expert-moe] held-out {scope}", flush=True)
        meta, _, carepd_raw = build_carepd_experts(scope, carepd_df, carepd_cols, carepd_groups)
        carepd_sources = sorted([d for d in carepd_df["dataset"].unique() if d != scope])
        all_sources = sorted([d for d in five_df["dataset"].unique() if d != scope])
        max_sources = len(all_sources)

        raw_train = five_df[five_df["dataset"] != scope].copy()
        raw_test = five_df[five_df["dataset"] == scope].copy()
        train_z, test_z = domain_robust_normalize(raw_train, raw_test, common_cols, "domain_z")
        five_raw = build_fivecohort_experts(train_z, test_z, common_cols, five_groups)

        carepd_roles = role_experts("public_source", carepd_raw)
        all_roles = role_experts("all_source", five_raw)
        expert_probs = {**carepd_roles, **all_roles}

        carepd_lo, carepd_hi = support_from_sources(carepd_df, carepd_sources)
        all_lo, all_hi = support_from_sources(five_df, all_sources)
        expert_meta = {}
        for name in carepd_roles:
            role = name.replace("public_source_", "")
            expert_meta[name] = {
                "source_scope": "public_source",
                "role": role,
                "source_names": "+".join(carepd_sources),
                "n_sources": len(carepd_sources),
                "coverage": len(carepd_sources) / max_sources,
                "support_min": carepd_lo,
                "support_max": carepd_hi,
            }
        for name in all_roles:
            role = name.replace("all_source_", "")
            expert_meta[name] = {
                "source_scope": "all_source",
                "role": role,
                "source_names": "+".join(all_sources),
                "n_sources": len(all_sources),
                "coverage": len(all_sources) / max_sources,
                "support_min": all_lo,
                "support_max": all_hi,
            }

        for name, p in expert_probs.items():
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
            p_gate, info = role_gate(expert_probs, expert_meta, **{k: v for k, v in cfg.items() if k != "name"})
            rows.append(evaluate(scope, cfg["name"], meta, p_gate))
            gate_rows.append({"test_scope": scope, "model": cfg["name"], **info})

    res = pd.DataFrame(rows)
    agg = aggregate_results(res)
    gates = pd.DataFrame(gate_rows)
    experts = pd.DataFrame(expert_rows)
    best_gate_model = agg[agg["model"].str.startswith("RoleMoE_")].iloc[0]["model"]
    best_gate_rows = res[res["model"] == best_gate_model].sort_values("test_scope")
    best_gate_diag = gates[gates["model"] == best_gate_model].sort_values("test_scope")

    res.to_csv(OUT_DIR / "role_expert_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "role_expert_aggregate.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(OUT_DIR / "role_expert_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    experts.to_csv(OUT_DIR / "role_expert_diagnostics.csv", index=False, encoding="utf-8-sig")
    best_gate_rows.to_csv(OUT_DIR / "role_expert_best_gate_by_scope.csv", index=False, encoding="utf-8-sig")

    report = [
        "# Role-Expert BO-OS-MoE",
        "",
        "This experiment collapses implementation-level classifiers into three role-level experts per source scope: nonlinear interpretable EBM, robust linear margin correction, and ordinal concept expert. The purpose is to make the expert library more principled and less tied to individual classifier hyperparameters.",
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
        "## Top Models",
        "",
        markdown_table(agg.head(40)),
    ]
    (OUT_DIR / "ROLE_EXPERT_MOE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("Best gate:", best_gate_model)
    print(best_gate_rows[["test_scope", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark"]].to_string(index=False))
    print(agg[agg["model"].str.startswith("RoleMoE_")].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
