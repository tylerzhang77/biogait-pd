from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "internal"))

from run_carepd_branch_hier_moe_segment import (  # noqa: E402
    local_source_gate_branch,
    population_local_branch,
)
from run_carepd_fivecohort_main150_domainnorm_moe import build_experts, domain_robust_normalize  # noqa: E402
from run_carepd_hierarchical_local_gate_role_moe import source_ordinal_support, weighted_probability_sum  # noqa: E402
from run_carepd_hierarchical_source_role_moe import (  # noqa: E402
    ROLES,
    get_role_dict,
    hierarchical_role_probs,
    similarity_reliability_weights,
    source_subject_counts,
)
from run_carepd_role_expert_moe import role_gate, support_from_sources  # noqa: E402
from run_carepd_source_reweighted_role_moe import weighted_support  # noqa: E402
from run_paper_lodo_v2_branch_hier_moe import (  # noqa: E402
    BENCHMARK,
    CAREPD_SCOPES,
    FINAL_ALPHA_LOCAL,
    FINAL_LOCAL_GATE_CFG,
    FINAL_PL_GATE_CFG,
    FINAL_SOURCE_CFG,
    META_COLS,
    concept_groups_v2,
    load_v2_features,
)
from run_carepd_similarity_scope_role_moe import source_similarity_ranking  # noqa: E402


OUT_DIR = ROOT / "paper" / "res" / "component_ablation"
CACHE_DIR = OUT_DIR / "expert_probability_cache"
PRED_DIR = OUT_DIR / "predictions"


def metric_row(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = np.argmax(proba, axis=1)
    return {
        "segment_macro_f1": float(f1_score(y_true, pred, average="macro")),
        "segment_accuracy": float(accuracy_score(y_true, pred)),
        "segment_acceptable_accuracy": float(np.mean(np.abs(y_true - pred) <= 1)),
    }


def save_prediction(path: Path, meta: pd.DataFrame, proba: np.ndarray) -> None:
    pred = np.argmax(proba, axis=1)
    out = meta.copy()
    out["pred"] = pred
    for i in range(proba.shape[1]):
        out[f"proba_{i}"] = proba[:, i]
    out.to_csv(path, index=False, encoding="utf-8-sig")


def save_roles(path: Path, roles: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{r: roles[r] for r in ROLES})


def load_roles(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    arr = np.load(path)
    return {r: arr[r] for r in ROLES}


def build_or_load_roles(
    cache_path: Path,
    role_prefix: str,
    train_z: pd.DataFrame,
    test_z: pd.DataFrame,
    cols: list[str],
    groups: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    cached = load_roles(cache_path)
    if cached is not None:
        return cached
    raw_probs = build_experts(train_z, test_z, cols, groups)
    roles = get_role_dict(role_prefix, raw_probs)
    save_roles(cache_path, roles)
    return roles


def source_weights_from_scores(names: list[str], scores: np.ndarray, uniform_blend: float) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or scores.sum() <= 0:
        scores = np.ones_like(scores)
    scores = scores / scores.sum()
    uniform = np.ones_like(scores) / len(scores)
    w = (1.0 - uniform_blend) * scores + uniform_blend * uniform
    w = w / w.sum()
    return {name: float(value) for name, value in zip(names, w)}


def source_weight_variants(
    ranking: list[tuple[str, float]],
    counts: dict[str, int],
    scope: str,
    train_setting: str,
) -> dict[str, dict[str, float]]:
    names = [name for name, _ in ranking]
    shifts = np.asarray([shift for _, shift in ranking], dtype=float)
    shifts = shifts - float(np.nanmin(shifts))
    rel = np.asarray([counts[name] / (counts[name] + FINAL_SOURCE_CFG["kappa"]) for name in names], dtype=float)
    sim = np.exp(-shifts / FINAL_SOURCE_CFG["tau"])
    blend = FINAL_SOURCE_CFG["uniform_blend"]
    rp = FINAL_SOURCE_CFG["reliability_power"]
    variants = {
        "full_source_similarity_reliability": similarity_reliability_weights(ranking, counts, **FINAL_SOURCE_CFG),
        "similarity_only_no_reliability": source_weights_from_scores(names, sim, blend),
        "reliability_only_no_similarity": source_weights_from_scores(names, np.power(rel, rp), blend),
        "uniform_source_weights": {name: 1.0 / len(names) for name in names},
        "nearest_source_only": {name: float(i == int(np.argmin(shifts))) for i, name in enumerate(names)},
        "no_uniform_blend": source_weights_from_scores(names, sim * np.power(rel, rp), 0.0),
    }
    seed = sum(ord(c) for c in f"{train_setting}:{scope}") + 123
    random_scores = np.random.default_rng(seed).random(len(names))
    variants["random_source_weights_negative_control"] = source_weights_from_scores(names, random_scores, 0.0)
    return variants


def population_branch_custom(
    raw_train: pd.DataFrame,
    population_roles: dict[str, np.ndarray],
    source_roles: dict[str, dict[str, np.ndarray]],
    source_weights: dict[str, float],
    counts: dict[str, int],
    source_cfg: dict,
    gate_cfg: dict,
    *,
    role_subset: list[str] | None = None,
    use_population_roles: bool = True,
    use_local_roles: bool = True,
    local_source_weights: dict[str, float] | None = None,
    shrinkage_kappa: float | None = None,
) -> tuple[np.ndarray, dict]:
    roles = role_subset or ROLES
    local_source_weights = local_source_weights or source_weights
    local_cfg = dict(source_cfg)
    if shrinkage_kappa is not None:
        local_cfg["kappa"] = shrinkage_kappa
    local_roles_all, shrinkage = hierarchical_role_probs(
        source_roles,
        population_roles,
        counts,
        local_source_weights,
        kappa=local_cfg["kappa"],
    )
    population_lo, population_hi = support_from_sources(raw_train, sorted(raw_train["dataset"].unique()))
    local_lo, local_hi, local_mean, local_sd = weighted_support(raw_train, local_source_weights)

    expert_probs: dict[str, np.ndarray] = {}
    expert_meta: dict[str, dict] = {}
    for role in roles:
        if use_population_roles:
            name = f"population_{role}"
            expert_probs[name] = population_roles[role]
            expert_meta[name] = {
                "source_scope": "population",
                "role": role,
                "source_names": "+".join(sorted(raw_train["dataset"].unique())),
                "n_sources": int(raw_train["dataset"].nunique()),
                "coverage": 1.0,
                "support_min": population_lo,
                "support_max": population_hi,
            }
        if use_local_roles:
            name = f"local_{role}"
            expert_probs[name] = local_roles_all[f"hier_source_{role}"]
            expert_meta[name] = {
                "source_scope": "local",
                "role": role,
                "source_names": "+".join(local_source_weights.keys()),
                "n_sources": len(local_source_weights),
                "coverage": 1.0,
                "support_min": local_lo,
                "support_max": local_hi,
                "support_mean": local_mean,
                "support_sd": local_sd,
            }
    p, info = role_gate(expert_probs, expert_meta, **gate_cfg)
    info.update({f"pl_shrinkage_{source}": value for source, value in shrinkage.items()})
    return p, info


def source_branch_custom(
    raw_train: pd.DataFrame,
    population_roles: dict[str, np.ndarray],
    source_roles: dict[str, dict[str, np.ndarray]],
    source_weights: dict[str, float],
    counts: dict[str, int],
    source_cfg: dict,
    local_gate_cfg: dict,
    *,
    role_subset: list[str] | None = None,
    gate_mode: str = "role_gate",
) -> tuple[np.ndarray, dict]:
    roles = role_subset or ROLES
    source_predictions: dict[str, np.ndarray] = {}
    selected_summary: list[str] = []
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
        for role in roles:
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
        if gate_mode == "role_gate":
            p_source, info = role_gate(role_probs, expert_meta, **gate_kwargs)
            selected_summary.append(f"{source}:{info['selected_expert']}")
        elif gate_mode == "uniform_roles":
            p_source = sum(role_probs.values()) / float(len(role_probs))
            selected_summary.append(f"{source}:uniform_roles")
        elif gate_mode == "single_role":
            if len(roles) != 1:
                raise ValueError("single_role mode requires exactly one role")
            p_source = next(iter(role_probs.values()))
            selected_summary.append(f"{source}:{roles[0]}")
        else:
            raise ValueError(gate_mode)
        source_predictions[source] = p_source
    return weighted_probability_sum(source_predictions, source_weights), {
        "local_selected_expert": " | ".join(selected_summary),
        **{f"local_shrinkage_{source}": value for source, value in shrinkage.items()},
    }


def full_model_with_parts(
    raw_train: pd.DataFrame,
    population_roles: dict[str, np.ndarray],
    source_roles: dict[str, dict[str, np.ndarray]],
    source_weights: dict[str, float],
    counts: dict[str, int],
    *,
    role_subset: list[str] | None = None,
    pl_gate_cfg: dict | None = None,
    local_gate_cfg: dict | None = None,
    alpha: float = FINAL_ALPHA_LOCAL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_pl, _ = population_branch_custom(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        pl_gate_cfg or FINAL_PL_GATE_CFG,
        role_subset=role_subset,
    )
    p_local, _ = source_branch_custom(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        local_gate_cfg or FINAL_LOCAL_GATE_CFG,
        role_subset=role_subset,
    )
    return (1.0 - alpha) * p_pl + alpha * p_local, p_pl, p_local


def add_variant(
    rows: list[dict],
    predictions: dict[str, np.ndarray],
    train_setting: str,
    scope: str,
    category: str,
    variant: str,
    proba: np.ndarray,
    y_true: np.ndarray,
    meta: pd.DataFrame,
) -> None:
    row = {
        "train_setting": train_setting,
        "test_scope": scope,
        "category": category,
        "variant": variant,
        **metric_row(y_true, proba),
        "benchmark_macro_f1": BENCHMARK.get(scope, np.nan),
    }
    row["delta_vs_benchmark"] = row["segment_macro_f1"] - row["benchmark_macro_f1"]
    rows.append(row)
    key = f"{train_setting}__{scope}__{category}__{variant}".replace("/", "_")
    predictions[key] = proba
    save_prediction(PRED_DIR / f"{key}.csv", meta, proba)


def run_one_fold(
    full: pd.DataFrame,
    cols: list[str],
    groups: dict[str, list[str]],
    train_setting: str,
    scope: str,
    include_main150: bool,
) -> tuple[list[dict], list[dict]]:
    raw_train = full[full["dataset"] != scope].copy()
    if not include_main150:
        raw_train = raw_train[raw_train["dataset"] != "main150"].copy()
    raw_test = full[full["dataset"] == scope].copy()
    meta = raw_test[META_COLS].copy()
    y_true = meta["gait_label"].to_numpy(int)

    counts = source_subject_counts(raw_train)
    ranking = source_similarity_ranking(raw_train, raw_test, groups)
    source_weights = similarity_reliability_weights(ranking, counts, **FINAL_SOURCE_CFG)
    print(f"[ablation] {train_setting} held-out={scope} source_weights={source_weights}", flush=True)

    train_z, test_z = domain_robust_normalize(raw_train, raw_test, cols, "domain_z")
    fold_cache = CACHE_DIR / train_setting / scope
    population_roles = build_or_load_roles(
        fold_cache / "population_roles.npz",
        "population",
        train_z,
        test_z,
        cols,
        groups,
    )
    source_roles: dict[str, dict[str, np.ndarray]] = {}
    for source, _ in ranking:
        source_train = raw_train[raw_train["dataset"] == source].copy()
        source_train_z, source_test_z = domain_robust_normalize(source_train, raw_test, cols, "domain_z")
        source_roles[source] = build_or_load_roles(
            fold_cache / f"source_{source.replace('-', '_')}_roles.npz",
            f"{source.replace('-', '_')}_source",
            source_train_z,
            source_test_z,
            cols,
            groups,
        )

    rows: list[dict] = []
    diag_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}

    # 1. Branch-level ablation.
    p_full, p_pl, p_local = full_model_with_parts(raw_train, population_roles, source_roles, source_weights, counts)
    add_variant(rows, predictions, train_setting, scope, "1_branch_level", "full_biogait_moe", p_full, y_true, meta)
    add_variant(rows, predictions, train_setting, scope, "1_branch_level", "population_branch_only", p_pl, y_true, meta)
    add_variant(rows, predictions, train_setting, scope, "1_branch_level", "source_branch_only", p_local, y_true, meta)
    add_variant(rows, predictions, train_setting, scope, "1_branch_level", "uniform_branch_blend", 0.5 * p_pl + 0.5 * p_local, y_true, meta)
    add_variant(rows, predictions, train_setting, scope, "1_branch_level", "population_ebm_only", population_roles["nonlinear_ebm"], y_true, meta)
    p_source_ebm, _ = source_branch_custom(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        FINAL_LOCAL_GATE_CFG,
        role_subset=["nonlinear_ebm"],
        gate_mode="single_role",
    )
    add_variant(rows, predictions, train_setting, scope, "1_branch_level", "source_local_ebm_only", p_source_ebm, y_true, meta)

    # 2. Population-branch internal ablation.
    add_variant(rows, predictions, train_setting, scope, "2_population_internal", "population_branch_full", p_pl, y_true, meta)
    p_pop_roles, _ = population_branch_custom(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        FINAL_PL_GATE_CFG,
        use_population_roles=True,
        use_local_roles=False,
    )
    add_variant(rows, predictions, train_setting, scope, "2_population_internal", "population_roles_only", p_pop_roles, y_true, meta)
    p_local_roles, _ = population_branch_custom(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        FINAL_PL_GATE_CFG,
        use_population_roles=False,
        use_local_roles=True,
    )
    add_variant(rows, predictions, train_setting, scope, "2_population_internal", "sourceweighted_local_roles_only", p_local_roles, y_true, meta)
    uniform_weights = {source: 1.0 / len(source_weights) for source in source_weights}
    p_uniform_local, _ = population_branch_custom(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        FINAL_PL_GATE_CFG,
        local_source_weights=uniform_weights,
    )
    add_variant(rows, predictions, train_setting, scope, "2_population_internal", "local_roles_uniform_source_weights", p_uniform_local, y_true, meta)
    p_no_shrink, _ = population_branch_custom(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        FINAL_PL_GATE_CFG,
        shrinkage_kappa=0.0,
    )
    add_variant(rows, predictions, train_setting, scope, "2_population_internal", "local_roles_no_shrinkage", p_no_shrink, y_true, meta)
    p_strong_shrink, _ = population_branch_custom(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        FINAL_PL_GATE_CFG,
        shrinkage_kappa=100.0,
    )
    add_variant(rows, predictions, train_setting, scope, "2_population_internal", "local_roles_strong_shrinkage_k100", p_strong_shrink, y_true, meta)

    # 3. Source-branch internal ablation.
    add_variant(rows, predictions, train_setting, scope, "3_source_internal", "source_branch_full", p_local, y_true, meta)
    p_no_role_gate, _ = source_branch_custom(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        FINAL_LOCAL_GATE_CFG,
        gate_mode="uniform_roles",
    )
    add_variant(rows, predictions, train_setting, scope, "3_source_internal", "no_source_role_gate_uniform_roles", p_no_role_gate, y_true, meta)
    for role in ROLES:
        p_role, _ = source_branch_custom(
            raw_train,
            population_roles,
            source_roles,
            source_weights,
            counts,
            FINAL_SOURCE_CFG,
            FINAL_LOCAL_GATE_CFG,
            role_subset=[role],
            gate_mode="single_role",
        )
        add_variant(rows, predictions, train_setting, scope, "3_source_internal", f"source_{role}_only", p_role, y_true, meta)
    top3_cfg = dict(FINAL_LOCAL_GATE_CFG)
    top3_cfg["top_k"] = 3
    top3_cfg["gap"] = 0.75
    p_top3, _ = source_branch_custom(
        raw_train,
        population_roles,
        source_roles,
        source_weights,
        counts,
        FINAL_SOURCE_CFG,
        top3_cfg,
    )
    add_variant(rows, predictions, train_setting, scope, "3_source_internal", "source_top3_soft_role_gate", p_top3, y_true, meta)

    # 4. Source-relevance gate ablation.
    for weight_variant, weights in source_weight_variants(ranking, counts, scope, train_setting).items():
        p_w, _, _ = full_model_with_parts(raw_train, population_roles, source_roles, weights, counts)
        add_variant(rows, predictions, train_setting, scope, "4_source_relevance", weight_variant, p_w, y_true, meta)
        diag_rows.append(
            {
                "train_setting": train_setting,
                "test_scope": scope,
                "category": "4_source_relevance",
                "variant": weight_variant,
                **{f"w_{k}": v for k, v in weights.items()},
            }
        )

    # 5. Expert-role ablation.
    role_sets = {
        "all_roles": ROLES,
        "remove_ebm": ["linear_margin_correction", "ordinal_concept"],
        "remove_linear_margin": ["nonlinear_ebm", "ordinal_concept"],
        "remove_ordinal_concept": ["nonlinear_ebm", "linear_margin_correction"],
        "ebm_and_linear": ["nonlinear_ebm", "linear_margin_correction"],
        "ebm_and_ordinal": ["nonlinear_ebm", "ordinal_concept"],
        "linear_and_ordinal": ["linear_margin_correction", "ordinal_concept"],
        "ebm_only": ["nonlinear_ebm"],
        "linear_margin_only": ["linear_margin_correction"],
        "ordinal_concept_only": ["ordinal_concept"],
    }
    for variant, roles in role_sets.items():
        p_roles, _, _ = full_model_with_parts(raw_train, population_roles, source_roles, source_weights, counts, role_subset=roles)
        add_variant(rows, predictions, train_setting, scope, "5_expert_roles", variant, p_roles, y_true, meta)

    return rows, diag_rows


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        results.groupby(["train_setting", "category", "variant"], sort=False)
        .agg(
            mean_segment_macro_f1=("segment_macro_f1", "mean"),
            min_segment_macro_f1=("segment_macro_f1", "min"),
            mean_segment_accuracy=("segment_accuracy", "mean"),
            mean_segment_acceptable_accuracy=("segment_acceptable_accuracy", "mean"),
            mean_delta_vs_benchmark=("delta_vs_benchmark", "mean"),
            min_delta_vs_benchmark=("delta_vs_benchmark", "min"),
        )
        .reset_index()
    )
    wide = results.pivot_table(
        index=["train_setting", "category", "variant"],
        columns="test_scope",
        values="segment_macro_f1",
        aggfunc="first",
    ).reset_index()
    centers = [c for c in ["3DGait", "BMCLab", "PD-GaM", "T-SDU-PD", "main150"] if c in wide.columns]
    wide["mean_carepd"] = wide[[c for c in centers if c != "main150"]].mean(axis=1)
    if "main150" in wide.columns:
        wide["mean_all_tested"] = wide[centers].mean(axis=1)
    return summary, wide


def compact_key_results(summary: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    categories = sorted(wide["category"].unique())
    for setting in ["CAREPD-only", "five-cohort"]:
        for category in categories:
            sub = summary[(summary["train_setting"] == setting) & (summary["category"] == category)].copy()
            sub = sub.sort_values("mean_segment_macro_f1", ascending=False)
            keep = sub.head(4)["variant"].tolist()
            for anchor in ["full_biogait_moe", "all_roles"]:
                if anchor in set(sub["variant"]) and anchor not in keep:
                    keep.append(anchor)
            chunks.append(
                wide[
                    (wide["train_setting"] == setting)
                    & (wide["category"] == category)
                    & (wide["variant"].isin(keep))
                ].copy()
            )
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    full, cols = load_v2_features()
    groups = concept_groups_v2(cols)

    settings = [
        ("CAREPD-only", False),
        ("five-cohort", True),
    ]
    rows: list[dict] = []
    diag_rows: list[dict] = []
    for train_setting, include_main150 in settings:
        for scope in CAREPD_SCOPES:
            fold_rows, fold_diag = run_one_fold(full, cols, groups, train_setting, scope, include_main150)
            rows.extend(fold_rows)
            diag_rows.extend(fold_diag)

    results = pd.DataFrame(rows)
    source_diag = pd.DataFrame(diag_rows)
    summary, wide = summarize(results)
    compact = compact_key_results(summary, wide)
    results.to_csv(OUT_DIR / "component_ablation_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "component_ablation_summary.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUT_DIR / "component_ablation_f1_by_test_center.csv", index=False, encoding="utf-8-sig")
    compact.to_csv(OUT_DIR / "component_ablation_compact_key_results.csv", index=False, encoding="utf-8-sig")
    source_diag.to_csv(OUT_DIR / "source_weight_ablation_weights.csv", index=False, encoding="utf-8-sig")

    report = [
        "# BioGait-MoE component ablation",
        "",
        "Segment-level macro-F1 is reported for each held-out CAREPD test center.",
        "",
        "## Mean performance by variant",
        "",
        summary.sort_values(["train_setting", "category", "mean_segment_macro_f1"], ascending=[True, True, False]).to_markdown(index=False),
        "",
        "## F1 by test center",
        "",
        wide.to_markdown(index=False),
        "",
        "## Compact key results",
        "",
        compact.to_markdown(index=False),
    ]
    (OUT_DIR / "COMPONENT_ABLATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR), "n_rows": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
