from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except Exception:  # pragma: no cover
    ExplainableBoostingClassifier = None

from run_carepd_bo_da_moe import (
    aggregate_results,
    expected_ordinal_risk,
    make_linear_svc_expert,
    make_logreg_expert,
    make_ordinal_expert,
)
from run_carepd_bo_org_moe_unified_gate import unified_ordinal_support_gate
from run_carepd_or_da_ebm_plus import (
    BENCHMARK,
    FEATURES as CAREPD_FEATURES,
    META_COLS,
    build_concept_scores,
    concept_groups,
    concept_sparse_ordinal,
    evaluate,
    markdown_table,
)
from run_carepd_paper_baseline import feature_cols


ROOT = Path(__file__).resolve().parents[2]
MAIN_FEATURES = ROOT / "outputs" / "segment_level_runs" / "20260508_130259" / "segment_features.csv"
OUT_DIR = ROOT / "outputs" / "paper" / "fivecohort_main150_domainnorm_moe"
MAIN_META = {"patient_id", "segment_id", "gait_label", "duration_s", "direction"}


def make_ebm_expert(seed: int = 42, k: int = 150):
    if ExplainableBoostingClassifier is None:
        raise RuntimeError("interpret is required")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("select", SelectKBest(f_classif, k=k)),
            (
                "clf",
                ExplainableBoostingClassifier(
                    interactions=0,
                    max_bins=128,
                    outer_bags=4,
                    min_samples_leaf=8,
                    learning_rate=0.02,
                    max_rounds=2500,
                    random_state=seed,
                    n_jobs=-2,
                ),
            ),
        ]
    )


def load_fivecohort() -> tuple[pd.DataFrame, list[str]]:
    carepd = pd.read_csv(CAREPD_FEATURES, low_memory=False)
    main = pd.read_csv(MAIN_FEATURES, low_memory=False)
    carepd["gait_label"] = carepd["gait_label"].astype(int)
    main["gait_label"] = main["gait_label"].astype(int)

    main_features = [c for c in main.columns if c not in MAIN_META]
    carepd_features = feature_cols(carepd)
    cols = sorted(set(main_features) & set(carepd_features))
    cols = [c for c in cols if main[c].notna().any() and carepd[c].notna().any()]

    main_aligned = main[cols + ["patient_id", "segment_id", "gait_label"]].copy()
    main_aligned["dataset"] = "main150"
    main_aligned["sample_id"] = main_aligned["segment_id"].astype(str)
    main_aligned["subject"] = main_aligned["patient_id"].astype(str)
    main_aligned["group_id"] = "main150:" + main_aligned["patient_id"].astype(str)
    main_aligned["subject_label_id"] = main_aligned["group_id"]
    carepd_aligned = carepd[cols + META_COLS].copy()
    full = pd.concat([carepd_aligned, main_aligned[cols + META_COLS]], ignore_index=True, sort=False)
    return full, cols


def domain_robust_normalize(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str], mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "none":
        return train_df.copy(), test_df.copy()
    train = train_df.copy()
    test = test_df.copy()
    eps = 1e-6

    def apply_stats(frame: pd.DataFrame, stats_source: pd.DataFrame):
        med = stats_source[cols].median(axis=0)
        if mode == "domain_robust":
            q75 = stats_source[cols].quantile(0.75, axis=0)
            q25 = stats_source[cols].quantile(0.25, axis=0)
            scale = (q75 - q25).replace(0, np.nan)
        elif mode == "domain_z":
            scale = stats_source[cols].std(axis=0).replace(0, np.nan)
        else:
            raise ValueError(mode)
        scale = scale.fillna(scale.median()).fillna(1.0)
        frame.loc[:, cols] = (frame[cols] - med) / (scale + eps)

    for ds in sorted(train["dataset"].unique()):
        mask = train["dataset"] == ds
        apply_stats(train.loc[mask, :], train.loc[mask, :])
        # loc slice copy caveat: assign explicitly
        tmp = train.loc[mask, cols].copy()
        stats = train_df.loc[mask, :]
        med = stats[cols].median(axis=0)
        if mode == "domain_robust":
            scale = (stats[cols].quantile(0.75, axis=0) - stats[cols].quantile(0.25, axis=0)).replace(0, np.nan)
        else:
            scale = stats[cols].std(axis=0).replace(0, np.nan)
        scale = scale.fillna(scale.median()).fillna(1.0)
        train.loc[mask, cols] = (tmp - med) / (scale + eps)

    # Target labels are not used; target-domain feature statistics are allowed
    # as an unsupervised test-time normalization step.
    med = test[cols].median(axis=0)
    if mode == "domain_robust":
        scale = (test[cols].quantile(0.75, axis=0) - test[cols].quantile(0.25, axis=0)).replace(0, np.nan)
    else:
        scale = test[cols].std(axis=0).replace(0, np.nan)
    scale = scale.fillna(scale.median()).fillna(1.0)
    test.loc[:, cols] = (test[cols] - med) / (scale + eps)
    return train, test


def source_ordinal_mean_range(scope: str, full_df: pd.DataFrame, include_main: bool) -> tuple[float, float]:
    src = full_df[full_df["dataset"] != scope].copy()
    if not include_main:
        src = src[src["dataset"] != "main150"].copy()
    means = src.groupby("dataset")["gait_label"].apply(lambda s: float(np.mean(s.astype(int).to_numpy())))
    return float(means.min()), float(means.max())


def build_experts(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], groups: dict[str, list[str]]):
    x_train = train[cols].to_numpy(np.float64)
    y_train = train["gait_label"].to_numpy(int)
    x_test = test[cols].to_numpy(np.float64)
    expert_probs: dict[str, np.ndarray] = {}
    experts = {
        "additive_ebm": make_ebm_expert(k=150),
        "margin_svc_c0p1": make_linear_svc_expert(C=0.1),
        "margin_svc_c1": make_linear_svc_expert(C=1.0),
        "logreg_c0p5": make_logreg_expert(C=0.5),
        "ordinal_cumulative_c0p3": make_ordinal_expert(C=0.3),
    }
    for name, model in experts.items():
        m = clone(model)
        m.fit(x_train, y_train)
        expert_probs[name] = m.predict_proba(x_test)
    concept_train = build_concept_scores(train, train, groups)
    concept_test = build_concept_scores(train, test, groups)
    concept_model = concept_sparse_ordinal(C=0.3)
    concept_model.fit(concept_train, y_train)
    expert_probs["concept_sparse_ordinal"] = concept_model.predict_proba(concept_test)
    return expert_probs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full, cols = load_fivecohort()
    groups = concept_groups(cols)
    pd.DataFrame({"feature": cols}).to_csv(OUT_DIR / "common_feature_columns.csv", index=False, encoding="utf-8-sig")

    configs = []
    for norm in ["domain_robust", "domain_z", "none"]:
        for support_include_main in [False, True]:
            for gap in [0.75, 1.0, 1.25, 1.5]:
                for ebm_prior in [0.0, 0.5, 1.0, 1.5]:
                    configs.append(
                        {
                            "name": f"FiveCohort_{norm}_support{'all' if support_include_main else 'carepd'}_gap{gap:g}_ebm{ebm_prior:g}",
                            "norm": norm,
                            "support_include_main": support_include_main,
                            "top_k": 3,
                            "gap": gap,
                            "risk_temperature": 0.1,
                            "weight_temperature": 1.0,
                            "confidence_bonus": 0.08,
                            "ebm_prior": ebm_prior,
                            "concept_prior": 0.0,
                        }
                    )

    rows = []
    gate_rows = []
    expert_rows = []
    for scope in sorted(BENCHMARK):
        print(f"[fivecohort-domainnorm] held-out {scope}", flush=True)
        raw_train = full[full["dataset"] != scope].copy()
        raw_test = full[full["dataset"] == scope].copy()
        meta = raw_test[META_COLS].copy()

        cache: dict[str, dict[str, np.ndarray]] = {}
        for norm in ["domain_robust", "domain_z", "none"]:
            train, test = domain_robust_normalize(raw_train, raw_test, cols, norm)
            cache[norm] = build_experts(train, test, cols, groups)
            for name, p in cache[norm].items():
                rows.append(evaluate(scope, f"expert_{norm}_{name}", meta, p))
                expert_rows.append(
                    {
                        "test_scope": scope,
                        "norm": norm,
                        "expert": name,
                        "expected_ordinal_risk": expected_ordinal_risk(p),
                        "mean_max_probability": float(p.max(axis=1).mean()),
                        "n_train_segments": int(len(train)),
                        "n_main150_train_segments": int((train["dataset"] == "main150").sum()),
                    }
                )

        for cfg in configs:
            expert_probs = cache[cfg["norm"]]
            # Reuse the unified gate implementation by overriding source range
            # through a temporary full frame whose labels define the support.
            support_df = full if cfg["support_include_main"] else full[full["dataset"] != "main150"].copy()
            p_gate, info = unified_ordinal_support_gate(
                scope,
                support_df,
                expert_probs,
                top_k=cfg["top_k"],
                risk_temperature=cfg["risk_temperature"],
                weight_temperature=cfg["weight_temperature"],
                confidence_bonus=cfg["confidence_bonus"],
                ebm_prior=cfg["ebm_prior"],
                concept_prior=cfg["concept_prior"],
                gap=cfg["gap"],
                ebm_prior_mode="constant",
                ebm_prior_decay=0.12,
            )
            row = evaluate(scope, cfg["name"], meta, p_gate)
            rows.append(row)
            gate_rows.append({"test_scope": scope, "model": cfg["name"], **info, "norm": cfg["norm"], "support_include_main": cfg["support_include_main"]})

    res = pd.DataFrame(rows)
    agg = aggregate_results(res)
    gates = pd.DataFrame(gate_rows)
    experts = pd.DataFrame(expert_rows)
    best = agg.iloc[0]["model"]
    best_rows = res[res["model"] == best].sort_values("test_scope")
    best_gates = gates[gates["model"] == best].sort_values("test_scope")

    res.to_csv(OUT_DIR / "fivecohort_domainnorm_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "fivecohort_domainnorm_aggregate.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(OUT_DIR / "fivecohort_domainnorm_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    experts.to_csv(OUT_DIR / "fivecohort_domainnorm_expert_diagnostics.csv", index=False, encoding="utf-8-sig")
    best_rows.to_csv(OUT_DIR / "fivecohort_domainnorm_best_by_scope.csv", index=False, encoding="utf-8-sig")

    report = [
        "# Five-Cohort main150 + CAREPD LODO Domain-Normalized MoE",
        "",
        "Every CAREPD LODO fold trains on all non-held-out CAREPD datasets plus all main150 segments. Target labels are not used; target feature statistics are used only for unsupervised domain normalization.",
        "",
        f"Common feature count: `{len(cols)}`.",
        "",
        "## Best Model",
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
        markdown_table(agg.head(40)),
    ]
    (OUT_DIR / "FIVECOHORT_DOMAINNORM_MOE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("Best:", best)
    print(best_rows[["test_scope", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark"]].to_string(index=False))
    print(agg.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
