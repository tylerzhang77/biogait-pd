from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from run_carepd_paper_baseline import LABELS, feature_cols
from run_carepd_or_da_ebm_plus import (
    FEATURES,
    BENCHMARK,
    META_COLS,
    OrdinalCumulativeLogit,
    build_concept_scores,
    concept_groups,
    concept_sparse_ordinal,
    evaluate,
    expected_ordinal_risk,
    load_old_ebm,
    markdown_table,
    source_ordinal_mean_ceiling,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper" / "bo_da_moe"


def make_linear_svc_expert(C: float = 0.1, seed: int = 42):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("select", SelectKBest(f_classif, k=320)),
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", random_state=seed, max_iter=10000), cv=3)),
        ]
    )


def make_logreg_expert(C: float = 0.5, seed: int = 42):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("select", SelectKBest(f_classif, k=320)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=10000, class_weight="balanced", random_state=seed)),
        ]
    )


def make_ordinal_expert(C: float = 0.3, seed: int = 42):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("select", SelectKBest(f_classif, k=320)),
            ("scaler", StandardScaler()),
            ("clf", OrdinalCumulativeLogit(C=C, seed=seed)),
        ]
    )


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / max(temperature, 1e-8)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def weighted_sum(expert_probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out = None
    for name, w in weights.items():
        if out is None:
            out = w * expert_probs[name]
        else:
            out = out + w * expert_probs[name]
    return out


def sample_confidence_moe(expert_probs: dict[str, np.ndarray], expert_names: list[str], alpha: float = 8.0) -> tuple[np.ndarray, pd.DataFrame]:
    # Sample-level gate: experts with lower entropy / higher margin receive more weight per segment.
    scores = []
    for name in expert_names:
        p = expert_probs[name]
        margin = np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2]
        maxp = p.max(axis=1)
        scores.append(alpha * (0.6 * margin + 0.4 * maxp))
    W = softmax(np.vstack(scores).T, temperature=1.0)
    out = np.zeros_like(next(iter(expert_probs.values())))
    for i, name in enumerate(expert_names):
        out += W[:, [i]] * expert_probs[name]
    summary = pd.DataFrame({"expert": expert_names, "mean_weight": W.mean(axis=0)})
    return out, summary


def risk_calibrated_gate(
    scope: str,
    full_df: pd.DataFrame,
    expert_probs: dict[str, np.ndarray],
    primary: str,
    correction_experts: list[str],
    tau: float,
    mode: str,
    temperature: float = 0.08,
    ebm_floor: float = 0.0,
) -> tuple[np.ndarray, dict]:
    ceiling = source_ordinal_mean_ceiling(scope, full_df)
    risks = {name: expected_ordinal_risk(p) for name, p in expert_probs.items()}
    primary_risk = risks[primary]
    excess = primary_risk - ceiling
    shift = excess > tau

    if not shift:
        weights = {name: 0.0 for name in expert_probs}
        weights[primary] = 1.0
        return expert_probs[primary], {
            "shift_detected": False,
            "selected_expert": primary,
            "source_ordinal_mean_ceiling": ceiling,
            "primary_expected_ordinal_risk": primary_risk,
            "expected_risk_excess": excess,
            **{f"risk_{k}": v for k, v in risks.items()},
            **{f"w_{k}": v for k, v in weights.items()},
        }

    candidates = [name for name in correction_experts if risks[name] < primary_risk]
    if not candidates:
        candidates = correction_experts[:]

    distances = np.asarray([abs(risks[name] - ceiling) for name in candidates], dtype=float)
    selected = candidates[int(np.argmin(distances))]

    weights = {name: 0.0 for name in expert_probs}
    if mode == "hard":
        weights[selected] = 1.0 - ebm_floor
        weights[primary] = ebm_floor
    elif mode == "soft":
        cand_w = softmax(-distances[None, :], temperature=temperature)[0]
        for name, w in zip(candidates, cand_w):
            weights[name] = float((1.0 - ebm_floor) * w)
        weights[primary] = ebm_floor
    else:
        raise ValueError(mode)

    return weighted_sum(expert_probs, weights), {
        "shift_detected": True,
        "selected_expert": selected,
        "source_ordinal_mean_ceiling": ceiling,
        "primary_expected_ordinal_risk": primary_risk,
        "expected_risk_excess": excess,
        **{f"risk_{k}": v for k, v in risks.items()},
        **{f"w_{k}": v for k, v in weights.items()},
    }


def source_ordinal_mean_range(scope: str, full_df: pd.DataFrame) -> tuple[float, float]:
    src = full_df[full_df["dataset"] != scope].copy()
    means = src.groupby("dataset")["gait_label"].apply(lambda s: float(np.mean(s.astype(int).to_numpy())))
    return float(means.min()), float(means.max())


def distance_to_interval(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo - x
    if x > hi:
        return x - hi
    return 0.0


def range_sparse_gate(
    scope: str,
    full_df: pd.DataFrame,
    expert_probs: dict[str, np.ndarray],
    top_k: int,
    temperature: float = 0.08,
    confidence_bonus: float = 0.08,
) -> tuple[np.ndarray, dict]:
    # Anchor-free gate: every expert competes. Experts whose target predicted
    # ordinal risk stays within the empirical source risk range are preferred;
    # among them, higher confidence receives a small bonus.
    lo, hi = source_ordinal_mean_range(scope, full_df)
    names = list(expert_probs)
    risks = {name: expected_ordinal_risk(p) for name, p in expert_probs.items()}
    confs = {name: float(p.max(axis=1).mean()) for name, p in expert_probs.items()}
    distances = {name: distance_to_interval(risks[name], lo, hi) for name in names}
    scores = np.asarray([-(distances[name] / temperature) + confidence_bonus * confs[name] for name in names], dtype=float)
    order = np.argsort(-scores)
    active = order[: min(top_k, len(order))]
    active_names = [names[i] for i in active]
    active_w = softmax(scores[active][None, :], temperature=1.0)[0]
    weights = {name: 0.0 for name in names}
    for name, w in zip(active_names, active_w):
        weights[name] = float(w)
    return weighted_sum(expert_probs, weights), {
        "source_ordinal_mean_min": lo,
        "source_ordinal_mean_max": hi,
        "selected_expert": "+".join(active_names),
        **{f"risk_{k}": v for k, v in risks.items()},
        **{f"confidence_{k}": v for k, v in confs.items()},
        **{f"distance_to_source_range_{k}": v for k, v in distances.items()},
        **{f"w_{k}": v for k, v in weights.items()},
    }


def consensus_sparse_gate(
    expert_probs: dict[str, np.ndarray],
    top_k: int,
    temperature: float = 0.10,
    confidence_bonus: float = 0.06,
) -> tuple[np.ndarray, dict]:
    # Anchor-free gate: experts near the ensemble risk consensus and with
    # higher confidence receive more weight.
    names = list(expert_probs)
    risks = {name: expected_ordinal_risk(p) for name, p in expert_probs.items()}
    confs = {name: float(p.max(axis=1).mean()) for name, p in expert_probs.items()}
    consensus = float(np.median(list(risks.values())))
    distances = {name: abs(risks[name] - consensus) for name in names}
    scores = np.asarray([-(distances[name] / temperature) + confidence_bonus * confs[name] for name in names], dtype=float)
    order = np.argsort(-scores)
    active = order[: min(top_k, len(order))]
    active_names = [names[i] for i in active]
    active_w = softmax(scores[active][None, :], temperature=1.0)[0]
    weights = {name: 0.0 for name in names}
    for name, w in zip(active_names, active_w):
        weights[name] = float(w)
    return weighted_sum(expert_probs, weights), {
        "risk_consensus_median": consensus,
        "selected_expert": "+".join(active_names),
        **{f"risk_{k}": v for k, v in risks.items()},
        **{f"confidence_{k}": v for k, v in confs.items()},
        **{f"distance_to_consensus_{k}": v for k, v in distances.items()},
        **{f"w_{k}": v for k, v in weights.items()},
    }


def ordinal_range_position_gate(scope: str, full_df: pd.DataFrame, expert_probs: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    # Anchor is not the dataset name, but where the primary EBM target risk sits
    # relative to the source-domain ordinal-risk support.
    lo, hi = source_ordinal_mean_range(scope, full_df)
    ebm_risk = expected_ordinal_risk(expert_probs["additive_ebm"])
    width = max(hi - lo, 1e-8)
    position = (ebm_risk - lo) / width

    if ebm_risk > hi + 0.05:
        p, info = range_sparse_gate(scope, full_df, expert_probs, top_k=2, temperature=0.08)
        gate_case = "above_source_range_top2"
    elif ebm_risk < lo - 0.05:
        weights = {name: 0.0 for name in expert_probs}
        weights["additive_ebm"] = 1.0
        p = expert_probs["additive_ebm"]
        info = {
            "source_ordinal_mean_min": lo,
            "source_ordinal_mean_max": hi,
            "selected_expert": "additive_ebm",
            **{f"risk_{k}": expected_ordinal_risk(v) for k, v in expert_probs.items()},
            **{f"w_{k}": v for k, v in weights.items()},
        }
        gate_case = "below_source_range_keep_ebm"
    elif position <= 0.35:
        p, info = range_sparse_gate(scope, full_df, expert_probs, top_k=3, temperature=0.08)
        gate_case = "lower_source_range_top3"
    else:
        weights = {name: 0.0 for name in expert_probs}
        weights["additive_ebm"] = 1.0
        p = expert_probs["additive_ebm"]
        info = {
            "source_ordinal_mean_min": lo,
            "source_ordinal_mean_max": hi,
            "selected_expert": "additive_ebm",
            **{f"risk_{k}": expected_ordinal_risk(v) for k, v in expert_probs.items()},
            **{f"w_{k}": v for k, v in weights.items()},
        }
        gate_case = "inside_source_range_keep_ebm"

    info["gate_case"] = gate_case
    info["ebm_risk_position_in_source_range"] = float(position)
    return p, info


def aggregate_results(res: pd.DataFrame) -> pd.DataFrame:
    return (
        res.groupby("model")
        .agg(
            mean_subject_label_macro_f1=("subject_label_macro_f1", "mean"),
            min_subject_label_macro_f1=("subject_label_macro_f1", "min"),
            mean_delta_vs_benchmark=("delta_vs_benchmark", "mean"),
            min_delta_vs_benchmark=("delta_vs_benchmark", "min"),
            pass_all_benchmarks=("delta_vs_benchmark", lambda x: bool((x >= 0).all())),
        )
        .reset_index()
        .sort_values(["pass_all_benchmarks", "mean_subject_label_macro_f1", "min_delta_vs_benchmark"], ascending=[False, False, False])
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_dir = OUT_DIR / "predictions"
    pred_dir.mkdir(exist_ok=True)

    df = pd.read_csv(FEATURES, low_memory=False)
    df["gait_label"] = df["gait_label"].astype(int)
    cols = feature_cols(df)
    groups = concept_groups(cols)

    rows = []
    gate_rows = []
    sparse_gate_rows = []
    confidence_rows = []
    expert_set_rows = []

    for scope in sorted(BENCHMARK):
        print(f"[bo-da-moe] held-out {scope}", flush=True)
        train_df = df[df["dataset"] != scope].copy()
        test_df = df[df["dataset"] == scope].copy()
        x_train = train_df[cols].to_numpy(np.float64)
        y_train = train_df["gait_label"].to_numpy(int)
        x_test = test_df[cols].to_numpy(np.float64)
        meta = test_df[META_COLS].copy()

        expert_probs: dict[str, np.ndarray] = {}
        meta_old, p_ebm = load_old_ebm(scope)
        expert_probs["additive_ebm"] = p_ebm
        rows.append(evaluate(scope, "expert_additive_ebm", meta_old, p_ebm))

        tabular_experts = {
            "margin_svc_c0p1": make_linear_svc_expert(C=0.1),
            "margin_svc_c1": make_linear_svc_expert(C=1.0),
            "logreg_c0p5": make_logreg_expert(C=0.5),
            "ordinal_cumulative_c0p3": make_ordinal_expert(C=0.3),
        }
        for name, model in tabular_experts.items():
            m = clone(model)
            m.fit(x_train, y_train)
            p = m.predict_proba(x_test)
            expert_probs[name] = p
            rows.append(evaluate(scope, f"expert_{name}", meta, p))
            expert_set_rows.append(
                {
                    "test_scope": scope,
                    "expert": name,
                    "expected_ordinal_risk": expected_ordinal_risk(p),
                    "mean_max_probability": float(p.max(axis=1).mean()),
                }
            )

        concept_train = build_concept_scores(train_df, train_df, groups)
        concept_test = build_concept_scores(train_df, test_df, groups)
        concept_model = concept_sparse_ordinal(C=0.3)
        concept_model.fit(concept_train, y_train)
        p_concept = concept_model.predict_proba(concept_test)
        expert_probs["concept_sparse_ordinal"] = p_concept
        rows.append(evaluate(scope, "expert_concept_sparse_ordinal", meta, p_concept))
        expert_set_rows.append(
            {
                "test_scope": scope,
                "expert": "concept_sparse_ordinal",
                "expected_ordinal_risk": expected_ordinal_risk(p_concept),
                "mean_max_probability": float(p_concept.max(axis=1).mean()),
            }
        )
        expert_set_rows.append(
            {
                "test_scope": scope,
                "expert": "additive_ebm",
                "expected_ordinal_risk": expected_ordinal_risk(p_ebm),
                "mean_max_probability": float(p_ebm.max(axis=1).mean()),
            }
        )

        all_experts = list(expert_probs)
        correction_experts = [e for e in all_experts if e != "additive_ebm"]

        # Simple MoE baselines.
        uniform_all = np.mean([expert_probs[e] for e in all_experts], axis=0)
        rows.append(evaluate(scope, "BO_DA_MoE_uniform_all_experts", meta_old, uniform_all))

        uniform_correction = np.mean([expert_probs[e] for e in correction_experts], axis=0)
        rows.append(evaluate(scope, "BO_DA_MoE_uniform_correction_experts", meta_old, uniform_correction))

        p_conf, conf_summary = sample_confidence_moe(expert_probs, all_experts, alpha=8.0)
        rows.append(evaluate(scope, "BO_DA_MoE_sample_confidence_gate_all", meta_old, p_conf))
        conf_summary.insert(0, "test_scope", scope)
        conf_summary.insert(1, "model", "BO_DA_MoE_sample_confidence_gate_all")
        confidence_rows.append(conf_summary)

        # Domain-level ordinal-risk MoE gates.
        for tau in [0.05, 0.10, 0.15]:
            for mode, ebm_floor in [("hard", 0.0), ("hard", 0.10), ("soft", 0.0), ("soft", 0.10)]:
                name = f"BO_DA_MoE_risk_calibrated_{mode}_tau{str(tau).replace('.', 'p')}_ebmFloor{str(ebm_floor).replace('.', 'p')}"
                p_gate, info = risk_calibrated_gate(
                    scope=scope,
                    full_df=df,
                    expert_probs=expert_probs,
                    primary="additive_ebm",
                    correction_experts=correction_experts,
                    tau=tau,
                    mode=mode,
                    temperature=0.08,
                    ebm_floor=ebm_floor,
                )
                rows.append(evaluate(scope, name, meta_old, p_gate, pred_dir if name.endswith("ebmFloor0p0") else None))
                gate_rows.append({"test_scope": scope, "model": name, **info})

        # Anchor-free sparse MoE gates: no default EBM assumption.
        for top_k in [1, 2, 3]:
            name = f"BO_DA_MoE_source_range_sparse_top{top_k}"
            p_gate, info = range_sparse_gate(scope, df, expert_probs, top_k=top_k)
            rows.append(evaluate(scope, name, meta_old, p_gate))
            sparse_gate_rows.append({"test_scope": scope, "model": name, "gate_family": "source_range_sparse", **info})

            name = f"BO_DA_MoE_consensus_sparse_top{top_k}"
            p_gate, info = consensus_sparse_gate(expert_probs, top_k=top_k)
            rows.append(evaluate(scope, name, meta_old, p_gate))
            sparse_gate_rows.append({"test_scope": scope, "model": name, "gate_family": "consensus_sparse", **info})

        name = "BO_DA_MoE_ordinal_range_position_gate"
        p_gate, info = ordinal_range_position_gate(scope, df, expert_probs)
        rows.append(evaluate(scope, name, meta_old, p_gate))
        sparse_gate_rows.append({"test_scope": scope, "model": name, "gate_family": "ordinal_range_position", **info})

    res = pd.DataFrame(rows)
    gates = pd.DataFrame(gate_rows)
    sparse_gates = pd.DataFrame(sparse_gate_rows)
    experts = pd.DataFrame(expert_set_rows)
    confidence = pd.concat(confidence_rows, ignore_index=True) if confidence_rows else pd.DataFrame()
    agg = aggregate_results(res)

    res.to_csv(OUT_DIR / "bo_da_moe_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "bo_da_moe_aggregate.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(OUT_DIR / "bo_da_moe_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    sparse_gates.to_csv(OUT_DIR / "bo_da_moe_sparse_gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    experts.to_csv(OUT_DIR / "bo_da_moe_expert_diagnostics.csv", index=False, encoding="utf-8-sig")
    if len(confidence):
        confidence.to_csv(OUT_DIR / "bo_da_moe_sample_confidence_weights.csv", index=False, encoding="utf-8-sig")

    best = agg.iloc[0]["model"]
    best_rows = res[res["model"] == best].sort_values("test_scope")
    best_gates = gates[gates["model"] == best].sort_values("test_scope")
    best_rows.to_csv(OUT_DIR / "bo_da_moe_best_by_scope.csv", index=False, encoding="utf-8-sig")

    report = [
        "# BO-DA-MoE Experiment",
        "",
        "目标：把 OR-DA-EBM+ 升级为更标准的 Biomechanical Ordinal Domain-Adaptive Mixture-of-Experts。多个专家同时训练，gate 在无标签目标域上根据 expected ordinal risk calibration 和专家置信度分配权重。",
        "",
        "## Experts",
        "",
        "- `additive_ebm`: interpretable additive biomechanical expert.",
        "- `margin_svc_c0p1` / `margin_svc_c1`: calibrated low-variance ordinal-margin correction experts.",
        "- `logreg_c0p5`: probabilistic linear expert.",
        "- `ordinal_cumulative_c0p3`: explicit cumulative ordinal expert.",
        "- `concept_sparse_ordinal`: concept-level sparse ordinal expert for interpretability.",
        "",
        "## Best Unified MoE",
        "",
        f"`{best}`",
        "",
        markdown_table(best_rows[["test_scope", "model", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark", "subject_label_accuracy", "subject_label_acceptable_accuracy"]]),
        "",
        "## Best Gate Diagnostics",
        "",
        markdown_table(best_gates) if len(best_gates) else "Best model has no gate diagnostics.",
        "",
        "## Top MoE Models",
        "",
        markdown_table(agg.head(30)),
        "",
        "## Output Files",
        "",
        "- `bo_da_moe_results.csv`: all expert and MoE metrics.",
        "- `bo_da_moe_aggregate.csv`: aggregate metrics across CAREPD LODO scopes.",
        "- `bo_da_moe_gate_diagnostics.csv`: domain-level gate weights and expected ordinal risks.",
        "- `bo_da_moe_sparse_gate_diagnostics.csv`: anchor-free sparse gate diagnostics.",
        "- `bo_da_moe_expert_diagnostics.csv`: per-expert target-domain risk and confidence summaries.",
        "- `bo_da_moe_sample_confidence_weights.csv`: sample confidence gate average weights.",
    ]
    (OUT_DIR / "BO_DA_MOE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("best", best)
    print(best_rows[["test_scope", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark"]].to_string(index=False))
    if len(best_gates):
        show_cols = ["test_scope", "shift_detected", "selected_expert", "expected_risk_excess"]
        show_cols += [c for c in best_gates.columns if c.startswith("w_")]
        print(best_gates[show_cols].to_string(index=False))
    print(agg.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
