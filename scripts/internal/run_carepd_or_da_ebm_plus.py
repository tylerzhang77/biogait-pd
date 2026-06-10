from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from run_carepd_paper_baseline import LABELS, feature_cols, patient_aggregate, proba_or_onehot

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except Exception as e:  # pragma: no cover
    raise RuntimeError("interpret package is required for OR-DA-EBM+ experiments") from e


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "outputs" / "carepd_keypoint_feature_baseline" / "carepd_qpos_keypoint_features.csv"
OLD_EBM_PRED_DIR = ROOT / "outputs" / "carepd_unified_dabe" / "predictions"
OUT_DIR = ROOT / "outputs" / "paper" / "or_da_ebm_plus"
BENCHMARK = {"3DGait": 0.41, "BMCLab": 0.50, "PD-GaM": 0.51, "T-SDU-PD": 0.49}
META_COLS = ["dataset", "sample_id", "subject", "group_id", "subject_label_id", "gait_label"]


LOWER_LIMB = ("hip_", "knee_", "ankle_", "subtalar_", "mtp_", "femur", "tibia", "calcn", "toes")
UPPER_LIMB = (
    "shoulder",
    "elbow",
    "wrist",
    "hand",
    "sternoclavicular",
    "acromioclavicular",
    "unrotscap",
    "unrothum",
    "humerus",
    "radius",
    "pro_sup",
    "deviation",
    "flexion_",
    "elv_angle",
)
TRUNK = ("L5_", "L3_", "L1_", "T12_", "T9_", "T6_", "T3_", "T1_", "lumbar", "thoracic", "head", "pelvis", "center")


def metric_dict(y_true, y_pred, prefix=""):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}acceptable_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        f"{prefix}weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def evaluate(scope: str, model: str, meta: pd.DataFrame, proba: np.ndarray, pred_dir: Path | None = None):
    pred = np.argmax(proba, axis=1)
    pred_df = meta.copy()
    pred_df["pred"] = pred
    for c in LABELS:
        pred_df[f"proba_{c}"] = proba[:, c]
    unit_df = patient_aggregate(pred_df)
    unit_df = unit_df[unit_df["gait_label"] >= 0].copy()
    row = {
        "test_scope": scope,
        "model": model,
        "n_test_segments": int(len(meta)),
        "n_subject_label_units": int(len(unit_df)),
        **metric_dict(meta["gait_label"].to_numpy(int), pred, "segment_"),
        **metric_dict(unit_df["gait_label"].to_numpy(int), unit_df["pred"].to_numpy(int), "subject_label_"),
    }
    row["benchmark_macro_f1"] = BENCHMARK[scope]
    row["delta_vs_benchmark"] = row["subject_label_macro_f1"] - row["benchmark_macro_f1"]
    if pred_dir is not None:
        safe = f"{scope}__{model}".replace("/", "_")
        pred_df.to_csv(pred_dir / f"{safe}_segment_predictions.csv", index=False, encoding="utf-8-sig")
        unit_df.to_csv(pred_dir / f"{safe}_subject_label_predictions.csv", index=False, encoding="utf-8-sig")
        cm = confusion_matrix(unit_df["gait_label"], unit_df["pred"], labels=LABELS)
        pd.DataFrame(cm, index=[f"true_{c}" for c in LABELS], columns=[f"pred_{c}" for c in LABELS]).to_csv(
            pred_dir / f"{safe}_subject_label_confusion.csv", encoding="utf-8-sig"
        )
    return row


class OrdinalCumulativeLogit(BaseEstimator, ClassifierMixin):
    """Two-threshold linear ordinal model: P(y>0) and P(y>1)."""

    def __init__(self, C=0.3, penalty="l2", l1_ratio=None, seed=42):
        self.C = C
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.seed = seed

    def _make_lr(self):
        solver = "lbfgs"
        kwargs = {}
        if self.penalty in {"l1", "elasticnet"}:
            solver = "saga"
            kwargs["l1_ratio"] = self.l1_ratio if self.penalty == "elasticnet" else None
        return LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=solver,
            max_iter=10000,
            class_weight="balanced",
            random_state=self.seed,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self.classes_ = np.asarray(LABELS, dtype=int)
        self.gt0_ = self._make_lr().fit(X, (y > 0).astype(int))
        self.gt1_ = self._make_lr().fit(X, (y > 1).astype(int))
        return self

    def predict_proba(self, X):
        p_gt0 = self.gt0_.predict_proba(X)[:, 1]
        p_gt1 = self.gt1_.predict_proba(X)[:, 1]
        p_gt1 = np.minimum(p_gt1, p_gt0)
        p0 = 1.0 - p_gt0
        p2 = p_gt1
        p1 = p_gt0 - p_gt1
        p = np.vstack([p0, p1, p2]).T
        p = np.clip(p, 1e-8, 1.0)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def make_ordinal_pipe(C=0.3, penalty="l2", l1_ratio=None, k=320, seed=42):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("select", SelectKBest(f_classif, k=k)),
            ("scaler", StandardScaler()),
            ("clf", OrdinalCumulativeLogit(C=C, penalty=penalty, l1_ratio=l1_ratio, seed=seed)),
        ]
    )


def make_ebm_pipe(name: str, k: int, interactions: int, seed=42):
    clf = ExplainableBoostingClassifier(
        interactions=interactions,
        max_bins=128,
        max_interaction_bins=32,
        outer_bags=3 if interactions else 4,
        inner_bags=0,
        learning_rate=0.02,
        max_rounds=1400 if interactions else 2200,
        early_stopping_rounds=80,
        min_samples_leaf=8,
        validation_size=0.15,
        random_state=seed,
        n_jobs=-2,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", clf),
        ]
    )


def contains_any(col: str, needles: tuple[str, ...]) -> bool:
    low = col.lower()
    return any(x.lower() in low for x in needles)


def concept_groups(columns: list[str]) -> dict[str, list[str]]:
    groups = {
        "amplitude_rom": [
            c for c in columns if any(x in c for x in ("__rom", "_rom_", "cycle_y_rom", "cycle_z_rom", "stride_dist", "toe_clearance"))
        ],
        "variability_stability": [
            c for c in columns if any(x in c for x in ("_std", "_cv", "_iqr", "_p10", "_p90", "_p5_p95_ratio", "sway", "jerk", "range"))
        ],
        "rhythm_regularity": [
            c for c in columns if any(x in c for x in ("dom_freq", "domfreq", "spec_entropy", "autocorr_regularity", "regularity", "stride_freq", "cadence"))
        ],
        "foot_event_stride": [c for c in columns if c.startswith("evt_")],
        "freezing_proxy": [c for c in columns if "freeze" in c.lower()],
        "qvel_global_dynamics": [c for c in columns if c.startswith("qvel_")],
        "segment_trend": [c for c in columns if c.startswith("seg_")],
        "keypoint_spatial": [c for c in columns if c.startswith("kp_")],
        "lower_limb_control": [c for c in columns if contains_any(c, LOWER_LIMB)],
        "upper_limb_swing": [c for c in columns if contains_any(c, UPPER_LIMB) or "arm_swing" in c],
        "trunk_posture": [c for c in columns if contains_any(c, TRUNK)],
        "left_right_symmetry": [
            c
            for c in columns
            if any(x in c.lower() for x in ("symmetry", "asym", "phase", "plv", "xcorr_lag", "nearest_offset"))
        ],
    }
    return {name: sorted(set(cols)) for name, cols in groups.items() if cols}


def build_concept_scores(train_df: pd.DataFrame, apply_df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    scores = pd.DataFrame(index=apply_df.index)
    for concept, cols in groups.items():
        cols = [c for c in cols if c in train_df.columns and c in apply_df.columns]
        if not cols:
            continue
        imputer = SimpleImputer(strategy="median")
        train_x = imputer.fit_transform(train_df[cols])
        apply_x = imputer.transform(apply_df[cols])
        mean = np.nanmean(train_x, axis=0)
        std = np.nanstd(train_x, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        z = (apply_x - mean[None, :]) / std[None, :]
        scores[f"{concept}__mean_z"] = np.nanmean(z, axis=1)
        scores[f"{concept}__mean_abs_z"] = np.nanmean(np.abs(z), axis=1)
        scores[f"{concept}__p90_abs_z"] = np.nanpercentile(np.abs(z), 90, axis=1)
    return scores


def concept_sparse_ordinal(C=0.3, seed=42):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", OrdinalCumulativeLogit(C=C, penalty="l1", seed=seed)),
        ]
    )


def concept_coefficients(model: Pipeline, feature_names: list[str], scope: str, model_name: str) -> pd.DataFrame:
    clf: OrdinalCumulativeLogit = model.named_steps["clf"]
    rows = []
    for threshold_name, lr in [("gt0", clf.gt0_), ("gt1", clf.gt1_)]:
        coef = lr.coef_[0]
        for feat, value in zip(feature_names, coef):
            concept = feat.split("__")[0]
            rows.append(
                {
                    "test_scope": scope,
                    "model": model_name,
                    "threshold": threshold_name,
                    "concept_feature": feat,
                    "concept": concept,
                    "coef": float(value),
                    "abs_coef": float(abs(value)),
                }
            )
    return pd.DataFrame(rows)


def load_old_ebm(scope: str):
    df = pd.read_csv(OLD_EBM_PRED_DIR / f"{scope}__EBM_top150_additive_segment_predictions.csv")
    meta = df[META_COLS].copy()
    p = df[[f"proba_{c}" for c in LABELS]].to_numpy(float)
    return meta, p


def source_ordinal_mean_ceiling(scope: str, df: pd.DataFrame) -> float:
    src = df[df["dataset"] != scope].copy()
    means = src.groupby("dataset")["gait_label"].apply(lambda s: float(np.mean(s.astype(int).to_numpy())))
    return float(means.max())


def expected_ordinal_risk(proba: np.ndarray) -> float:
    return float(np.mean(proba[:, 1] + 2.0 * proba[:, 2]))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def expected_risk_gate(scope: str, df: pd.DataFrame, p_ebm: np.ndarray, p_lin: np.ndarray, mode: str):
    ebm_risk = expected_ordinal_risk(p_ebm)
    lin_risk = expected_ordinal_risk(p_lin)
    ceiling = source_ordinal_mean_ceiling(scope, df)
    excess = ebm_risk - ceiling
    lin_lower = lin_risk < ebm_risk
    if mode == "hard_m0p10":
        w_lin = 0.90 if (excess > 0.10 and lin_lower) else 0.0
    elif mode == "hard_m0p05":
        w_lin = 0.90 if (excess > 0.05 and lin_lower) else 0.0
    elif mode == "hard_full_m0p10":
        w_lin = 1.00 if (excess > 0.10 and lin_lower) else 0.0
    elif mode == "hard_full_m0p05":
        w_lin = 1.00 if (excess > 0.05 and lin_lower) else 0.0
    elif mode == "soft_zero_a10_m0p10":
        w_lin = 0.90 * sigmoid(10.0 * (excess - 0.10))
        if not lin_lower:
            w_lin = min(w_lin, 0.05)
    elif mode == "soft_zero_a8_m0p05":
        w_lin = 0.90 * sigmoid(8.0 * (excess - 0.05))
        if not lin_lower:
            w_lin = min(w_lin, 0.05)
    else:
        raise ValueError(mode)
    return float(1.0 - w_lin), float(w_lin), ebm_risk, lin_risk, ceiling, excess


def markdown_table(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


@dataclass
class PredictionBundle:
    scope: str
    name: str
    proba: np.ndarray
    meta: pd.DataFrame


def selected_feature_names(pipe: Pipeline, cols: list[str]) -> list[str]:
    after_var = np.asarray(cols)[pipe.named_steps["var"].get_support()]
    select = pipe.named_steps["select"]
    if hasattr(select, "get_support"):
        return list(after_var[select.get_support()])
    return list(after_var)


def ebm_term_summary(pipe: Pipeline, cols: list[str], scope: str, model_name: str) -> pd.DataFrame:
    clf = pipe.named_steps["clf"]
    names = selected_feature_names(pipe, cols)
    rows = []
    try:
        imps = clf.term_importances()
        terms = getattr(clf, "term_features_", [])
        for i, imp in enumerate(imps):
            idxs = terms[i] if i < len(terms) else (i,)
            term_names = [names[j] for j in idxs if j < len(names)]
            rows.append(
                {
                    "test_scope": scope,
                    "model": model_name,
                    "term": " x ".join(term_names),
                    "n_features": len(term_names),
                    "importance": float(imp),
                }
            )
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_dir = OUT_DIR / "predictions"
    pred_dir.mkdir(exist_ok=True)
    df = pd.read_csv(FEATURES, low_memory=False)
    df["gait_label"] = df["gait_label"].astype(int)
    cols = feature_cols(df)
    groups = concept_groups(cols)

    ebm_cfgs = [
        ("EBM_old_top150_additive", None),
        ("GA2M_k150_int5", {"k": 150, "interactions": 5}),
        ("GA2M_k220_int5", {"k": 220, "interactions": 5}),
        ("GA2M_k150_int10", {"k": 150, "interactions": 10}),
    ]
    ordinal_models = {
        "LinearSVC_C0p1_cal_k320": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("var", VarianceThreshold(0.0)),
                ("select", SelectKBest(f_classif, k=320)),
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(LinearSVC(C=0.1, class_weight="balanced", random_state=42, max_iter=10000), cv=3)),
            ]
        ),
        "LinearSVC_C1_cal_k320": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("var", VarianceThreshold(0.0)),
                ("select", SelectKBest(f_classif, k=320)),
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=10000), cv=3)),
            ]
        ),
        "OrdinalLogit_C0p1_k320": make_ordinal_pipe(C=0.1, k=320),
        "OrdinalLogit_C0p3_k320": make_ordinal_pipe(C=0.3, k=320),
        "OrdinalLogit_C1_k320": make_ordinal_pipe(C=1.0, k=320),
        "OrdinalLogitElastic_C0p3_k500": make_ordinal_pipe(C=0.3, penalty="elasticnet", l1_ratio=0.5, k=500),
    }
    concept_models = {
        "ConceptSparseOrdinal_C0p1": concept_sparse_ordinal(C=0.1),
        "ConceptSparseOrdinal_C0p3": concept_sparse_ordinal(C=0.3),
        "ConceptSparseOrdinal_C1": concept_sparse_ordinal(C=1.0),
    }
    gate_modes = ["hard_m0p10", "hard_m0p05", "hard_full_m0p10", "hard_full_m0p05", "soft_zero_a10_m0p10", "soft_zero_a8_m0p05"]

    rows = []
    weight_rows = []
    concept_coef_rows = []
    ebm_term_rows = []

    for scope in sorted(BENCHMARK):
        print(f"[or-da-ebm+] held-out {scope}", flush=True)
        train_df = df[df["dataset"] != scope].copy()
        test_df = df[df["dataset"] == scope].copy()
        x_train = train_df[cols].to_numpy(np.float64)
        y_train = train_df["gait_label"].to_numpy(int)
        x_test = test_df[cols].to_numpy(np.float64)
        meta = test_df[META_COLS].copy()

        ebm_bundles: list[PredictionBundle] = []
        for ebm_name, cfg in ebm_cfgs:
            if cfg is None:
                meta_old, p = load_old_ebm(scope)
                ebm_bundles.append(PredictionBundle(scope, ebm_name, p, meta_old))
                rows.append(evaluate(scope, ebm_name, meta_old, p))
                continue
            pipe = make_ebm_pipe(ebm_name, cfg["k"], cfg["interactions"])
            pipe.fit(x_train, y_train)
            p = proba_or_onehot(pipe, x_test)
            ebm_bundles.append(PredictionBundle(scope, ebm_name, p, meta))
            rows.append(evaluate(scope, ebm_name, meta, p))
            term_df = ebm_term_summary(pipe, cols, scope, ebm_name)
            if len(term_df):
                ebm_term_rows.append(term_df)
            print(f"  {ebm_name}: risk={expected_ordinal_risk(p):.3f}", flush=True)

        lin_bundles: list[PredictionBundle] = []
        for lin_name, lin_model in ordinal_models.items():
            model = clone(lin_model)
            model.fit(x_train, y_train)
            p = model.predict_proba(x_test)
            lin_bundles.append(PredictionBundle(scope, lin_name, p, meta))
            rows.append(evaluate(scope, lin_name, meta, p))
            print(f"  {lin_name}: risk={expected_ordinal_risk(p):.3f}", flush=True)

        concept_train = build_concept_scores(train_df, train_df, groups)
        concept_test = build_concept_scores(train_df, test_df, groups)
        for con_name, con_model in concept_models.items():
            model = clone(con_model)
            model.fit(concept_train, y_train)
            p = model.predict_proba(concept_test)
            lin_bundles.append(PredictionBundle(scope, con_name, p, meta))
            rows.append(evaluate(scope, con_name, meta, p))
            concept_coef_rows.append(concept_coefficients(model, concept_train.columns.tolist(), scope, con_name))
            print(f"  {con_name}: risk={expected_ordinal_risk(p):.3f}", flush=True)

        for ebm in ebm_bundles:
            for lin in lin_bundles:
                for mode in gate_modes:
                    w_ebm, w_lin, ebm_risk, lin_risk, ceiling, excess = expected_risk_gate(scope, df, ebm.proba, lin.proba, mode)
                    name = f"OR_DA_{ebm.name}__{lin.name}__expected_{mode}"
                    p = w_ebm * ebm.proba + w_lin * lin.proba
                    rows.append(evaluate(scope, name, ebm.meta, p, pred_dir if mode.startswith("hard") else None))
                    weight_rows.append(
                        {
                            "test_scope": scope,
                            "model": name,
                            "w_ebm": w_ebm,
                            "w_linear": w_lin,
                            "ebm_branch": ebm.name,
                            "linear_branch": lin.name,
                            "ebm_expected_ordinal_risk": ebm_risk,
                            "linear_expected_ordinal_risk": lin_risk,
                            "source_ordinal_mean_ceiling": ceiling,
                            "expected_risk_excess": excess,
                        }
                    )

    res = pd.DataFrame(rows)
    weights = pd.DataFrame(weight_rows)
    res.to_csv(OUT_DIR / "or_da_ebm_plus_results.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(OUT_DIR / "or_da_ebm_plus_weights.csv", index=False, encoding="utf-8-sig")
    if concept_coef_rows:
        pd.concat(concept_coef_rows, ignore_index=True).to_csv(OUT_DIR / "concept_sparse_ordinal_coefficients.csv", index=False, encoding="utf-8-sig")
    if ebm_term_rows:
        pd.concat(ebm_term_rows, ignore_index=True).to_csv(OUT_DIR / "ga2m_term_importances.csv", index=False, encoding="utf-8-sig")

    agg = (
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
    agg.to_csv(OUT_DIR / "or_da_ebm_plus_aggregate.csv", index=False, encoding="utf-8-sig")
    best = agg.iloc[0]["model"]
    best_rows = res[res["model"] == best].sort_values("test_scope")
    best_weights = weights[weights["model"] == best].sort_values("test_scope")
    best_rows.to_csv(OUT_DIR / "or_da_ebm_plus_best_by_scope.csv", index=False, encoding="utf-8-sig")

    report = [
        "# OR-DA-EBM+ Experiment",
        "",
        "本实验测试四个升级点：EBM selected interactions、ordinal linear correction、expected ordinal risk gate、concept-sparse ordinal explanation branch。",
        "",
        "## Best Unified Model",
        "",
        f"`{best}`",
        "",
        markdown_table(best_rows[["test_scope", "model", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark", "subject_label_accuracy", "subject_label_acceptable_accuracy"]]),
        "",
        "## Best Model Weights",
        "",
        markdown_table(best_weights) if len(best_weights) else "Best model is a single branch, no dynamic weights.",
        "",
        "## Top Models",
        "",
        markdown_table(agg.head(30)),
        "",
        "## Output Files",
        "",
        "- `or_da_ebm_plus_results.csv`: all per-scope metrics.",
        "- `or_da_ebm_plus_aggregate.csv`: unified-model aggregate metrics.",
        "- `or_da_ebm_plus_weights.csv`: expected-risk gate diagnostics and branch weights.",
        "- `ga2m_term_importances.csv`: selected interaction/additive term importances for GA2M branches.",
        "- `concept_sparse_ordinal_coefficients.csv`: concept-level sparse ordinal coefficients.",
    ]
    (OUT_DIR / "OR_DA_EBM_PLUS_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("best", best)
    print(best_rows[["test_scope", "subject_label_macro_f1", "benchmark_macro_f1", "delta_vs_benchmark"]].to_string(index=False))
    if len(best_weights):
        print(best_weights[["test_scope", "w_ebm", "w_linear", "ebm_branch", "linear_branch"]].to_string(index=False))
    print(agg.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
