from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from parkinson_gait.advanced_features import extract_qvel_features, extract_segment_features
from parkinson_gait.data_io import build_qpos_mapping, load_motion
from parkinson_gait.features import extract_gait_proxy_features, extract_joint_features

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except Exception:  # pragma: no cover
    ExplainableBoostingClassifier = None


RETARGET_ROOT = Path(os.environ.get("BIOGAIT_RETARGET_ROOT", ROOT / "data" / "retargeted"))
CAREPD = RETARGET_ROOT / "CAREPD"
DATASETS = {
    "3DGait": RETARGET_ROOT / "CAREPDretarget" / "01_3DGait",
    "BMCLab": RETARGET_ROOT / "CAREPDretarget" / "01_BMCLab",
    "PD-GaM": RETARGET_ROOT / "CAREPDretarget_rest" / "04_PD-GaM",
    "T-SDU-PD": RETARGET_ROOT / "CAREPDretarget_rest" / "06_T-SDU-PD",
}
PKLS = {
    "3DGait": CAREPD / "3DGait.pkl",
    "BMCLab": CAREPD / "BMCLab.pkl",
    "PD-GaM": CAREPD / "PD-GaM.pkl",
    "T-SDU-PD": CAREPD / "T-SDU-PD.pkl",
}
MODEL_XML = Path(os.environ.get("BIOGAIT_MODEL_XML", ROOT / "data" / "models" / "MS-Human-700_skeleton.xml"))
OUT_DIR = ROOT / "outputs" / "carepd_paper_baseline"
LABELS = [0, 1, 2]


def load_trial_labels(dataset: str) -> dict[tuple[str, str], dict]:
    with PKLS[dataset].open("rb") as f:
        obj = pickle.load(f)
    labels = {}
    for subject, subj_data in obj.items():
        if not isinstance(subj_data, dict):
            continue
        for trial, trial_data in subj_data.items():
            if isinstance(trial_data, dict):
                labels[(str(subject), str(trial))] = {
                    "updrs_gait_raw": trial_data.get("UPDRS_GAIT"),
                    "medication": trial_data.get("medication"),
                    "other": trial_data.get("other"),
                }
    return labels


def map_label(raw) -> int | None:
    if raw is None:
        return None
    raw_i = int(raw)
    return min(raw_i, 2)


def build_manifest() -> pd.DataFrame:
    rows = []
    for dataset, root in DATASETS.items():
        label_map = load_trial_labels(dataset)
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            meta_path = d / "meta.json"
            npz_path = d / "ms700_retarget.npz"
            if not meta_path.exists() or not npz_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            subject = str(meta["subject"])
            trial = str(meta["trial"])
            lab = label_map.get((subject, trial), {})
            y = map_label(lab.get("updrs_gait_raw"))
            if y is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "sample_id": d.name,
                    "subject": subject,
                    "group_id": f"{dataset}::{subject}",
                    "subject_label_id": f"{dataset}::{subject}::label{y}",
                    "trial": trial,
                    "retarget_npz": str(npz_path),
                    "updrs_gait_raw": int(lab["updrs_gait_raw"]),
                    "gait_label": int(y),
                    "medication": "" if lab.get("medication") is None else str(lab.get("medication")),
                    "other": "" if lab.get("other") is None else str(lab.get("other")),
                    "frames": int(meta.get("frames", 0)),
                    "fps": float(meta.get("fps", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def extract_features(manifest: pd.DataFrame) -> pd.DataFrame:
    first_qpos, _ = load_motion(Path(manifest.iloc[0]["retarget_npz"]))
    qpos_names, mapping_df = build_qpos_mapping(MODEL_XML, int(first_qpos.shape[1]))
    mapping_df.to_csv(OUT_DIR / "carepd_joint_mapping_qpos.csv", index=False, encoding="utf-8-sig")

    rows = []
    errors = []
    for i, r in manifest.iterrows():
        try:
            npz_path = Path(str(r["retarget_npz"]))
            qpos, fps = load_motion(npz_path)
            d = np.load(npz_path, allow_pickle=True)
            qvel = np.asarray(d["qvel"], dtype=np.float64) if "qvel" in d.files else None
            feats = {}
            feats.update(extract_joint_features(qpos, fps, qpos_names))
            feats.update(extract_gait_proxy_features(qpos, fps, qpos_names))
            feats.update(extract_qvel_features(qvel, fps, qpos_names))
            feats.update(extract_segment_features(qpos, fps, qpos_names))
            rows.append({**feats, **r.to_dict()})
            if (i + 1) % 250 == 0:
                print(f"[carepd-features] {i + 1}/{len(manifest)}")
        except Exception as e:
            err = r.to_dict()
            err["error"] = repr(e)
            errors.append(err)
    pd.DataFrame(errors).to_csv(OUT_DIR / "carepd_feature_errors.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}acceptable_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        f"{prefix}weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def patient_aggregate(df_pred: pd.DataFrame, group_col: str = "subject_label_id") -> pd.DataFrame:
    rows = []
    for gid, g in df_pred.groupby(group_col):
        proba = g[[f"proba_{c}" for c in LABELS]].to_numpy(dtype=np.float64).mean(axis=0)
        y_values = sorted(g["gait_label"].unique().tolist())
        rows.append(
            {
                "group_id": gid,
                "dataset": g["dataset"].iloc[0],
                "subject": g["subject"].iloc[0],
                "gait_label": int(y_values[0]) if len(y_values) == 1 else -1,
                "pred": int(np.argmax(proba)),
                **{f"proba_{c}": float(proba[c]) for c in LABELS},
                "n_segments": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def feature_cols(df: pd.DataFrame) -> list[str]:
    meta = {
        "dataset",
        "sample_id",
        "subject",
        "group_id",
        "subject_label_id",
        "trial",
        "retarget_npz",
        "updrs_gait_raw",
        "gait_label",
        "medication",
        "other",
        "frames",
        "fps",
    }
    return [c for c in df.columns if c not in meta]


def make_models(seed: int = 42) -> list[tuple[str, Pipeline]]:
    models: list[tuple[str, Pipeline]] = []

    def pipe(clf, k="all", scale=False):
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("select", SelectKBest(f_classif, k=k)),
        ]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", clf))
        return Pipeline(steps)

    for k in [160, 320, "all"]:
        models.append(
            (
                f"ExtraTrees_k{k}",
                pipe(
                    ExtraTreesClassifier(
                        n_estimators=500,
                        random_state=seed,
                        class_weight="balanced_subsample",
                        max_features="sqrt",
                        min_samples_leaf=1,
                        n_jobs=-1,
                    ),
                    k=k,
                ),
            )
        )
    for k in [160, 320]:
        models.append(
            (
                f"RandomForest_k{k}",
                pipe(
                    RandomForestClassifier(
                        n_estimators=500,
                        random_state=seed,
                        class_weight="balanced_subsample",
                        max_features="sqrt",
                        min_samples_leaf=2,
                        n_jobs=-1,
                    ),
                    k=k,
                ),
            )
        )
    models.append(
        (
            "LogReg_k320",
            pipe(LogisticRegression(C=0.5, max_iter=10000, class_weight="balanced", random_state=seed), k=320, scale=True),
        )
    )
    if ExplainableBoostingClassifier is not None:
        models.append(
            (
                "EBM_top150_additive",
                pipe(
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
                    k=150,
                ),
            )
        )
    return models


def proba_or_onehot(model, X):
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=np.float64)
        if proba.shape[1] == len(LABELS):
            return proba
    pred = np.asarray(model.predict(X)).reshape(-1)
    proba = np.zeros((len(pred), len(LABELS)), dtype=np.float64)
    proba[np.arange(len(pred)), pred] = 1.0
    return proba


def run_grouped_cv(df: pd.DataFrame, subset_name: str, seed: int = 42) -> list[dict]:
    subset_dir = OUT_DIR / "grouped_cv" / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)
    cols = feature_cols(df)
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["gait_label"].to_numpy(dtype=int)
    groups = df["group_id"].to_numpy()
    n_splits = min(5, len(np.unique(groups)), int(pd.Series(y).value_counts().min()))
    if n_splits < 2:
        return []

    results = []
    for model_name, model in make_models(seed):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        proba = np.zeros((len(df), len(LABELS)), dtype=np.float64)
        pred = np.zeros(len(df), dtype=int)
        for tr, te in cv.split(X, y, groups):
            m = clone(model)
            m.fit(X[tr], y[tr])
            proba[te] = proba_or_onehot(m, X[te])
            pred[te] = np.argmax(proba[te], axis=1)

        pred_df = df[["dataset", "sample_id", "subject", "group_id", "subject_label_id", "gait_label"]].copy()
        pred_df["pred"] = pred
        for c in LABELS:
            pred_df[f"proba_{c}"] = proba[:, c]
        patient_df = patient_aggregate(pred_df)
        patient_df = patient_df[patient_df["gait_label"] >= 0].copy()
        row = {
            "scope": subset_name,
            "model": model_name,
            "n_segments": int(len(df)),
            "n_subjects": int(df["group_id"].nunique()),
            "n_subject_label_units": int(patient_df["group_id"].nunique()),
            "n_splits": int(n_splits),
            **metric_dict(y, pred, "segment_"),
            **metric_dict(patient_df["gait_label"].to_numpy(int), patient_df["pred"].to_numpy(int), "subject_label_"),
        }
        results.append(row)
        pred_df.to_csv(subset_dir / f"{model_name}_segment_oof.csv", index=False, encoding="utf-8-sig")
        patient_df.to_csv(subset_dir / f"{model_name}_subject_label_oof.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(confusion_matrix(patient_df["gait_label"], patient_df["pred"], labels=LABELS), index=[f"true_{c}" for c in LABELS], columns=[f"pred_{c}" for c in LABELS]).to_csv(
            subset_dir / f"{model_name}_subject_label_confusion.csv", encoding="utf-8-sig"
        )
        print(f"[carepd-cv] {subset_name} {model_name}: subject_label_macro_f1={row['subject_label_macro_f1']:.4f}")
    return results


def fit_eval_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, model_name: str, model: Pipeline) -> dict:
    cols = feature_cols(train_df)
    X_train = train_df[cols].to_numpy(dtype=np.float64)
    y_train = train_df["gait_label"].to_numpy(dtype=int)
    X_test = test_df[cols].to_numpy(dtype=np.float64)
    y_test = test_df["gait_label"].to_numpy(dtype=int)
    m = clone(model)
    m.fit(X_train, y_train)
    proba = proba_or_onehot(m, X_test)
    pred = np.argmax(proba, axis=1)
    pred_df = test_df[["dataset", "sample_id", "subject", "group_id", "subject_label_id", "gait_label"]].copy()
    pred_df["pred"] = pred
    for c in LABELS:
        pred_df[f"proba_{c}"] = proba[:, c]
    patient_df = patient_aggregate(pred_df)
    return {
        "train_scope": "+".join(sorted(train_df["dataset"].unique())),
        "test_scope": "+".join(sorted(test_df["dataset"].unique())),
        "model": model_name,
        "n_train_segments": int(len(train_df)),
        "n_test_segments": int(len(test_df)),
        **metric_dict(y_test, pred, "segment_"),
        **metric_dict(patient_df["gait_label"].to_numpy(int), patient_df["pred"].to_numpy(int), "subject_label_"),
    }


def run_leave_one_dataset_out(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rows = []
    models = make_models(seed)
    for test_dataset in sorted(df["dataset"].unique()):
        train_df = df[df["dataset"] != test_dataset].copy()
        test_df = df[df["dataset"] == test_dataset].copy()
        for model_name, model in models:
            row = fit_eval_train_test(train_df, test_df, model_name, model)
            rows.append(row)
            print(
                f"[carepd-lodo] test={test_dataset} {model_name}: "
                f"subject_label_macro_f1={row['subject_label_macro_f1']:.4f}"
            )
    return pd.DataFrame(rows)


def write_report(summary_df: pd.DataFrame, lodo_df: pd.DataFrame):
    best_cv = summary_df.sort_values("subject_label_macro_f1", ascending=False).head(12)
    best_lodo = lodo_df.sort_values(["test_scope", "subject_label_macro_f1"], ascending=[True, False]).groupby("test_scope").head(3)

    def md_table(df: pd.DataFrame) -> str:
        cols = df.columns.tolist()
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# CAREPD Paper Baseline",
        "",
        "## Label Definition",
        "",
        "`UPDRS_GAIT` is mapped to a 3-class label by `0 -> 0`, `1 -> 1`, and `2/3 -> 2`.",
        "Splits are grouped by subject. Patient-level metrics are reported on `dataset::subject::label` units because some CAREPD subjects have multiple UPDRS_GAIT labels across trials/days.",
        "",
        "## Best Grouped-CV Results",
        "",
        md_table(best_cv),
        "",
        "## Best Leave-One-Dataset-Out Results",
        "",
        md_table(best_lodo),
        "",
        "## Files",
        "",
        "- `carepd_manifest.csv`: unified trial manifest.",
        "- `carepd_qpos_features.csv`: qpos/qvel handcrafted features.",
        "- `grouped_cv_summary.csv`: within/combined CAREPD grouped-CV results.",
        "- `leave_one_dataset_out_results.csv`: cross-dataset generalization results.",
    ]
    (OUT_DIR / "CAREPD_PAPER_BASELINE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "carepd_manifest.csv"
    features_path = OUT_DIR / "carepd_qpos_features.csv"

    if features_path.exists():
        df = pd.read_csv(features_path)
    else:
        manifest = build_manifest()
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        df = extract_features(manifest)
        df.to_csv(features_path, index=False, encoding="utf-8-sig")

    # Keep the three-class mapped task only.
    df["gait_label"] = df["gait_label"].astype(int)
    dataset_counts = []
    for dataset, g in df.groupby("dataset"):
        dataset_counts.append(
            {
                "dataset": dataset,
                "n_segments": int(len(g)),
                "n_subjects": int(g["group_id"].nunique()),
                "label_counts": ";".join(f"{k}:{v}" for k, v in sorted(Counter(g["gait_label"]).items())),
                "subject_label_units": int(g["subject_label_id"].nunique()),
            }
        )
    pd.DataFrame(dataset_counts).to_csv(OUT_DIR / "carepd_three_class_dataset_counts.csv", index=False, encoding="utf-8-sig")

    all_results = []
    for scope in sorted(df["dataset"].unique()):
        all_results.extend(run_grouped_cv(df[df["dataset"] == scope].copy(), scope))
    all_results.extend(run_grouped_cv(df.copy(), "CAREPD_all4"))
    summary_df = pd.DataFrame(all_results).sort_values(["scope", "subject_label_macro_f1"], ascending=[True, False])
    summary_df.to_csv(OUT_DIR / "grouped_cv_summary.csv", index=False, encoding="utf-8-sig")

    lodo_df = run_leave_one_dataset_out(df)
    lodo_df.to_csv(OUT_DIR / "leave_one_dataset_out_results.csv", index=False, encoding="utf-8-sig")
    write_report(summary_df, lodo_df)

    best = summary_df.sort_values("subject_label_macro_f1", ascending=False).head(1).to_dict(orient="records")[0]
    print(json.dumps({"best_grouped_cv": best, "out_dir": str(OUT_DIR)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
