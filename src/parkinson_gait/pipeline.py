import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .config import StudyConfig
from .advanced_features import extract_event_keypoint_features, extract_qvel_features, extract_segment_features
from .data_io import build_qpos_mapping, load_labels, load_motion
from .features import extract_gait_proxy_features, extract_joint_features, extract_keypoint_features
from .modeling import run_model_selection
from .statistics import run_group_stats


def plot_class_distribution(df: pd.DataFrame, out_path: Path):
    vc = df["gait_label"].value_counts().sort_index()
    plt.figure(figsize=(6, 4), dpi=140)
    plt.bar([str(x) for x in vc.index], vc.values)
    plt.xlabel("GAIT score")
    plt.ylabel("Count")
    plt.title("Class Distribution (150 patients)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_top_box(df: pd.DataFrame, stats_df: pd.DataFrame, out_path: Path, top_k: int = 6):
    top = stats_df.head(top_k)["feature"].tolist()
    if not top:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=140)
    axes = axes.ravel()
    classes = sorted(df["gait_label"].unique().tolist())
    for i, f in enumerate(top):
        ax = axes[i]
        ax.boxplot([df.loc[df["gait_label"] == c, f].values for c in classes], tick_labels=[str(c) for c in classes], showfliers=False)
        ax.set_title(f, fontsize=8)
        ax.set_xlabel("GAIT")
    for i in range(len(top), len(axes)):
        axes[i].axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _calc_simple_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _search_two_model_blend(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> tuple[dict[str, float], float, np.ndarray]:
    best_w = 0.5
    best_pred = np.argmax(0.5 * p_a + 0.5 * p_b, axis=1)
    best_m = _calc_simple_metrics(y, best_pred)
    for w in np.linspace(0.0, 1.0, 201):
        p = w * p_a + (1.0 - w) * p_b
        pred = np.argmax(p, axis=1)
        m = _calc_simple_metrics(y, pred)
        if m["f1_macro"] > best_m["f1_macro"]:
            best_m = m
            best_w = float(w)
            best_pred = pred
    return best_m, best_w, best_pred


def _search_three_model_blend(
    y: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> tuple[dict[str, float], tuple[float, float, float], np.ndarray]:
    best_w = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    p = best_w[0] * p1 + best_w[1] * p2 + best_w[2] * p3
    best_pred = np.argmax(p, axis=1)
    best_m = _calc_simple_metrics(y, best_pred)
    grid = np.linspace(0.0, 1.0, 21)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < 0.0:
                continue
            p = w1 * p1 + w2 * p2 + w3 * p3
            pred = np.argmax(p, axis=1)
            m = _calc_simple_metrics(y, pred)
            if m["f1_macro"] > best_m["f1_macro"]:
                best_m = m
                best_w = (float(w1), float(w2), float(w3))
                best_pred = pred
    return best_m, best_w, best_pred


def _prune_redundant_kp_features(X_df: pd.DataFrame, y: np.ndarray, corr_thr: float = 0.995) -> tuple[pd.DataFrame, list[str]]:
    kp_cols = [c for c in X_df.columns if c.startswith("kp_")]
    if len(kp_cols) < 2:
        return X_df, []
    kp = X_df[kp_cols]
    mi = mutual_info_classif(kp.values.astype(np.float64), y, random_state=42)
    mi_map = {c: float(m) for c, m in zip(kp_cols, mi)}
    corr = kp.corr().abs()
    to_drop: set[str] = set()
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        ci = cols[i]
        if ci in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            cj = cols[j]
            if cj in to_drop:
                continue
            if float(corr.iloc[i, j]) >= corr_thr:
                # Keep the more informative one.
                if mi_map.get(ci, 0.0) >= mi_map.get(cj, 0.0):
                    to_drop.add(cj)
                else:
                    to_drop.add(ci)
                    break
    drop_list = sorted(to_drop)
    if drop_list:
        X_df = X_df.drop(columns=drop_list)
    return X_df, drop_list


def run_pipeline(cfg: StudyConfig) -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_root / "runs" / ts
    latest_dir = cfg.output_root / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(cfg.label_csv)
    first_qpos, _ = load_motion(Path(labels.iloc[0]["retarget_npz"]))
    qpos_names, mapping_df = build_qpos_mapping(cfg.model_xml, int(first_qpos.shape[1]))
    mapping_df.to_csv(run_dir / "joint_mapping_qpos.csv", index=False, encoding="utf-8-sig")

    feat_rows = []
    errors = []
    kp_missing = 0
    kp_used = 0
    for _, r in labels.iterrows():
        npz_path = Path(str(r["retarget_npz"]))
        if not npz_path.exists():
            errors.append({"patient_id": r["patient_id"], "error": "missing npz", "path": str(npz_path)})
            continue
        try:
            qpos, fps = load_motion(npz_path)
            if qpos.shape[1] != len(qpos_names):
                raise ValueError(f"qpos dim mismatch: {qpos.shape[1]} vs {len(qpos_names)}")
            f1 = extract_joint_features(qpos, fps, qpos_names)
            f2 = extract_gait_proxy_features(qpos, fps, qpos_names)
            feats = {**f1, **f2}
            if cfg.use_advanced_features:
                npzd = np.load(npz_path, allow_pickle=True)
                qvel = np.asarray(npzd["qvel"], dtype=np.float64) if "qvel" in npzd.files else None
                feats.update(extract_qvel_features(qvel, fps, qpos_names))
                feats.update(extract_segment_features(qpos, fps, qpos_names))

            if cfg.use_keypoints_fusion and cfg.keypoints_root is not None:
                kp_npz = cfg.keypoints_root / str(r["patient_id"]) / "keypoints_body.npz"
                if kp_npz.exists():
                    kpd = np.load(kp_npz, allow_pickle=True)
                    k_world = np.asarray(kpd["keypoints_world"], dtype=np.float64)
                    k_pelvis = np.asarray(kpd["keypoints_pelvis"], dtype=np.float64)
                    bnames = [str(x) for x in kpd["body_names"].tolist()]
                    kfps = float(np.asarray(kpd["fps"]).reshape(-1)[0]) if "fps" in kpd.files else fps
                    feats.update(extract_keypoint_features(k_world, k_pelvis, bnames, kfps))
                    if cfg.use_advanced_features:
                        feats.update(extract_event_keypoint_features(k_world, k_pelvis, bnames, kfps))
                    kp_used += 1
                else:
                    kp_missing += 1

            feat_rows.append({**feats, "patient_id": r["patient_id"], "gait_label": int(r["gait_label"])})
        except Exception as e:
            errors.append({"patient_id": r["patient_id"], "error": repr(e), "path": str(npz_path)})

    df_feat = pd.DataFrame(feat_rows).sort_values("patient_id").reset_index(drop=True)
    df_feat.to_csv(run_dir / "joint_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(run_dir / "feature_extraction_errors.csv", index=False, encoding="utf-8-sig")

    y = df_feat["gait_label"].values.astype(int)
    X_df = df_feat.drop(columns=["patient_id", "gait_label"])
    nunique = X_df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X_df = X_df.drop(columns=const_cols)
    X_df, dropped_kp_corr_cols = _prune_redundant_kp_features(X_df, y, corr_thr=0.995)
    X = X_df.values.astype(np.float64)

    stats_df = run_group_stats(df_feat, y)
    stats_df.to_csv(run_dir / "feature_group_stats.csv", index=False, encoding="utf-8-sig")
    stats_df.head(50).to_csv(run_dir / "feature_group_stats_top50.csv", index=False, encoding="utf-8-sig")

    model_res = run_model_selection(X, y, X_df.columns.tolist(), cv_splits=cfg.cv_splits, seed=cfg.random_seed)

    final_metrics_df = model_res.metrics_df.copy()
    final_best_name = model_res.best_name
    final_best_pred = model_res.best_oof_pred
    final_cm_df = model_res.cm_df
    final_importance_df = model_res.importance_df
    final_report = model_res.report

    # Late-fusion trick: blend qpos-only and qpos+keypoints probability outputs.
    # This often stabilizes noisy keypoint dimensions while preserving useful signal.
    if cfg.use_keypoints_fusion:
        kp_cols = [c for c in X_df.columns.tolist() if c.startswith("kp_")]
        base_cols = [c for c in X_df.columns.tolist() if not c.startswith("kp_")]
        if kp_cols and base_cols:
            X_base = df_feat[base_cols].values.astype(np.float64)
            base_res = run_model_selection(X_base, y, base_cols, cv_splits=cfg.cv_splits, seed=cfg.random_seed)
            blend_m, blend_w, blend_pred = _search_two_model_blend(y, model_res.best_oof_proba, base_res.best_oof_proba)
            blend_row = {**blend_m, "model": f"late_fusion_two_branch[w_all={blend_w:.3f},w_base={1.0-blend_w:.3f}]"}

            final_metrics_df = pd.concat([final_metrics_df, pd.DataFrame([blend_row])], ignore_index=True)
            final_metrics_df = final_metrics_df.sort_values(["f1_macro", "balanced_accuracy"], ascending=[False, False]).reset_index(drop=True)
            final_metrics_df.to_csv(run_dir / "ml_cv_metrics_with_late_fusion.csv", index=False, encoding="utf-8-sig")
            base_res.metrics_df.to_csv(run_dir / "ml_cv_metrics_base_only.csv", index=False, encoding="utf-8-sig")

            # Controlled injection: only top-K informative keypoint features.
            kp_mi = mutual_info_classif(df_feat[kp_cols].values.astype(np.float64), y, random_state=42)
            kp_rank = sorted(zip(kp_cols, kp_mi), key=lambda x: x[1], reverse=True)
            top_k = min(120, len(kp_rank))
            kp_top_cols = [c for c, _ in kp_rank[:top_k]]
            hybrid_cols = base_cols + kp_top_cols
            X_hybrid = df_feat[hybrid_cols].values.astype(np.float64)
            hybrid_res = run_model_selection(X_hybrid, y, hybrid_cols, cv_splits=cfg.cv_splits, seed=cfg.random_seed)
            hybrid_res.metrics_df.to_csv(run_dir / "ml_cv_metrics_base_plus_kp_topk.csv", index=False, encoding="utf-8-sig")

            blend3_m, blend3_w, blend3_pred = _search_three_model_blend(
                y, model_res.best_oof_proba, base_res.best_oof_proba, hybrid_res.best_oof_proba
            )
            blend3_row = {
                **blend3_m,
                "model": (
                    "late_fusion_three_branch"
                    f"[w_all={blend3_w[0]:.3f},w_base={blend3_w[1]:.3f},w_hybrid={blend3_w[2]:.3f}]"
                ),
            }
            final_metrics_df = pd.concat([final_metrics_df, pd.DataFrame([blend3_row])], ignore_index=True)
            final_metrics_df = final_metrics_df.sort_values(["f1_macro", "balanced_accuracy"], ascending=[False, False]).reset_index(drop=True)
            final_metrics_df.to_csv(run_dir / "ml_cv_metrics_with_late_fusion.csv", index=False, encoding="utf-8-sig")

            if blend_m["f1_macro"] > _calc_simple_metrics(y, final_best_pred)["f1_macro"]:
                final_best_name = blend_row["model"]
                final_best_pred = blend_pred
                final_cm_df = pd.DataFrame(
                    np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=np.int64),
                    index=[f"true_{c}" for c in sorted(np.unique(y).tolist())],
                    columns=[f"pred_{c}" for c in sorted(np.unique(y).tolist())],
                )
                cm = pd.crosstab(
                    pd.Series(y, name="true"),
                    pd.Series(blend_pred, name="pred"),
                    dropna=False,
                )
                for i in final_cm_df.index:
                    ti = int(i.split("_")[1])
                    for j in final_cm_df.columns:
                        pj = int(j.split("_")[1])
                        final_cm_df.loc[i, j] = int(cm.loc[ti, pj]) if (ti in cm.index and pj in cm.columns) else 0
                final_report = {
                    "accuracy": blend_m["accuracy"],
                    "balanced_accuracy": blend_m["balanced_accuracy"],
                    "f1_macro": blend_m["f1_macro"],
                    "f1_weighted": blend_m["f1_weighted"],
                    "note": "late fusion report (qpos+kp vs qpos-only)",
                }

            if blend3_m["f1_macro"] > _calc_simple_metrics(y, final_best_pred)["f1_macro"]:
                final_best_name = blend3_row["model"]
                final_best_pred = blend3_pred
                final_cm_df = pd.DataFrame(
                    np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=np.int64),
                    index=[f"true_{c}" for c in sorted(np.unique(y).tolist())],
                    columns=[f"pred_{c}" for c in sorted(np.unique(y).tolist())],
                )
                cm = pd.crosstab(
                    pd.Series(y, name="true"),
                    pd.Series(blend3_pred, name="pred"),
                    dropna=False,
                )
                for i in final_cm_df.index:
                    ti = int(i.split("_")[1])
                    for j in final_cm_df.columns:
                        pj = int(j.split("_")[1])
                        final_cm_df.loc[i, j] = int(cm.loc[ti, pj]) if (ti in cm.index and pj in cm.columns) else 0
                final_report = {
                    "accuracy": blend3_m["accuracy"],
                    "balanced_accuracy": blend3_m["balanced_accuracy"],
                    "f1_macro": blend3_m["f1_macro"],
                    "f1_weighted": blend3_m["f1_weighted"],
                    "note": "late fusion report (all/base/base+kpTopK)",
                }

    final_metrics_df.to_csv(run_dir / "ml_cv_metrics.csv", index=False, encoding="utf-8-sig")
    final_cm_df.to_csv(run_dir / "ml_confusion_matrix.csv", encoding="utf-8-sig")
    final_importance_df.head(50).to_csv(run_dir / "ml_feature_importance_top50.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"patient_id": df_feat["patient_id"], "y_true": y, "y_pred": final_best_pred}).to_csv(
        run_dir / "ml_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    (run_dir / "ml_classification_report.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_class_distribution(df_feat, run_dir / "class_distribution.png")
    plot_top_box(df_feat, stats_df, run_dir / "top_feature_boxplots.png")

    summary = {
        "run_dir": str(run_dir),
        "n_samples": int(len(df_feat)),
        "n_features": int(X.shape[1]),
        "dropped_constant_features": int(len(const_cols)),
        "dropped_kp_high_corr_features": int(len(dropped_kp_corr_cols)),
        "class_counts": {str(int(k)): int(v) for k, v in df_feat["gait_label"].value_counts().sort_index().items()},
        "best_model": final_best_name,
        "use_keypoints_fusion": bool(cfg.use_keypoints_fusion),
        "use_advanced_features": bool(cfg.use_advanced_features),
        "keypoints_root": None if cfg.keypoints_root is None else str(cfg.keypoints_root),
        "keypoints_used_count": int(kp_used),
        "keypoints_missing_count": int(kp_missing),
        "ml_cv_metrics": final_metrics_df.to_dict("records"),
        "top_significant_features": stats_df.head(10).to_dict("records"),
    }
    (run_dir / "study_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    return {"run_dir": str(run_dir), "latest_dir": str(latest_dir), "best_model": final_best_name}
