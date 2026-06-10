from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def build_candidates(seed: int) -> list[tuple[str, Pipeline]]:
    cands: list[tuple[str, Pipeline]] = []
    for k in [80, 120, 200, 320, 560, "all"]:
        cands.append(
            (
                f"et_anova_k{k}",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("var", VarianceThreshold(threshold=0.0)),
                        ("select", SelectKBest(score_func=f_classif, k=k)),
                        (
                            "clf",
                            ExtraTreesClassifier(
                                n_estimators=800,
                                random_state=seed,
                                class_weight="balanced_subsample",
                                n_jobs=-1,
                                max_features="sqrt",
                            ),
                        ),
                    ]
                ),
            )
        )

    for k in [80, 160, 320]:
        cands.append(
            (
                f"svc_mi_k{k}",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("var", VarianceThreshold(threshold=0.0)),
                        ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
                        ("scaler", StandardScaler()),
                        ("clf", SVC(C=2.2, gamma="scale", class_weight="balanced", probability=True)),
                    ]
                ),
            )
        )

    for k in [120, 320]:
        cands.append(
            (
                f"rf_anova_k{k}",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("var", VarianceThreshold(threshold=0.0)),
                        ("select", SelectKBest(score_func=f_classif, k=k)),
                        (
                            "clf",
                            RandomForestClassifier(
                                n_estimators=700,
                                random_state=seed,
                                class_weight="balanced_subsample",
                                n_jobs=-1,
                                max_features="sqrt",
                                min_samples_leaf=2,
                            ),
                        ),
                    ]
                ),
            )
        )

    cands.append(
        (
            "logreg_k160",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("var", VarianceThreshold(threshold=0.0)),
                    ("select", SelectKBest(score_func=f_classif, k=160)),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=7000, class_weight="balanced", random_state=seed, C=0.7)),
                ]
            ),
        )
    )
    return cands


@dataclass
class ModelResult:
    metrics_df: pd.DataFrame
    best_name: str
    best_oof_pred: np.ndarray
    best_oof_proba: np.ndarray | None
    importance_df: pd.DataFrame
    cm_df: pd.DataFrame
    report: dict


def _predict_proba_or_vote(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == n_classes:
            return proba
    pred = model.predict(x)
    out = np.zeros((pred.shape[0], n_classes), dtype=np.float64)
    out[np.arange(pred.shape[0]), pred] = 1.0
    return out


def _blend_probs(prob_list: list[np.ndarray], y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple random-search weighted blending on OOF probabilities.
    """
    rng = np.random.default_rng(seed)
    m = len(prob_list)
    if m == 1:
        w = np.array([1.0], dtype=np.float64)
        return prob_list[0], w
    best_w = np.ones(m, dtype=np.float64) / m
    best_pred = np.argmax(np.mean(np.stack(prob_list, axis=0), axis=0), axis=1)
    best_score = f1_score(y, best_pred, average="macro")
    for _ in range(3000):
        w = rng.random(m)
        w = w / np.sum(w)
        p = np.zeros_like(prob_list[0])
        for i in range(m):
            p += w[i] * prob_list[i]
        pred = np.argmax(p, axis=1)
        sc = f1_score(y, pred, average="macro")
        if sc > best_score:
            best_score = sc
            best_w = w
    p_best = np.zeros_like(prob_list[0])
    for i in range(m):
        p_best += best_w[i] * prob_list[i]
    return p_best, best_w


def _optimize_class_bias(proba: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Optimize per-class multiplicative bias on probabilities for macro-F1.
    """
    rng = np.random.default_rng(seed)
    n_classes = proba.shape[1]
    best_b = np.ones(n_classes, dtype=np.float64)
    best_pred = np.argmax(proba, axis=1)
    best_score = f1_score(y, best_pred, average="macro")
    for _ in range(2500):
        b = np.exp(rng.normal(loc=0.0, scale=0.35, size=n_classes))
        p = proba * b.reshape(1, -1)
        pred = np.argmax(p, axis=1)
        sc = f1_score(y, pred, average="macro")
        if sc > best_score:
            best_score = sc
            best_b = b
    p_best = proba * best_b.reshape(1, -1)
    return p_best, best_b


def run_model_selection(X: np.ndarray, y: np.ndarray, feature_names: list[str], cv_splits: int, seed: int) -> ModelResult:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    candidates = build_candidates(seed)
    n_classes = int(np.max(y) + 1)

    rows = []
    oof_map = {}
    oof_proba_map = {}
    fitted_map = {}

    for name, est in candidates:
        pred = np.zeros_like(y)
        proba = np.zeros((y.shape[0], n_classes), dtype=np.float64)
        for tr, te in skf.split(X, y):
            m = clone(est)
            m.fit(X[tr], y[tr])
            fold_pred = m.predict(X[te])
            pred[te] = fold_pred
            proba[te] = _predict_proba_or_vote(m, X[te], n_classes=n_classes)
        m = calc_metrics(y, pred)
        m["model"] = name
        rows.append(m)
        oof_map[name] = pred
        oof_proba_map[name] = proba
        full = clone(est)
        full.fit(X, y)
        fitted_map[name] = full

        # Bias-optimized variant (often helps imbalanced macro-F1)
        proba_bias, bias_vec = _optimize_class_bias(proba, y, seed=seed)
        pred_bias = np.argmax(proba_bias, axis=1)
        m_bias = calc_metrics(y, pred_bias)
        m_bias["model"] = f"{name}_bias"
        rows.append(m_bias)
        oof_map[m_bias["model"]] = pred_bias
        oof_proba_map[m_bias["model"]] = proba_bias

    metrics_df = pd.DataFrame(rows).sort_values(["f1_macro", "balanced_accuracy"], ascending=[False, False]).reset_index(drop=True)
    best_name = str(metrics_df.iloc[0]["model"])
    best_pred = oof_map[best_name]
    best_proba = oof_proba_map[best_name]

    # Stacking trick: use top model OOF probabilities as meta-features.
    # This can capture complementary decision boundaries better than pure weighted blending.
    base_rank = metrics_df["model"].tolist()
    top_stack = base_rank[: min(8, len(base_rank))]
    if len(top_stack) >= 3:
        X_meta = np.concatenate([oof_proba_map[n] for n in top_stack], axis=1)
        for c in [0.6, 1.0, 2.0]:
            meta_name = f"stack_lr_c{c}"
            pred_meta = np.zeros_like(y)
            proba_meta = np.zeros((y.shape[0], n_classes), dtype=np.float64)
            for tr, te in skf.split(X_meta, y):
                meta = LogisticRegression(
                    max_iter=8000,
                    class_weight="balanced",
                    random_state=seed,
                    C=float(c),
                )
                meta.fit(X_meta[tr], y[tr])
                pred_meta[te] = meta.predict(X_meta[te])
                proba_meta[te] = _predict_proba_or_vote(meta, X_meta[te], n_classes=n_classes)
            mm = calc_metrics(y, pred_meta)
            mm["model"] = meta_name
            rows.append(mm)
            oof_map[meta_name] = pred_meta
            oof_proba_map[meta_name] = proba_meta

            # Bias-optimized stacking variant.
            proba_meta_bias, _ = _optimize_class_bias(proba_meta, y, seed=seed)
            pred_meta_bias = np.argmax(proba_meta_bias, axis=1)
            mm_bias = calc_metrics(y, pred_meta_bias)
            mm_bias["model"] = f"{meta_name}_bias"
            rows.append(mm_bias)
            oof_map[mm_bias["model"]] = pred_meta_bias
            oof_proba_map[mm_bias["model"]] = proba_meta_bias

        metrics_df = pd.DataFrame(rows).sort_values(["f1_macro", "balanced_accuracy"], ascending=[False, False]).reset_index(drop=True)
        best_name = str(metrics_df.iloc[0]["model"])
        best_pred = oof_map[best_name]
        best_proba = oof_proba_map[best_name]

    # Ensemble trick: blend top-N probability models. Top-5 was the best
    # performer in the advanced-feature experiment.
    blend_payloads = {}
    max_blend_n = min(5, len(metrics_df))
    top_for_importance = metrics_df.head(max_blend_n)["model"].tolist()
    for blend_n in range(2, max_blend_n + 1):
        topn = metrics_df.head(blend_n)["model"].tolist()
        prob_list = [oof_proba_map[n] for n in topn]
        blend_proba, blend_w = _blend_probs(prob_list, y, seed=seed + blend_n)
        blend_pred = np.argmax(blend_proba, axis=1)
        blend_metrics = calc_metrics(y, blend_pred)
        blend_name = f"blend_top{blend_n}"
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{**blend_metrics, "model": blend_name}])], ignore_index=True)
        blend_payloads[blend_name] = (topn, blend_w, None, blend_pred, blend_proba)

        blend_bias_proba, blend_bias_vec = _optimize_class_bias(blend_proba, y, seed=seed + 100 + blend_n)
        blend_bias_pred = np.argmax(blend_bias_proba, axis=1)
        blend_bias_metrics = calc_metrics(y, blend_bias_pred)
        blend_bias_name = f"blend_top{blend_n}_bias"
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{**blend_bias_metrics, "model": blend_bias_name}])], ignore_index=True)
        blend_payloads[blend_bias_name] = (topn, blend_w, blend_bias_vec, blend_bias_pred, blend_bias_proba)

    metrics_df = metrics_df.sort_values(["f1_macro", "balanced_accuracy"], ascending=[False, False]).reset_index(drop=True)
    top_metric_name = str(metrics_df.iloc[0]["model"])
    if top_metric_name in blend_payloads:
        topn, blend_w, blend_bias_vec, blend_pred, blend_proba = blend_payloads[top_metric_name]
        if blend_bias_vec is None:
            best_name = f"{top_metric_name}[{','.join(topn)}]|w={np.round(blend_w,4).tolist()}"
        else:
            best_name = (
                f"{top_metric_name}[{','.join(topn)}]|w={np.round(blend_w,4).tolist()}|"
                f"bias={np.round(blend_bias_vec,4).tolist()}"
            )
        best_pred = blend_pred
        best_proba = blend_proba

    # Select a fitted base model for feature-importance reporting.
    best_model_key = str(metrics_df.iloc[0]["model"])
    if best_model_key in fitted_map:
        importance_model = fitted_map[best_model_key]
    elif best_model_key.endswith("_bias") and best_model_key[:-5] in fitted_map:
        importance_model = fitted_map[best_model_key[:-5]]
    else:
        # blend/stack models have no single fitted estimator; use the strongest
        # available fitted base model for feature-importance reporting.
        base_key = None
        for candidate_key in top_for_importance:
            ck = candidate_key[:-5] if candidate_key.endswith("_bias") else candidate_key
            if ck in fitted_map:
                base_key = ck
                break
        if base_key is None:
            base_key = next(iter(fitted_map))
        importance_model = fitted_map[base_key]

    cm = confusion_matrix(y, best_pred, labels=sorted(np.unique(y).tolist()))
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in sorted(np.unique(y).tolist())], columns=[f"pred_{c}" for c in sorted(np.unique(y).tolist())])

    selected = importance_model.named_steps["select"].get_support()
    selected_names = np.array(feature_names)[selected]
    clf = importance_model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=np.float64)
    elif hasattr(clf, "coef_"):
        imp = np.mean(np.abs(np.asarray(clf.coef_, dtype=np.float64)), axis=0)
    else:
        scores = getattr(importance_model.named_steps["select"], "scores_", None)
        if scores is not None:
            imp = np.nan_to_num(np.asarray(scores, dtype=np.float64)[selected], nan=0.0, posinf=0.0, neginf=0.0)
        else:
            imp = np.zeros(selected_names.shape[0], dtype=np.float64)
    importance_df = pd.DataFrame({"feature": selected_names, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)

    report = classification_report(y, best_pred, output_dict=True)
    return ModelResult(
        metrics_df=metrics_df,
        best_name=best_name,
        best_oof_pred=best_pred,
        best_oof_proba=best_proba,
        importance_df=importance_df,
        cm_df=cm_df,
        report=report,
    )
