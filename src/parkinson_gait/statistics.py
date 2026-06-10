import numpy as np
import pandas as pd
from scipy import stats


def fdr_bh(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return out.tolist()


def run_group_stats(df_feat: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    feature_cols = [c for c in df_feat.columns if c not in {"patient_id", "gait_label"}]
    classes = sorted(np.unique(y).tolist())
    rows = []
    for f in feature_cols:
        groups = [df_feat.loc[df_feat["gait_label"] == c, f].values for c in classes]
        groups = [g for g in groups if g.size > 0]
        if len(groups) < 2:
            continue
        concat = np.concatenate(groups)
        if np.allclose(concat, concat[0], rtol=0.0, atol=1e-12):
            continue
        h, p = stats.kruskal(*groups)
        n = sum(g.size for g in groups)
        k = len(groups)
        eps2 = (h - k + 1.0) / (n - k) if n > k else 0.0
        rows.append({"feature": f, "kruskal_h": float(h), "p_value": float(p), "effect_eps2": float(max(eps2, 0.0))})

    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    out["q_value"] = fdr_bh(out["p_value"].tolist()) if len(out) else []
    return out.sort_values(["q_value", "p_value"], ascending=[True, True]).reset_index(drop=True)
