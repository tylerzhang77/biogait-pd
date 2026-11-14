"""Export fold-specific BioGait expert pipelines and validate their probabilities.

This script intentionally fits the same expert factories used by the revision pipeline,
then serializes the fitted sklearn/Interpret pipelines with joblib.  The historical
revision package retained role-level probability caches but not fitted estimators; the
files produced here are therefore reconstructed checkpoints until they are verified
against the archived manuscript predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_carepd_bo_da_moe import (  # noqa: E402
    make_linear_svc_expert,
    make_logreg_expert,
    make_ordinal_expert,
)
from run_carepd_fivecohort_main150_domainnorm_moe import (  # noqa: E402
    domain_robust_normalize,
    make_ebm_expert,
)
from run_carepd_hierarchical_source_role_moe import get_role_dict  # noqa: E402
from run_carepd_or_da_ebm_plus import (  # noqa: E402
    build_concept_scores,
    concept_sparse_ordinal,
)
from run_paper_lodo_v2_branch_hier_moe import (  # noqa: E402
    META_COLS,
    concept_groups_v2,
    load_v2_features,
)


FEATURE_TABLE = ROOT / "outputs" / "axis_corrected_features_v3_fpsfix" / "fivecohort_axis_corrected_common_feature_table.csv"
OUT_ROOT = ROOT / "paper" / "res" / "reconstructed_checkpoints"
CACHE_ROOT = ROOT / "paper" / "res" / "component_ablation" / "expert_probability_cache"
CAREPD_SCOPES = ["3DGait", "BMCLab", "PD-GaM", "T-SDU-PD"]
ROLES = ["nonlinear_ebm", "linear_margin_correction", "ordinal_concept"]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def safe_name(name: str) -> str:
    return name.replace("-", "_").replace("/", "_")


def load_roles(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {role: np.asarray(data[role]) for role in ROLES}


def fit_experts(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], groups: dict[str, list[str]]):
    """Fit the same six implementation-level experts used by build_experts."""
    x_train = train[cols].to_numpy(np.float64)
    y_train = train["gait_label"].to_numpy(int)
    x_test = test[cols].to_numpy(np.float64)
    factories = {
        "additive_ebm": make_ebm_expert(k=150),
        "margin_svc_c0p1": make_linear_svc_expert(C=0.1),
        "margin_svc_c1": make_linear_svc_expert(C=1.0),
        "logreg_c0p5": make_logreg_expert(C=0.5),
        "ordinal_cumulative_c0p3": make_ordinal_expert(C=0.3),
    }
    fitted = {}
    probs = {}
    for name, estimator in factories.items():
        print(f"  fitting {name} (n={len(train)})", flush=True)
        estimator.fit(x_train, y_train)
        fitted[name] = estimator
        probs[name] = estimator.predict_proba(x_test)

    concept_train = build_concept_scores(train, train, groups)
    concept_test = build_concept_scores(train, test, groups)
    concept_model = concept_sparse_ordinal(C=0.3)
    print(f"  fitting concept_sparse_ordinal (n={len(train)})", flush=True)
    concept_model.fit(concept_train, y_train)
    fitted["concept_sparse_ordinal"] = concept_model
    probs["concept_sparse_ordinal"] = concept_model.predict_proba(concept_test)
    return fitted, probs, list(concept_train.columns)


def selected_features(model, cols: list[str]) -> list[str]:
    """Recover feature names after VarianceThreshold and SelectKBest."""
    try:
        var = model.named_steps["var"]
        selector = model.named_steps["select"]
        after_var = np.asarray(cols, dtype=object)[var.get_support()]
        return after_var[selector.get_support()].astype(str).tolist()
    except Exception:
        return []


def export_one(full: pd.DataFrame, cols: list[str], groups: dict[str, list[str]], setting: str, scope: str) -> dict:
    include_main150 = setting == "five-cohort"
    raw_train = full[full["dataset"] != scope].copy()
    if not include_main150:
        raw_train = raw_train[raw_train["dataset"] != "main150"].copy()
    raw_test = full[full["dataset"] == scope].copy()
    setting_dir = OUT_ROOT / setting / scope
    setting_dir.mkdir(parents=True, exist_ok=True)

    model_entries = []
    role_diffs = {}

    def fit_and_save(label: str, train_frame: pd.DataFrame):
        train_z, test_z = domain_robust_normalize(train_frame, raw_test, cols, "domain_z")
        fitted, probs, concept_cols = fit_experts(train_z, test_z, cols, groups)
        out_dir = setting_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted, out_dir / "experts.joblib", compress=3)
        np.savez_compressed(out_dir / "role_probabilities.npz", **get_role_dict(label, probs))
        (out_dir / "feature_columns.json").write_text(json.dumps(cols, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "concept_columns.json").write_text(json.dumps(concept_cols, ensure_ascii=False, indent=2), encoding="utf-8")
        selected = {name: selected_features(model, cols) for name, model in fitted.items() if name != "concept_sparse_ordinal"}
        (out_dir / "selected_features.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

        cache_path = CACHE_ROOT / setting / scope / ("population_roles.npz" if label == "population" else f"source_{label.removeprefix('source_')}_roles.npz")
        cached = load_roles(cache_path)
        current = get_role_dict(label, probs)
        diff = {}
        if cached is not None and cached.keys() == current.keys() and all(cached[r].shape == current[r].shape for r in ROLES):
            diff = {r: float(np.max(np.abs(cached[r] - current[r]))) for r in ROLES}
            role_diffs[label] = diff
        model_entries.append({
            "label": label,
            "path": str((out_dir / "experts.joblib").relative_to(OUT_ROOT)),
            "n_train_rows": int(len(train_frame)),
            "n_test_rows": int(len(raw_test)),
            "cache_path": str(cache_path),
            "max_abs_probability_difference_vs_cache": diff,
        })

    fit_and_save("population", raw_train)
    for source in sorted(raw_train["dataset"].unique()):
        fit_and_save(f"source_{safe_name(source)}", raw_train[raw_train["dataset"] == source].copy())

    manifest = {
        "checkpoint_status": "reconstructed",
        "train_setting": setting,
        "held_out_scope": scope,
        "include_main150": include_main150,
        "feature_table": str(FEATURE_TABLE),
        "feature_table_sha256": sha256_file(FEATURE_TABLE),
        "n_features": len(cols),
        "n_train_rows": int(len(raw_train)),
        "n_test_rows": int(len(raw_test)),
        "source_datasets": sorted(raw_train["dataset"].unique()),
        "expert_factories": {
            "additive_ebm": {"k": 150, "interactions": 0, "max_bins": 128, "outer_bags": 4, "min_samples_leaf": 8, "learning_rate": 0.02, "max_rounds": 2500, "random_state": 42},
            "margin_svc_c0p1": {"C": 0.1, "k": 320, "calibration_cv": 3},
            "margin_svc_c1": {"C": 1.0, "k": 320, "calibration_cv": 3},
            "logreg_c0p5": {"C": 0.5, "k": 320},
            "ordinal_cumulative_c0p3": {"C": 0.3, "k": 320},
            "concept_sparse_ordinal": {"C": 0.3, "penalty": "l1"},
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "packages": {name: metadata.version(name) for name in ["scikit-learn", "joblib", "interpret-core"] if _has_package(name)},
        "models": model_entries,
        "role_cache_comparison": role_diffs,
    }
    (setting_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _has_package(name: str) -> bool:
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["CAREPD-only", "five-cohort"], default="CAREPD-only")
    parser.add_argument("--scope", choices=CAREPD_SCOPES, default="3DGait")
    parser.add_argument("--all", action="store_true", help="Export all CAREPD-only and five-cohort folds")
    args = parser.parse_args()
    print(f"Loading feature table: {FEATURE_TABLE}", flush=True)
    full, cols = load_v2_features()
    groups = concept_groups_v2(cols)
    if args.all:
        manifests = []
        for setting in ["CAREPD-only", "five-cohort"]:
            for scope in CAREPD_SCOPES:
                existing = OUT_ROOT / setting / scope / "checkpoint_manifest.json"
                if existing.exists():
                    print(f"Skipping existing checkpoint: {setting}/{scope}", flush=True)
                    manifests.append(json.loads(existing.read_text(encoding="utf-8")))
                else:
                    manifests.append(export_one(full, cols, groups, setting, scope))
        print(json.dumps({"n_folds": len(manifests), "out_root": str(OUT_ROOT)}, ensure_ascii=False, indent=2))
        return
    manifest = export_one(full, cols, groups, args.setting, args.scope)
    print(json.dumps({"out_dir": str(OUT_ROOT / args.setting / args.scope), "role_cache_comparison": manifest["role_cache_comparison"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
