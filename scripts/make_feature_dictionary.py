from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "outputs" / "paper" / "handcrafted_feature_statistics"
OUT_CSV = STATS / "handcrafted_feature_dictionary.csv"
OUT_MD = STATS / "HANDCRAFTED_FEATURE_DICTIONARY.md"


STAT_DESC = {
    "mean": "temporal mean of the signal within a gait segment",
    "std": "temporal standard deviation, reflecting variability within a segment",
    "rom": "range of motion, computed as the 95th percentile minus the 5th percentile",
    "vel_rms": "root-mean-square angular velocity",
    "acc_rms": "root-mean-square angular acceleration",
    "dom_freq": "dominant frequency estimated from the power spectrum",
    "spec_entropy": "normalized spectral entropy, reflecting frequency dispersion",
    "autocorr_regularity": "autocorrelation peak ratio within the locomotor frequency range",
    "disp_rms": "root-mean-square displacement relative to the pelvis coordinate frame",
    "speed_rms": "root-mean-square keypoint speed",
    "jerk_rms": "root-mean-square jerk",
    "y_rom": "vertical range of motion, 95th percentile minus 5th percentile",
    "z_rom": "anteroposterior range of motion, 95th percentile minus 5th percentile",
    "x_domfreq": "dominant frequency of mediolateral keypoint displacement",
    "y_domfreq": "dominant frequency of vertical keypoint displacement",
    "y_spec_entropy": "spectral entropy of vertical keypoint displacement",
    "y_regularity": "autocorrelation regularity of vertical keypoint displacement",
    "y_freeze_index": "high-frequency to locomotor-band energy ratio of vertical displacement",
    "path_efficiency": "end-to-end displacement divided by accumulated path length",
    "phase_plv": "phase-locking value between left and right paired signals",
    "phase_mean_abs_diff": "mean absolute phase difference between paired signals",
    "plv": "phase-locking value between left and right paired keypoints",
    "mad": "mean absolute phase difference between paired keypoints",
    "xcorr_lag_s": "time lag at maximum cross-correlation between paired keypoints",
    "y_rom_ratio": "left-right asymmetry ratio of vertical range of motion",
    "regularity": "autocorrelation-based movement regularity",
    "freeze_index": "high-frequency to locomotor-band energy ratio",
    "cv": "coefficient of variation, standard deviation divided by absolute mean",
    "p10": "10th percentile",
    "p90": "90th percentile",
    "iqr": "interquartile range",
    "max": "maximum absolute value",
    "count": "number of detected proxy events",
    "stride_dist_time_corr": "correlation between stride distance and stride time",
    "seg_mean_slope": "linear slope of subsegment means across the gait segment",
    "seg_mean_first_last_delta": "difference between the last and first subsegment means",
    "seg_mean_range": "range of subsegment means",
    "seg_std_mean": "mean of subsegment standard deviations",
    "seg_std_range": "range of subsegment standard deviations",
}


def split_double_underscore(feature: str) -> tuple[str, str] | None:
    if "__" not in feature:
        return None
    left, right = feature.split("__", 1)
    return left, right


def describe_joint_feature(feature: str) -> tuple[str, str, str]:
    left, stat = split_double_underscore(feature) or (feature, "")
    return "joint angle", left, STAT_DESC.get(stat, f"joint-angle statistic: {stat}")


def describe_keypoint_feature(feature: str) -> tuple[str, str, str]:
    body = feature
    stat = ""
    if feature.startswith("kp_sym_"):
        left, stat = split_double_underscore(feature) or (feature, "")
        return "keypoint left-right symmetry", left.removeprefix("kp_sym_"), STAT_DESC.get(stat, stat)
    if feature.startswith("kp_phase_"):
        left, stat = split_double_underscore(feature) or (feature, "")
        return "keypoint left-right phase", left.removeprefix("kp_phase_"), STAT_DESC.get(stat, stat)
    if feature.startswith("kp_global_speed_"):
        stat = feature.removeprefix("kp_global_speed_")
        return "global pelvis trajectory", "pelvis horizontal speed", STAT_DESC.get(stat, f"global speed statistic: {stat}")
    if feature.startswith("kp_center_sway_"):
        stat = feature.removeprefix("kp_center_sway_")
        return "whole-body keypoint centroid", "keypoint centroid sway", STAT_DESC.get(stat, f"centroid sway statistic: {stat}")
    if feature.startswith("kp_head_rel_"):
        stat = feature.removeprefix("kp_head_rel_")
        return "head-trunk relative motion", "head relative to pelvis", STAT_DESC.get(stat, f"head-pelvis statistic: {stat}")
    left, stat = split_double_underscore(feature) or (feature, "")
    body = left.removeprefix("kp_")
    return "keypoint trajectory", body, STAT_DESC.get(stat, f"keypoint statistic: {stat}")


def describe_event_feature(feature: str) -> tuple[str, str, str]:
    if feature.startswith("evt_lr_"):
        rest = feature.removeprefix("evt_lr_")
        for prefix in ["step_time", "stride_time_asym", "nearest_offset"]:
            if rest.startswith(prefix + "_"):
                stat = rest.removeprefix(prefix + "_")
                target = prefix
                return "foot-event proxy", f"left-right {target}", STAT_DESC.get(stat, f"{target} statistic: {stat}")
        return "foot-event proxy", "left-right event timing", rest
    side = "left" if feature.startswith("evt_l_") else "right" if feature.startswith("evt_r_") else ""
    rest = feature[6:] if side else feature.removeprefix("evt_")
    if rest == "count":
        return "foot-event proxy", f"{side} foot event count", "number of detected foot-height minima used as gait-event proxies"
    for prefix in ["stride_time", "cycle_y_rom", "cycle_z_rom", "stride_dist", "foot_speed", "foot_y"]:
        if rest.startswith(prefix + "_"):
            stat = rest.removeprefix(prefix + "_")
            target = f"{side} {prefix}".strip()
            return "foot-event proxy", target, STAT_DESC.get(stat, f"{target} statistic: {stat}")
    return "foot-event proxy", f"{side} foot event feature".strip(), rest


def describe_qvel_feature(feature: str) -> tuple[str, str, str]:
    if feature.startswith("qvel_root_speed_"):
        stat = feature.removeprefix("qvel_root_speed_")
        return "qvel/global dynamics", "root horizontal speed from qvel", STAT_DESC.get(stat, f"root speed statistic: {stat}")
    if feature.startswith("qvel_joint_abs_global_"):
        stat = feature.removeprefix("qvel_joint_abs_global_")
        return "qvel/global dynamics", "absolute joint velocity distribution", STAT_DESC.get(stat, f"joint velocity statistic: {stat}")
    if feature.startswith("qvel_joint_frame_energy_"):
        stat = feature.removeprefix("qvel_joint_frame_energy_")
        return "qvel/global dynamics", "per-frame joint velocity energy", STAT_DESC.get(stat, f"joint energy statistic: {stat}")
    if feature.startswith("qvel_joint_abs_"):
        stat = feature.removeprefix("qvel_joint_abs_")
        return "qvel/global dynamics", "absolute joint velocity distribution", STAT_DESC.get(stat, f"joint velocity statistic: {stat}")
    if feature.startswith("qvel_joint_energy_"):
        stat = feature.removeprefix("qvel_joint_energy_")
        return "qvel/global dynamics", "per-frame joint velocity energy", STAT_DESC.get(stat, f"joint energy statistic: {stat}")
    return "qvel/global dynamics", "qvel signal", feature


def describe_segment_feature(feature: str) -> tuple[str, str, str]:
    body = feature.removeprefix("seg_")
    suffixes = [
        "seg_mean_first_last_delta",
        "seg_mean_slope",
        "seg_mean_range",
        "seg_std_mean",
        "seg_std_range",
    ]
    for suffix in suffixes:
        if body.endswith("_" + suffix):
            signal = body[: -len("_" + suffix)]
            return "within-segment trend", signal, STAT_DESC[suffix]
    return "within-segment trend", body, "trend or variability statistic across temporal subsegments"


def describe_gait_proxy(feature: str) -> tuple[str, str, str]:
    mapping = {
        "gait_speed_mean": ("global gait proxy", "root horizontal speed", "mean walking speed within the segment"),
        "gait_speed_std": ("global gait proxy", "root horizontal speed", "standard deviation of walking speed"),
        "gait_speed_p5_p95_ratio": ("global gait proxy", "root horizontal speed", "95th to 5th percentile speed ratio"),
        "vertical_excursion": ("global gait proxy", "root vertical motion", "vertical displacement range of the root/pelvis"),
        "root_jerk_rms": ("global gait proxy", "root acceleration", "RMS of root acceleration used as a jerk-like smoothness proxy"),
        "turn_total_abs_deg": ("turning/postural proxy", "root yaw", "total absolute yaw rotation in degrees"),
        "turn_net_deg": ("turning/postural proxy", "root yaw", "net yaw rotation in degrees"),
        "turn_rate_std": ("turning/postural proxy", "root yaw rate", "standard deviation of yaw rate"),
        "stride_freq_hz": ("gait rhythm proxy", "knee angle rhythm", "average dominant knee-angle frequency"),
        "cadence_step_per_min": ("gait rhythm proxy", "knee angle rhythm", "stride frequency converted to steps per minute"),
        "knee_phase_plv": ("left-right symmetry", "left and right knee angles", "phase-locking value between knee signals"),
        "knee_phase_mean_abs_diff": ("left-right symmetry", "left and right knee angles", "mean absolute knee phase difference"),
        "knee_rom_symmetry": ("left-right symmetry", "left and right knee ROM", "left-right asymmetry ratio of knee ROM"),
        "knee_regularity_mean": ("rhythm regularity", "left and right knee angles", "mean autocorrelation regularity of knee signals"),
        "freeze_index_knee_mean": ("freezing proxy", "left and right knee angles", "mean high-frequency to locomotor-band energy ratio"),
        "heel_strike_proxy_count_r": ("foot-event proxy", "right knee angle minima", "number of right heel-strike proxy events detected from knee angle minima"),
        "heel_strike_proxy_count_l": ("foot-event proxy", "left knee angle minima", "number of left heel-strike proxy events detected from knee angle minima"),
        "toe_clearance_proxy_r": ("toe-clearance proxy", "right ankle + MTP angles", "right toe-clearance proxy ROM"),
        "toe_clearance_proxy_l": ("toe-clearance proxy", "left ankle + MTP angles", "left toe-clearance proxy ROM"),
        "toe_clearance_proxy_symmetry": ("left-right symmetry", "left and right toe-clearance proxies", "left-right asymmetry ratio of toe-clearance proxy"),
        "arm_swing_rom_r": ("upper-limb swing", "right shoulder elevation", "right shoulder elevation ROM as an arm-swing proxy"),
        "arm_swing_rom_l": ("upper-limb swing", "left shoulder elevation", "left shoulder elevation ROM as an arm-swing proxy"),
        "arm_swing_symmetry": ("upper-limb swing", "left and right shoulder elevation", "left-right asymmetry ratio of arm-swing ROM"),
    }
    return mapping.get(feature, ("miscellaneous proxy", feature, "derived gait proxy feature"))


def describe(feature: str) -> tuple[str, str, str]:
    if feature.startswith("kp_"):
        return describe_keypoint_feature(feature)
    if feature.startswith("evt_"):
        return describe_event_feature(feature)
    if feature.startswith("qvel_"):
        return describe_qvel_feature(feature)
    if feature.startswith("seg_"):
        return describe_segment_feature(feature)
    if feature in {
        "gait_speed_mean",
        "gait_speed_std",
        "gait_speed_p5_p95_ratio",
        "vertical_excursion",
        "root_jerk_rms",
        "turn_total_abs_deg",
        "turn_net_deg",
        "turn_rate_std",
        "stride_freq_hz",
        "cadence_step_per_min",
        "knee_phase_plv",
        "knee_phase_mean_abs_diff",
        "knee_rom_symmetry",
        "knee_regularity_mean",
        "freeze_index_knee_mean",
        "heel_strike_proxy_count_r",
        "heel_strike_proxy_count_l",
        "toe_clearance_proxy_r",
        "toe_clearance_proxy_l",
        "toe_clearance_proxy_symmetry",
        "arm_swing_rom_r",
        "arm_swing_rom_l",
        "arm_swing_symmetry",
    }:
        return describe_gait_proxy(feature)
    if "__" in feature:
        return describe_joint_feature(feature)
    return "miscellaneous", feature, "derived handcrafted feature"


def family_order(family: str) -> int:
    order = {
        "joint angle": 0,
        "global gait proxy": 1,
        "gait rhythm proxy": 2,
        "left-right symmetry": 3,
        "foot-event proxy": 4,
        "toe-clearance proxy": 5,
        "upper-limb swing": 6,
        "qvel/global dynamics": 7,
        "within-segment trend": 8,
        "keypoint trajectory": 9,
        "keypoint left-right symmetry": 10,
        "keypoint left-right phase": 11,
        "global pelvis trajectory": 12,
        "whole-body keypoint centroid": 13,
        "head-trunk relative motion": 14,
        "turning/postural proxy": 15,
        "rhythm regularity": 16,
        "freezing proxy": 17,
    }
    return order.get(family, 99)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4g}")
            else:
                vals.append(str(val).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    stats = pd.read_csv(STATS / "handcrafted_feature_stats_unit_level.csv")
    rows = []
    for _, row in stats.iterrows():
        feature = str(row["feature"])
        family, signal, explanation = describe(feature)
        rows.append(
            {
                "feature": feature,
                "feature_family": family,
                "biomechanical_concept": row["concept_label"],
                "source_signal_or_body_part": signal,
                "computed_quantity": explanation,
                "kw_fdr_q": row["kruskal_q"],
                "spearman_fdr_q": row["spearman_q"],
                "effect_size_epsilon_squared": row["epsilon_squared"],
                "spearman_rho": row["spearman_rho"],
            }
        )
    out = pd.DataFrame(rows)
    out["_order"] = out["feature_family"].map(family_order)
    out = out.sort_values(["_order", "biomechanical_concept", "feature"]).drop(columns=["_order"])
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    summary = (
        out.groupby(["feature_family", "biomechanical_concept"])
        .agg(
            n_features=("feature", "count"),
            n_kw_fdr_005=("kw_fdr_q", lambda s: int((s < 0.05).sum())),
            n_trend_fdr_005=("spearman_fdr_q", lambda s: int((s < 0.05).sum())),
            mean_effect_size=("effect_size_epsilon_squared", "mean"),
        )
        .reset_index()
        .sort_values(["feature_family", "n_features"], ascending=[True, False])
    )
    family_summary = (
        out.groupby("feature_family")
        .agg(
            n_features=("feature", "count"),
            n_kw_fdr_005=("kw_fdr_q", lambda s: int((s < 0.05).sum())),
            n_trend_fdr_005=("spearman_fdr_q", lambda s: int((s < 0.05).sum())),
        )
        .reset_index()
        .sort_values("n_features", ascending=False)
    )

    md = [
        "# Handcrafted Feature Dictionary",
        "",
        f"Total features: **{len(out)}**.",
        "",
        "Each feature is computed from one gait segment after GMR retargeting. The main inputs are joint-angle time series, qvel, pelvis-centered/world keypoint trajectories, and foot-event proxy signals.",
        "",
        "## Feature Family Summary",
        "",
        md_table(family_summary),
        "",
        "## Feature Family x Biomechanical Concept Summary",
        "",
        md_table(summary),
        "",
        "## Full Dictionary",
        "",
        "The complete per-feature dictionary is saved as `handcrafted_feature_dictionary.csv`.",
        "",
        md_table(out, max_rows=80),
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()
