from __future__ import annotations

import numpy as np
from scipy import signal

from .features import autocorr_peak_ratio, freeze_index, get_joint_signal, rms, spectral_features


def _safe_stats(prefix: str, x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_cv": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_iqr": 0.0,
        }
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return {
        f"{prefix}_mean": mu,
        f"{prefix}_std": sd,
        f"{prefix}_cv": float(sd / (abs(mu) + 1e-6)),
        f"{prefix}_p10": float(np.percentile(x, 10)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
        f"{prefix}_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
    }


def _detect_minima(x: np.ndarray, fps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < int(fps * 2):
        return np.array([], dtype=np.int64)
    y = signal.savgol_filter(x, window_length=min(x.size // 2 * 2 - 1, 21), polyorder=2) if x.size >= 21 else x
    prominence = max(float(np.std(y) * 0.15), 1e-6)
    peaks, _ = signal.find_peaks(-y, distance=max(1, int(0.35 * fps)), prominence=prominence)
    return peaks.astype(np.int64)


def _cycle_values(sig: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = []
    lengths = []
    for a, b in zip(events[:-1], events[1:]):
        if b <= a + 2:
            continue
        seg = sig[a:b]
        vals.append(float(np.percentile(seg, 95) - np.percentile(seg, 5)))
        lengths.append(b - a)
    return np.asarray(vals, dtype=np.float64), np.asarray(lengths, dtype=np.float64)


def _cycle_principal_rom(pos_xy: np.ndarray, events: np.ndarray) -> np.ndarray:
    vals = []
    pos_xy = np.asarray(pos_xy, dtype=np.float64)
    for a, b in zip(events[:-1], events[1:]):
        if b <= a + 2:
            continue
        seg = pos_xy[a:b]
        centered = seg - np.mean(seg, axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            axis = vt[0]
            proj = centered @ axis
        except Exception:
            proj = np.linalg.norm(centered, axis=1)
        vals.append(float(np.percentile(proj, 95) - np.percentile(proj, 5)))
    return np.asarray(vals, dtype=np.float64)


def _segment_trend(prefix: str, x: np.ndarray, n_segments: int = 4) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    chunks = [c for c in np.array_split(x, n_segments) if c.size]
    means = np.asarray([np.mean(c) for c in chunks], dtype=np.float64)
    stds = np.asarray([np.std(c) for c in chunks], dtype=np.float64)
    feats: dict[str, float] = {}
    if means.size >= 2:
        idx = np.arange(means.size, dtype=np.float64)
        slope = np.polyfit(idx, means, deg=1)[0]
        feats[f"{prefix}_seg_mean_slope"] = float(slope)
        feats[f"{prefix}_seg_mean_first_last_delta"] = float(means[-1] - means[0])
        feats[f"{prefix}_seg_mean_range"] = float(np.max(means) - np.min(means))
        feats[f"{prefix}_seg_std_mean"] = float(np.mean(stds))
        feats[f"{prefix}_seg_std_range"] = float(np.max(stds) - np.min(stds))
    else:
        feats[f"{prefix}_seg_mean_slope"] = 0.0
        feats[f"{prefix}_seg_mean_first_last_delta"] = 0.0
        feats[f"{prefix}_seg_mean_range"] = 0.0
        feats[f"{prefix}_seg_std_mean"] = 0.0
        feats[f"{prefix}_seg_std_range"] = 0.0
    return feats


def extract_qvel_features(qvel: np.ndarray | None, fps: float, qpos_names: list[str]) -> dict[str, float]:
    feats: dict[str, float] = {}
    if qvel is None or qvel.size == 0:
        return feats
    qvel = np.asarray(qvel, dtype=np.float64)
    root_v = qvel[:, :3] if qvel.shape[1] >= 3 else np.zeros((qvel.shape[0], 3), dtype=np.float64)
    # Retargeted coordinates are z-up, so horizontal root speed uses x-y.
    root_speed = np.linalg.norm(root_v[:, [0, 1]], axis=1)
    feats.update(_safe_stats("qvel_root_speed", root_speed))
    feats["qvel_root_speed_regularity"] = autocorr_peak_ratio(root_speed, fps)
    feats["qvel_root_speed_freeze_index"] = freeze_index(root_speed, fps)

    joint_vel = qvel[:, 6:] if qvel.shape[1] > 6 else qvel
    if joint_vel.size:
        abs_v = np.abs(joint_vel)
        feats.update(_safe_stats("qvel_joint_abs_global", abs_v.reshape(-1)))
        feats["qvel_joint_abs_p95"] = float(np.percentile(abs_v, 95))
        feats["qvel_joint_abs_max"] = float(np.max(abs_v))
        energy = np.sqrt(np.mean(joint_vel * joint_vel, axis=1))
        feats.update(_safe_stats("qvel_joint_frame_energy", energy))
        feats["qvel_joint_energy_regularity"] = autocorr_peak_ratio(energy, fps)
    return feats


def extract_segment_features(qpos: np.ndarray, fps: float, qpos_names: list[str]) -> dict[str, float]:
    feats: dict[str, float] = {}
    root_pos = qpos[:, :3]
    speed = np.linalg.norm(np.diff(root_pos[:, [0, 1]], axis=0), axis=1) * fps
    feats.update(_segment_trend("seg_root_speed", speed))
    for joint in [
        "hip_flexion_l",
        "hip_flexion_r",
        "knee_angle_l",
        "knee_angle_r",
        "ankle_angle_l",
        "ankle_angle_r",
        "subtalar_angle_l",
        "subtalar_angle_r",
        "shoulder_elv_l",
        "shoulder_elv_r",
    ]:
        x = get_joint_signal(qpos, qpos_names, joint)
        if x is not None:
            feats.update(_segment_trend(f"seg_{joint}", x))
    return feats


def extract_event_keypoint_features(
    keypoints_world: np.ndarray,
    keypoints_pelvis: np.ndarray,
    body_names: list[str],
    fps: float,
) -> dict[str, float]:
    feats: dict[str, float] = {}
    idx = {n: i for i, n in enumerate(body_names)}
    required = ["calcn_l", "calcn_r", "toes_l", "toes_r"]
    if any(n not in idx for n in required):
        return feats

    event_map: dict[str, np.ndarray] = {}
    for side in ["l", "r"]:
        cal = keypoints_pelvis[:, idx[f"calcn_{side}"], :]
        toe = keypoints_pelvis[:, idx[f"toes_{side}"], :]
        foot_horizontal_y = 0.65 * cal[:, 1] + 0.35 * toe[:, 1]
        foot_z = 0.5 * cal[:, 2] + 0.5 * toe[:, 2]
        foot_world = 0.5 * keypoints_world[:, idx[f"calcn_{side}"], :] + 0.5 * keypoints_world[:, idx[f"toes_{side}"], :]
        events = _detect_minima(foot_z, fps)
        event_map[side] = events
        feats[f"evt_{side}_count"] = float(events.size)
        if events.size >= 2:
            intervals = np.diff(events) / fps
            feats.update(_safe_stats(f"evt_{side}_stride_time", intervals))
            y_rom, cycle_len = _cycle_values(foot_horizontal_y, events)
            horizontal_rom = _cycle_principal_rom(foot_world[:, [0, 1]], events)
            z_rom, _ = _cycle_values(foot_z, events)
            feats.update(_safe_stats(f"evt_{side}_cycle_y_rom", y_rom))
            feats.update(_safe_stats(f"evt_{side}_cycle_horizontal_rom", horizontal_rom))
            feats.update(_safe_stats(f"evt_{side}_cycle_z_rom", z_rom))
            stride_dist = np.linalg.norm(np.diff(foot_world[events][:, [0, 1]], axis=0), axis=1)
            feats.update(_safe_stats(f"evt_{side}_stride_dist", stride_dist))
            feats[f"evt_{side}_stride_dist_time_corr"] = (
                float(np.corrcoef(stride_dist, intervals)[0, 1])
                if stride_dist.size > 2 and np.std(stride_dist) > 1e-9 and np.std(intervals) > 1e-9
                else 0.0
            )
        else:
            feats.update(_safe_stats(f"evt_{side}_stride_time", np.array([])))
            feats.update(_safe_stats(f"evt_{side}_cycle_y_rom", np.array([])))
            feats.update(_safe_stats(f"evt_{side}_cycle_horizontal_rom", np.array([])))
            feats.update(_safe_stats(f"evt_{side}_cycle_z_rom", np.array([])))
            feats.update(_safe_stats(f"evt_{side}_stride_dist", np.array([])))
            feats[f"evt_{side}_stride_dist_time_corr"] = 0.0

        speed = np.linalg.norm(np.diff(foot_world[:, [0, 1]], axis=0), axis=1) * fps
        feats.update(_safe_stats(f"evt_{side}_foot_speed", speed))
        feats[f"evt_{side}_foot_z_regularity"] = autocorr_peak_ratio(foot_z, fps)
        feats[f"evt_{side}_foot_z_freeze_index"] = freeze_index(foot_z, fps)
        feats[f"evt_{side}_foot_z_domfreq"] = spectral_features(foot_z, fps)["dom_freq"]

    l_events = event_map["l"]
    r_events = event_map["r"]
    if l_events.size >= 2 and r_events.size >= 2:
        all_events = np.sort(np.concatenate([l_events, r_events]))
        step_times = np.diff(all_events) / fps
        feats.update(_safe_stats("evt_lr_step_time", step_times))
        l_intervals = np.diff(l_events) / fps
        r_intervals = np.diff(r_events) / fps
        n = min(l_intervals.size, r_intervals.size)
        if n:
            asym = np.abs(l_intervals[:n] - r_intervals[:n]) / (l_intervals[:n] + r_intervals[:n] + 1e-6)
            feats.update(_safe_stats("evt_lr_stride_time_asym", asym))
        phase_offsets = []
        for e in l_events:
            nearest = r_events[np.argmin(np.abs(r_events - e))]
            phase_offsets.append(abs(nearest - e) / fps)
        feats.update(_safe_stats("evt_lr_nearest_offset", np.asarray(phase_offsets, dtype=np.float64)))
    else:
        feats.update(_safe_stats("evt_lr_step_time", np.array([])))
        feats.update(_safe_stats("evt_lr_stride_time_asym", np.array([])))
        feats.update(_safe_stats("evt_lr_nearest_offset", np.array([])))
    return feats
