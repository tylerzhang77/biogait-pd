import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation as Rot


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def spectral_features(x: np.ndarray, fs: float) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 16:
        return {"dom_freq": 0.0, "spec_entropy": 0.0}
    x = x - np.mean(x)
    f, pxx = signal.welch(x, fs=fs, nperseg=min(256, x.size))
    if pxx.size <= 1:
        return {"dom_freq": 0.0, "spec_entropy": 0.0}
    p = np.maximum(pxx[1:], 1e-12)
    ff = f[1:]
    dom = float(ff[np.argmax(p)])
    p_norm = p / np.sum(p)
    ent = float(-np.sum(p_norm * np.log(p_norm)) / np.log(p_norm.size))
    return {"dom_freq": dom, "spec_entropy": ent}


def phase_features(x_l: np.ndarray, x_r: np.ndarray) -> dict[str, float]:
    n = min(x_l.size, x_r.size)
    if n < 32:
        return {"phase_plv": 0.0, "phase_mean_abs_diff": 0.0}
    zl = signal.hilbert(x_l[:n] - np.mean(x_l[:n]))
    zr = signal.hilbert(x_r[:n] - np.mean(x_r[:n]))
    diff = np.angle(np.exp(1j * (np.angle(zl) - np.angle(zr))))
    return {
        "phase_plv": float(np.abs(np.mean(np.exp(1j * diff)))),
        "phase_mean_abs_diff": float(np.mean(np.abs(diff))),
    }


def autocorr_peak_ratio(x: np.ndarray, fs: float, min_hz: float = 0.5, max_hz: float = 3.0) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 64:
        return 0.0
    x = x - np.mean(x)
    ac = signal.correlate(x, x, mode="full")
    ac = ac[ac.size // 2 :]
    ac0 = ac[0] if ac.size else 1.0
    if ac0 <= 1e-12:
        return 0.0
    min_lag = max(1, int(fs / max_hz))
    max_lag = min(ac.size - 1, int(fs / min_hz))
    if max_lag <= min_lag:
        return 0.0
    return float(np.max(ac[min_lag:max_lag]) / ac0)


def freeze_index(x: np.ndarray, fs: float) -> float:
    """
    Proxy freezing index: high-frequency locomotor energy ratio.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 64:
        return 0.0
    f, pxx = signal.welch(x - np.mean(x), fs=fs, nperseg=min(256, x.size))
    loco = np.sum(pxx[(f >= 0.5) & (f < 3.0)])
    freeze = np.sum(pxx[(f >= 3.0) & (f < 8.0)])
    return float(freeze / (1e-9 + loco))


def get_joint_signal(qpos: np.ndarray, qpos_names: list[str], joint_name: str) -> np.ndarray | None:
    if joint_name not in qpos_names:
        return None
    return qpos[:, qpos_names.index(joint_name)]


def extract_joint_features(qpos: np.ndarray, fps: float, qpos_names: list[str]) -> dict[str, float]:
    angles = qpos[:, 7:]
    vel = np.diff(angles, axis=0) * fps
    acc = np.diff(vel, axis=0) * fps if vel.shape[0] > 1 else np.zeros((0, vel.shape[1]))
    feats: dict[str, float] = {}
    for j in range(angles.shape[1]):
        name = qpos_names[j + 7]
        x = angles[:, j]
        v = vel[:, j] if vel.shape[0] else np.array([0.0])
        a = acc[:, j] if acc.shape[0] else np.array([0.0])
        feats[f"{name}__mean"] = float(np.mean(x))
        feats[f"{name}__std"] = float(np.std(x))
        feats[f"{name}__rom"] = float(np.percentile(x, 95) - np.percentile(x, 5))
        feats[f"{name}__vel_rms"] = rms(v)
        feats[f"{name}__acc_rms"] = rms(a)
        sf = spectral_features(x, fps)
        feats[f"{name}__dom_freq"] = sf["dom_freq"]
        feats[f"{name}__spec_entropy"] = sf["spec_entropy"]
        feats[f"{name}__autocorr_regularity"] = autocorr_peak_ratio(x, fps)
    return feats


def extract_gait_proxy_features(qpos: np.ndarray, fps: float, qpos_names: list[str]) -> dict[str, float]:
    feats: dict[str, float] = {}
    root_pos = qpos[:, :3]
    root_quat = qpos[:, 3:7]
    # The MS-Human-700 retargeted coordinates are z-up. Horizontal motion is
    # therefore measured in the x-y plane and vertical excursion on z.
    speed = np.linalg.norm(np.diff(root_pos[:, [0, 1]], axis=0), axis=1) * fps
    feats["gait_speed_mean"] = float(np.mean(speed)) if speed.size else 0.0
    feats["gait_speed_std"] = float(np.std(speed)) if speed.size else 0.0
    feats["gait_speed_p5_p95_ratio"] = float(np.percentile(speed, 95) / (1e-6 + np.percentile(speed, 5))) if speed.size else 0.0
    feats["vertical_excursion"] = float(np.percentile(root_pos[:, 2], 95) - np.percentile(root_pos[:, 2], 5))
    root_vel = np.diff(root_pos, axis=0) * fps
    root_acc = np.diff(root_vel, axis=0) * fps if root_vel.shape[0] > 1 else np.zeros((0, 3))
    root_jerk = np.diff(root_acc, axis=0) * fps if root_acc.shape[0] > 1 else np.zeros((0, 3))
    feats["root_acc_rms"] = rms(root_acc)
    feats["root_jerk_rms"] = rms(root_jerk)

    try:
        rot = Rot.from_quat(np.column_stack([root_quat[:, 1], root_quat[:, 2], root_quat[:, 3], root_quat[:, 0]]))
        yaw = np.unwrap(rot.as_euler("xyz", degrees=False)[:, 2])
        yaw_rate = np.diff(yaw) * fps
        feats["turn_total_abs_deg"] = float(np.sum(np.abs(np.diff(yaw))) * 180.0 / np.pi)
        feats["turn_net_deg"] = float((yaw[-1] - yaw[0]) * 180.0 / np.pi)
        feats["turn_rate_std"] = float(np.std(yaw_rate)) if yaw_rate.size else 0.0
    except Exception:
        feats["turn_total_abs_deg"] = 0.0
        feats["turn_net_deg"] = 0.0
        feats["turn_rate_std"] = 0.0

    knee_r = get_joint_signal(qpos, qpos_names, "knee_angle_r")
    knee_l = get_joint_signal(qpos, qpos_names, "knee_angle_l")
    ankle_r = get_joint_signal(qpos, qpos_names, "ankle_angle_r")
    ankle_l = get_joint_signal(qpos, qpos_names, "ankle_angle_l")
    mtp_r = get_joint_signal(qpos, qpos_names, "mtp_angle_r")
    mtp_l = get_joint_signal(qpos, qpos_names, "mtp_angle_l")
    shoulder_r = get_joint_signal(qpos, qpos_names, "shoulder_elv_r")
    shoulder_l = get_joint_signal(qpos, qpos_names, "shoulder_elv_l")

    if knee_r is not None and knee_l is not None:
        stride_freq = (spectral_features(knee_r, fps)["dom_freq"] + spectral_features(knee_l, fps)["dom_freq"]) / 2.0
        feats["stride_freq_hz"] = float(stride_freq)
        feats["cadence_step_per_min"] = float(stride_freq * 120.0)
        ph = phase_features(knee_l, knee_r)
        feats["knee_phase_plv"] = ph["phase_plv"]
        feats["knee_phase_mean_abs_diff"] = ph["phase_mean_abs_diff"]
        rom_l = float(np.percentile(knee_l, 95) - np.percentile(knee_l, 5))
        rom_r = float(np.percentile(knee_r, 95) - np.percentile(knee_r, 5))
        feats["knee_rom_symmetry"] = float(abs(rom_l - rom_r) / (1e-6 + rom_l + rom_r))
        feats["knee_regularity_mean"] = float((autocorr_peak_ratio(knee_l, fps) + autocorr_peak_ratio(knee_r, fps)) / 2.0)
        feats["freeze_index_knee_mean"] = float((freeze_index(knee_l, fps) + freeze_index(knee_r, fps)) / 2.0)
        peaks_r, _ = signal.find_peaks(-knee_r, distance=max(1, int(0.35 * fps)))
        peaks_l, _ = signal.find_peaks(-knee_l, distance=max(1, int(0.35 * fps)))
        feats["heel_strike_proxy_count_r"] = float(len(peaks_r))
        feats["heel_strike_proxy_count_l"] = float(len(peaks_l))
    else:
        feats["stride_freq_hz"] = 0.0
        feats["cadence_step_per_min"] = 0.0
        feats["knee_phase_plv"] = 0.0
        feats["knee_phase_mean_abs_diff"] = 0.0
        feats["knee_rom_symmetry"] = 0.0
        feats["knee_regularity_mean"] = 0.0
        feats["freeze_index_knee_mean"] = 0.0
        feats["heel_strike_proxy_count_r"] = 0.0
        feats["heel_strike_proxy_count_l"] = 0.0

    if ankle_r is not None and ankle_l is not None and mtp_r is not None and mtp_l is not None:
        toe_r = ankle_r + mtp_r
        toe_l = ankle_l + mtp_l
        clr_r = float(np.percentile(toe_r, 95) - np.percentile(toe_r, 5))
        clr_l = float(np.percentile(toe_l, 95) - np.percentile(toe_l, 5))
        feats["toe_clearance_proxy_r"] = clr_r
        feats["toe_clearance_proxy_l"] = clr_l
        feats["toe_clearance_proxy_symmetry"] = float(abs(clr_l - clr_r) / (1e-6 + clr_l + clr_r))
    else:
        feats["toe_clearance_proxy_r"] = 0.0
        feats["toe_clearance_proxy_l"] = 0.0
        feats["toe_clearance_proxy_symmetry"] = 0.0

    if shoulder_r is not None and shoulder_l is not None:
        arm_r = float(np.percentile(shoulder_r, 95) - np.percentile(shoulder_r, 5))
        arm_l = float(np.percentile(shoulder_l, 95) - np.percentile(shoulder_l, 5))
        feats["arm_swing_rom_r"] = arm_r
        feats["arm_swing_rom_l"] = arm_l
        feats["arm_swing_symmetry"] = float(abs(arm_l - arm_r) / (1e-6 + arm_l + arm_r))
    else:
        feats["arm_swing_rom_r"] = 0.0
        feats["arm_swing_rom_l"] = 0.0
        feats["arm_swing_symmetry"] = 0.0

    return feats


def _spectral_band_energy_ratio(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 32:
        return 0.0
    f, pxx = signal.welch(x - np.mean(x), fs=fs, nperseg=min(256, x.size))
    total = np.sum(pxx) + 1e-12
    band = np.sum(pxx[(f >= f_lo) & (f < f_hi)])
    return float(band / total)


def _crosscorr_peak_lag(x: np.ndarray, y: np.ndarray, fs: float, max_lag_sec: float = 1.0) -> float:
    n = min(x.size, y.size)
    if n < 32:
        return 0.0
    x0 = x[:n] - np.mean(x[:n])
    y0 = y[:n] - np.mean(y[:n])
    cc = signal.correlate(x0, y0, mode="full")
    lags = signal.correlation_lags(x0.size, y0.size, mode="full")
    max_lag = int(max(1, fs * max_lag_sec))
    mask = np.abs(lags) <= max_lag
    if not np.any(mask):
        return 0.0
    lag = lags[mask][int(np.argmax(np.abs(cc[mask])))]
    return float(lag / fs)


def extract_keypoint_features(
    keypoints_world: np.ndarray,
    keypoints_pelvis: np.ndarray,
    body_names: list[str],
    fps: float,
) -> dict[str, float]:
    """
    keypoints_world / pelvis: (T, K, 3)
    """
    feats: dict[str, float] = {}
    t_len, k_len, _ = keypoints_world.shape
    if t_len < 4 or k_len == 0:
        return feats

    idx = {n: i for i, n in enumerate(body_names)}
    vel = np.diff(keypoints_pelvis, axis=0) * fps  # (T-1,K,3)
    acc = np.diff(vel, axis=0) * fps if vel.shape[0] > 1 else np.zeros((0, k_len, 3), dtype=np.float64)
    jerk = np.diff(acc, axis=0) * fps if acc.shape[0] > 1 else np.zeros((0, k_len, 3), dtype=np.float64)

    # Per-keypoint motion descriptors.
    for i, n in enumerate(body_names):
        # Pelvis-centered pelvis coordinates are identically zero, so per-body
        # descriptors for pelvis are uninformative. Global pelvis trajectory is
        # still computed below from world coordinates.
        if n == "pelvis":
            continue
        p = keypoints_pelvis[:, i, :]
        v = vel[:, i, :] if vel.shape[0] else np.zeros((1, 3), dtype=np.float64)
        a = acc[:, i, :] if acc.shape[0] else np.zeros((1, 3), dtype=np.float64)
        j = jerk[:, i, :] if jerk.shape[0] else np.zeros((1, 3), dtype=np.float64)
        z_sig = p[:, 2]
        sf_z = spectral_features(z_sig, fps)
        path_len = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        end_to_end = float(np.linalg.norm(p[-1] - p[0])) if p.shape[0] > 1 else 0.0
        feats[f"kp_{n}__disp_rms"] = float(np.sqrt(np.mean(np.sum(p * p, axis=1))))
        feats[f"kp_{n}__speed_rms"] = float(np.sqrt(np.mean(np.sum(v * v, axis=1))))
        feats[f"kp_{n}__acc_rms"] = float(np.sqrt(np.mean(np.sum(a * a, axis=1))))
        feats[f"kp_{n}__jerk_rms"] = float(np.sqrt(np.mean(np.sum(j * j, axis=1))))
        feats[f"kp_{n}__x_rom"] = float(np.percentile(p[:, 0], 95) - np.percentile(p[:, 0], 5))
        feats[f"kp_{n}__y_rom"] = float(np.percentile(p[:, 1], 95) - np.percentile(p[:, 1], 5))
        feats[f"kp_{n}__z_rom"] = float(np.percentile(z_sig, 95) - np.percentile(z_sig, 5))
        feats[f"kp_{n}__x_domfreq"] = spectral_features(p[:, 0], fps)["dom_freq"]
        feats[f"kp_{n}__y_domfreq"] = spectral_features(p[:, 1], fps)["dom_freq"]
        feats[f"kp_{n}__z_domfreq"] = sf_z["dom_freq"]
        feats[f"kp_{n}__z_spec_entropy"] = sf_z["spec_entropy"]
        feats[f"kp_{n}__z_regularity"] = autocorr_peak_ratio(z_sig, fps)
        feats[f"kp_{n}__z_freeze_index"] = freeze_index(z_sig, fps)
        feats[f"kp_{n}__path_efficiency"] = float(end_to_end / (1e-6 + path_len))

    # Symmetry and phase between left-right pairs.
    pairs = [
        ("femur_l", "femur_r"),
        ("tibia_l", "tibia_r"),
        ("calcn_l", "calcn_r"),
        ("toes_l", "toes_r"),
        ("humerus_l", "humerus_r"),
        ("radius_l", "radius_r"),
        ("hand_l", "hand_r"),
    ]
    for l, r in pairs:
        if l in idx and r in idx:
            pl = keypoints_pelvis[:, idx[l], :]
            pr = keypoints_pelvis[:, idx[r], :]
            rom_l = np.percentile(pl[:, 2], 95) - np.percentile(pl[:, 2], 5)
            rom_r = np.percentile(pr[:, 2], 95) - np.percentile(pr[:, 2], 5)
            feats[f"kp_sym_{l}_{r}__z_rom_ratio"] = float(abs(rom_l - rom_r) / (1e-6 + rom_l + rom_r))
            ph = phase_features(pl[:, 2], pr[:, 2])
            feats[f"kp_phase_{l}_{r}__z_plv"] = ph["phase_plv"]
            feats[f"kp_phase_{l}_{r}__z_mad"] = ph["phase_mean_abs_diff"]
            feats[f"kp_phase_{l}_{r}__z_xcorr_lag_s"] = _crosscorr_peak_lag(pl[:, 2], pr[:, 2], fps)

    # Global trajectory descriptors from pelvis in world coordinates.
    if "pelvis" in idx:
        pel = keypoints_world[:, idx["pelvis"], :]
        speed = np.linalg.norm(np.diff(pel[:, [0, 1]], axis=0), axis=1) * fps
        feats["kp_global_speed_mean"] = float(np.mean(speed)) if speed.size else 0.0
        feats["kp_global_speed_std"] = float(np.std(speed)) if speed.size else 0.0
        feats["kp_global_speed_entropy"] = spectral_features(speed, fps)["spec_entropy"] if speed.size > 16 else 0.0
        feats["kp_global_speed_band_0p5_3"] = _spectral_band_energy_ratio(speed, fps, 0.5, 3.0) if speed.size > 16 else 0.0
        if speed.size > 2:
            feats["kp_global_speed_cv"] = float(np.std(speed) / (1e-6 + np.mean(speed)))
        else:
            feats["kp_global_speed_cv"] = 0.0

    # Centroid sway/stability in pelvis coordinates.
    center = np.mean(keypoints_pelvis, axis=1)  # (T,3)
    feats["kp_center_sway_x_rms"] = rms(center[:, 0])
    feats["kp_center_sway_y_rms"] = rms(center[:, 1])
    feats["kp_center_sway_z_rms"] = rms(center[:, 2])
    feats["kp_center_sway_z_domfreq"] = spectral_features(center[:, 2], fps)["dom_freq"]
    feats["kp_center_sway_z_regularity"] = autocorr_peak_ratio(center[:, 2], fps)

    # Relative trunk stability: head vs pelvis if available.
    head_name = "head_neck" if "head_neck" in idx else "head"
    if head_name in idx and "pelvis" in idx:
        rel = keypoints_pelvis[:, idx[head_name], :] - keypoints_pelvis[:, idx["pelvis"], :]
        feats["kp_head_neck_rel_x_std"] = float(np.std(rel[:, 0]))
        feats["kp_head_neck_rel_y_std"] = float(np.std(rel[:, 1]))
        feats["kp_head_neck_rel_z_std"] = float(np.std(rel[:, 2]))
        feats["kp_head_neck_rel_z_rom"] = float(np.percentile(rel[:, 2], 95) - np.percentile(rel[:, 2], 5))

    return feats
