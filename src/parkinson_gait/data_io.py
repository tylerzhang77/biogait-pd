from pathlib import Path

import mujoco
import numpy as np
import pandas as pd


def load_labels(label_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(label_csv)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["gait_label"] = pd.to_numeric(df["gait_label"], errors="coerce")
    df = df.dropna(subset=["gait_label", "retarget_npz"]).copy()
    df["gait_label"] = df["gait_label"].astype(int)
    return df


def safe_name(s: str) -> str:
    return s.replace(" ", "_")


def build_qpos_mapping(model_xml: Path, expected_nq: int) -> tuple[list[str], pd.DataFrame]:
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    if model.nq != expected_nq:
        raise ValueError(f"Model nq={model.nq} != expected nq={expected_nq}")

    qpos_names = ["root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz"]
    rows = []
    for i in range(7):
        rows.append({"qpos_index": i, "qpos_name": qpos_names[i], "joint_name": "pelvis_free", "is_root": True})

    joint_name_by_qpos = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        qadr = int(model.jnt_qposadr[j])
        jtype = int(model.jnt_type[j])
        width = {0: 7, 1: 4, 2: 1, 3: 1}.get(jtype, 1)
        for k in range(width):
            idx = qadr + k
            if idx < 7:
                continue
            suffix = "" if width == 1 else f"_q{k}"
            joint_name_by_qpos[idx] = safe_name(f"{name}{suffix}")

    for i in range(7, expected_nq):
        nm = joint_name_by_qpos.get(i, f"joint_{i - 6:03d}")
        qpos_names.append(nm)
        rows.append({"qpos_index": i, "qpos_name": nm, "joint_name": nm, "is_root": False})

    return qpos_names, pd.DataFrame(rows)


def load_motion(npz_path: Path) -> tuple[np.ndarray, float]:
    d = np.load(npz_path, allow_pickle=True)
    qpos = np.asarray(d["qpos"], dtype=np.float64)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d.files else 30.0
    return qpos, fps
