import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StudyConfig:
    repo_root: Path
    label_csv: Path
    model_xml: Path
    output_root: Path
    keypoints_root: Path | None = None
    use_keypoints_fusion: bool = True
    use_advanced_features: bool = True
    random_seed: int = 42
    cv_splits: int = 5


def load_config(path: Path) -> StudyConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return StudyConfig(
        repo_root=Path(data["repo_root"]),
        label_csv=Path(data["label_csv"]),
        model_xml=Path(data["model_xml"]),
        output_root=Path(data["output_root"]),
        keypoints_root=Path(data["keypoints_root"]) if data.get("keypoints_root") else None,
        use_keypoints_fusion=bool(data.get("use_keypoints_fusion", True)),
        use_advanced_features=bool(data.get("use_advanced_features", True)),
        random_seed=int(data.get("random_seed", 42)),
        cv_splits=int(data.get("cv_splits", 5)),
    )
