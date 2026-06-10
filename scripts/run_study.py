import argparse
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from parkinson_gait.config import load_config
    from parkinson_gait.pipeline import run_pipeline

    parser = argparse.ArgumentParser(description="Run the BioGait-PD feature extraction and modelling pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "study_config.local.example.json",
        help="Path to a local JSON configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_pipeline(cfg)
    print(result)


if __name__ == "__main__":
    main()
