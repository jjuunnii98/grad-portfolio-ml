from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"

    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))

    from src.features.build_features import run_feature_pipeline_from_config
    from src.models.survival_model import (
        run_refinement_pipeline_from_config,
        train_and_save_inference_bundle_from_config,
    )
    from src.utils.config import load_config

    config_path = project_root / "configs" / "config.yaml"
    config = load_config(config_path)

    print("=" * 60)
    print("Survival Analysis Pipeline Started")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Config path : {config_path}")
    print()

    print("[1/3] Running feature engineering pipeline...")
    featurized_df = run_feature_pipeline_from_config(config_path)
    print("Feature engineering completed.")
    print(f"Featurized shape: {featurized_df.shape}")
    print()

    print("[2/3] Running survival model refinement pipeline...")
    model_performance, comparison_df = run_refinement_pipeline_from_config(config_path)
    print("Model refinement completed.")
    print()

    print("[3/3] Training final penalized Cox model and saving inference artifact...")
    artifact_path = train_and_save_inference_bundle_from_config(config_path)
    print(f"Inference artifact saved to: {artifact_path}")
    print()

    print("=" * 60)
    print("Best Model Performance")
    print("=" * 60)
    print(model_performance)
    print()

    print("=" * 60)
    print("Coefficient Comparison Preview")
    print("=" * 60)
    print(comparison_df.head())
    print()

    print("=" * 60)
    print("Configured Output Paths")
    print("=" * 60)
    print("Processed data path           :", config["data"]["processed_data_path"])
    print("Model performance output     :", config["output"]["model_performance_path"])
    print("Coefficient comparison output:", config["output"]["coefficient_comparison_path"])
    print("Model artifact path          :", config["output"]["model_artifact_path"])
    print()

    print("=" * 60)
    print("Pipeline Finished Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()