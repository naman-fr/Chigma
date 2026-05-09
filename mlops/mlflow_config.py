"""
MLflow Configuration and Setup
==============================
Industrial grade MLflow configuration for the Chigma project.
Handles tracking URI, registry, and experiment lifecycle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
import mlflow


class MLflowConfig:
    """Manages MLflow tracking setup and experiment configuration."""

    def __init__(self, config_path: str | Path = "configs/mlops.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.mlflow_cfg = self.config.get("mlflow", {})
        
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
        self.registry_uri = self.mlflow_cfg.get("registry_uri", "sqlite:///mlflow.db")
        self.experiment_name = self.mlflow_cfg.get("default_experiment", "chigma-defect-detection")

    def _load_config(self) -> dict[str, Any]:
        """Load MLOps configuration from YAML."""
        if not self.config_path.exists():
            logger.warning(f"MLOps config not found at {self.config_path}, using defaults")
            return {}
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def setup(self) -> None:
        """Initialize MLflow tracking and experiment."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_registry_uri(self.registry_uri)
            
            # Create or set experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.mlflow_cfg.get("artifact_root", "./mlruns")
                )
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow initialized: URI={self.tracking_uri}, Experiment={self.experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")

if __name__ == "__main__":
    # Test initialization
    config = MLflowConfig()
    config.setup()
