# fraud_detection/config.py
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any
from pathlib import Path


@dataclass
class DataConfig:
    raw_path: str
    database: str
    tables: Dict[str, str]
    train_ratio: float
    random_seed: int


@dataclass
class ModelConfig:
    name: str
    experiment_name: str
    params: Dict[str, Any]
    threshold: float


@dataclass
class StreamingConfig:
    trigger_interval: str
    checkpoint_path: str
    output_table: str


@dataclass
class MonitoringConfig:
    drift_threshold: float
    performance_threshold: Dict[str, float]
    alert_email: str


@dataclass
class ProjectConfig:
    project_name: str
    version: str
    data: DataConfig
    model: ModelConfig
    streaming: StreamingConfig
    monitoring: MonitoringConfig

    @classmethod
    def from_yaml(cls, path: str) -> "ProjectConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return cls(
            project_name=raw["project"]["name"],
            version=raw["project"]["version"],
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            streaming=StreamingConfig(**raw["streaming"]),
            monitoring=MonitoringConfig(**raw["monitoring"]),
        )

    def __repr__(self):
        return (
            f"ProjectConfig(project={self.project_name} v{self.version}, "
            f"db={self.data.database}, model={self.model.name})"
        )