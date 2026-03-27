# fraud_detection/config.py
import yaml
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DataConfig:
    catalog: str
    database: str
    tables: Dict[str, str]
    train_ratio: float
    random_seed: int

    def full_table(self, key: str) -> str:
        """
        Return fully qualified 3-level Unity Catalog table name.
        e.g. full_table("train") -> "workspace.fraud_db.transactions_train"
        """
        return f"{self.catalog}.{self.database}.{self.tables[key]}"

    def full_database(self) -> str:
        """
        Return fully qualified catalog.database string.
        e.g. "workspace.fraud_db"
        """
        return f"{self.catalog}.{self.database}"


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
            f"ProjectConfig("
            f"project={self.project_name} v{self.version}, "
            f"db={self.data.full_database()}, "
            f"model={self.model.name})"
        )
