# tests/test_features.py
"""
Unit tests for feature engineering transformers.
Runs in CI without PySpark by mocking all Spark dependencies.
"""
import sys
from unittest.mock import MagicMock

# ─── Mock PySpark before any imports ──────────────────────
pyspark_mock = MagicMock()
sys.modules["pyspark"]                = pyspark_mock
sys.modules["pyspark.sql"]           = pyspark_mock
sys.modules["pyspark.sql.functions"] = pyspark_mock
sys.modules["pyspark.sql.types"]     = pyspark_mock
sys.modules["pyspark.sql.window"]    = pyspark_mock

# Now safe to import
from fraud_detection.features.transformers import (    # noqa: E402
    AmountTransformer,
    TimeTransformer,
    PCAInteractionTransformer,
    VelocityTransformer,
)


# ─── AmountTransformer ────────────────────────────────────
class TestAmountTransformer:

    def test_instantiates(self):
        assert AmountTransformer() is not None

    def test_has_transform_method(self):
        assert hasattr(AmountTransformer(), "transform")

    def test_is_callable(self):
        t = AmountTransformer()
        assert callable(t.transform)


# ─── TimeTransformer ──────────────────────────────────────
class TestTimeTransformer:

    def test_instantiates(self):
        assert TimeTransformer() is not None

    def test_seconds_in_hour_correct(self):
        assert TimeTransformer.SECONDS_IN_HOUR == 3600

    def test_seconds_in_day_correct(self):
        assert TimeTransformer.SECONDS_IN_DAY == 86400

    def test_seconds_in_day_equals_24_hours(self):
        t = TimeTransformer()
        assert t.SECONDS_IN_DAY == t.SECONDS_IN_HOUR * 24

    def test_has_transform_method(self):
        assert hasattr(TimeTransformer(), "transform")


# ─── PCAInteractionTransformer ────────────────────────────
class TestPCAInteractionTransformer:

    def test_instantiates(self):
        assert PCAInteractionTransformer() is not None

    def test_top_features_contains_v14(self):
        """V14 is the strongest fraud signal — must be included."""
        assert "V14" in PCAInteractionTransformer.TOP_FEATURES

    def test_top_features_contains_v4(self):
        assert "V4" in PCAInteractionTransformer.TOP_FEATURES

    def test_top_features_contains_v11(self):
        assert "V11" in PCAInteractionTransformer.TOP_FEATURES

    def test_top_features_length(self):
        """Must have exactly 6 top features."""
        assert len(PCAInteractionTransformer.TOP_FEATURES) == 6

    def test_has_transform_method(self):
        assert hasattr(PCAInteractionTransformer(), "transform")


# ─── VelocityTransformer ──────────────────────────────────
class TestVelocityTransformer:

    def test_instantiates(self):
        assert VelocityTransformer() is not None

    def test_windows_contains_100s(self):
        assert 100 in VelocityTransformer.WINDOWS

    def test_windows_contains_500s(self):
        assert 500 in VelocityTransformer.WINDOWS

    def test_windows_contains_1000s(self):
        assert 1000 in VelocityTransformer.WINDOWS

    def test_windows_has_three_sizes(self):
        assert len(VelocityTransformer.WINDOWS) == 3

    def test_windows_are_ordered(self):
        """Windows should go smallest to largest."""
        w = VelocityTransformer.WINDOWS
        assert w == sorted(w)

    def test_has_transform_method(self):
        assert hasattr(VelocityTransformer(), "transform")


# ─── Config tests ─────────────────────────────────────────
class TestProjectConfig:

    def test_config_loads_from_yaml(self, tmp_path):
        """Config must load correctly from a YAML file."""
        config_content = """
project:
  name: test-project
  version: "1.0.0"
data:
  catalog: workspace
  database: fraud_db
  tables:
    raw: transactions_raw
    train: transactions_train
    test: transactions_test
    features: features_train
    scored: scored_transactions
  train_ratio: 0.8
  random_seed: 42
model:
  name: fraud_xgboost
  experiment_name: /test/experiments
  params:
    n_estimators: 100
  threshold: 0.5
streaming:
  trigger_interval: "10 seconds"
  checkpoint_path: /tmp/checkpoints
  output_table: workspace.fraud_db.scored
monitoring:
  drift_threshold: 0.1
  performance_threshold:
    min_auc_pr: 0.85
    min_f1: 0.80
  alert_email: test@test.com
"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)

        from fraud_detection.config import ProjectConfig
        config = ProjectConfig.from_yaml(str(config_file))

        assert config.project_name == "test-project"
        assert config.data.catalog  == "workspace"
        assert config.data.database == "fraud_db"
        assert config.data.train_ratio == 0.8
        assert config.data.random_seed == 42

    def test_full_table_returns_correct_string(self, tmp_path):
        config_content = """
project:
  name: test
  version: "1.0.0"
data:
  catalog: workspace
  database: fraud_db
  tables:
    raw: transactions_raw
    train: transactions_train
    test: transactions_test
    features: features_train
    scored: scored_transactions
  train_ratio: 0.8
  random_seed: 42
model:
  name: fraud_xgboost
  experiment_name: /test/exp
  params:
    n_estimators: 100
  threshold: 0.5
streaming:
  trigger_interval: "10 seconds"
  checkpoint_path: /tmp/checkpoints
  output_table: workspace.fraud_db.scored
monitoring:
  drift_threshold: 0.1
  performance_threshold:
    min_auc_pr: 0.85
    min_f1: 0.80
  alert_email: test@test.com
"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)

        from fraud_detection.config import ProjectConfig
        config = ProjectConfig.from_yaml(str(config_file))

        assert config.data.full_table("train") == \
            "workspace.fraud_db.transactions_train"
        assert config.data.full_table("test") == \
            "workspace.fraud_db.transactions_test"
        assert config.data.full_database() == \
            "workspace.fraud_db"
