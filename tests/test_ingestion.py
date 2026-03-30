# tests/test_ingestion.py
"""
Unit tests for ingestion module.
These tests run in CI without PySpark by mocking
all Spark dependencies.
"""
import sys
from unittest.mock import MagicMock, patch

# ─── Mock PySpark before any imports ──────────────────────
# This tricks Python into thinking PySpark is installed
# Must happen BEFORE importing any fraud_detection modules
pyspark_mock = MagicMock()
sys.modules["pyspark"]                    = pyspark_mock
sys.modules["pyspark.sql"]               = pyspark_mock
sys.modules["pyspark.sql.functions"]     = pyspark_mock
sys.modules["pyspark.sql.types"]         = pyspark_mock
sys.modules["pyspark.sql.window"]        = pyspark_mock
sys.modules["pyspark.dbutils"]           = pyspark_mock

# Now safe to import our modules
from fraud_detection.ingestion.validator import (       # noqa: E402
    DataValidator,
    ValidationResult,
)


# ─── Helper ───────────────────────────────────────────────
def make_mock_df(
    columns=None,
    row_count=1000,
    has_negatives=False,
    null_counts=None,
):
    """Create a mock Spark DataFrame for testing."""
    if columns is None:
        columns = (
            ["Time", "Amount", "Class"]
            + [f"V{i}" for i in range(1, 29)]
        )

    df = MagicMock()
    df.columns = columns
    df.count.return_value = row_count

    # Mock filter — returns a new mock whose count()
    # gives the right value based on has_negatives flag
    # This works regardless of what expression is passed to filter()
    filter_mock = MagicMock()
    filter_mock.count.return_value = 10 if has_negatives else 0
    df.filter.return_value = filter_mock

    # Mock null check — return zero nulls by default
    null_row = MagicMock()
    null_row.asDict.return_value = {
        c: (null_counts or {}).get(c, 0)
        for c in columns
    }
    df.select.return_value.collect.return_value = [null_row]

    # Mock distinct class values for check_class_labels
    mock_class_0 = MagicMock()
    mock_class_0.__getitem__ = lambda self, k: 0
    mock_class_1 = MagicMock()
    mock_class_1.__getitem__ = lambda self, k: 1
    (
        df.select.return_value
        .distinct.return_value
        .collect.return_value
    ) = [mock_class_0, mock_class_1]

    return df


# ─── ValidationResult tests ───────────────────────────────
class TestValidationResult:

    def test_passed_repr(self):
        result = ValidationResult(
            passed=True, total_rows=1000, issues=[]
        )
        assert "PASSED" in repr(result)

    def test_failed_repr(self):
        result = ValidationResult(
            passed=False,
            total_rows=100,
            issues=["Something wrong"]
        )
        assert "FAILED" in repr(result)

    def test_issue_count_in_repr(self):
        result = ValidationResult(
            passed=False,
            total_rows=100,
            issues=["issue1", "issue2"]
        )
        assert "issues=2" in repr(result)


# ─── DataValidator tests ───────────────────────────────────
class TestDataValidator:

    def test_instantiates(self):
        df = make_mock_df()
        assert DataValidator(df) is not None

    def test_passes_with_valid_data(self):
        df        = make_mock_df()
        validator = DataValidator(df)
        validator.check_required_columns()
        assert len(validator.issues) == 0

    def test_fails_missing_columns(self):
        df        = make_mock_df(columns=["Time", "Amount"])
        validator = DataValidator(df)
        validator.check_required_columns()
        assert any(
            "Missing required columns" in i
            for i in validator.issues
        )

    def test_fails_too_few_rows(self):
        df        = make_mock_df(row_count=50)
        validator = DataValidator(df)
        validator.check_row_count(min_rows=1000)
        assert any(
            "below minimum" in i
            for i in validator.issues
        )

    def test_passes_sufficient_rows(self):
        df        = make_mock_df(row_count=300_000)
        validator = DataValidator(df)
        validator.check_row_count(min_rows=1000)
        assert not any(
            "below minimum" in i
            for i in validator.issues
        )

    def test_fails_negative_amounts(self):
        df        = make_mock_df(has_negatives=True)
        validator = DataValidator(df)
        validator.check_amount_validity()
        assert any(
            "negative Amount" in i
            for i in validator.issues
        )

    def test_passes_non_negative_amounts(self):
        df        = make_mock_df(has_negatives=False)
        validator = DataValidator(df)
        validator.check_amount_validity()
        assert not any(
            "negative Amount" in i
            for i in validator.issues
        )

    def test_required_columns_list_contains_v14(self):
        """V14 must always be in required columns."""
        assert "V14" in DataValidator.REQUIRED_COLUMNS

    def test_required_columns_list_length(self):
        """Should have Time, Amount, Class + V1-V28 = 31 columns."""
        assert len(DataValidator.REQUIRED_COLUMNS) == 31

    def test_max_null_rate_is_five_percent(self):
        """Null threshold must be 5% — business requirement."""
        assert DataValidator.MAX_NULL_RATE == 0.05

    def test_validate_returns_validation_result(self):
        df     = make_mock_df()
        result = DataValidator(df).validate()
        assert isinstance(result, ValidationResult)

    def test_validate_passed_is_true_with_valid_data(self):
        df     = make_mock_df()
        result = DataValidator(df).validate()
        assert result.passed is True
