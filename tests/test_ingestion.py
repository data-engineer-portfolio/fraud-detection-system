# tests/test_ingestion.py
import pytest
from unittest.mock import MagicMock, patch
from fraud_detection.ingestion.validator import DataValidator, ValidationResult


def make_mock_df(
    columns=None,
    row_count=1000,
    has_negatives=False,
    class_values=None
):
    """Helper to create a mock Spark DataFrame for testing."""
    if columns is None:
        columns = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    if class_values is None:
        class_values = [0, 1]

    df = MagicMock()
    df.columns = columns
    df.count.return_value = row_count

    # Mock .select().collect() for null check
    null_row = MagicMock()
    null_row.asDict.return_value = {c: 0 for c in columns}
    df.select.return_value.collect.return_value = [null_row]

    # Mock .filter().count() for negative amount check
    df.filter.return_value.count.return_value = 10 if has_negatives else 0

    # Mock .select().distinct().collect() for class check
    class_rows = [MagicMock()] * len(class_values)
    for i, v in enumerate(class_values):
        class_rows[i].__getitem__ = lambda self, key, val=v: val
    df.select.return_value.distinct.return_value.collect.return_value = (
        class_rows
    )

    return df


class TestDataValidator:

    def test_passes_with_valid_data(self):
        df = make_mock_df()
        result = DataValidator(df).validate()
        assert isinstance(result, ValidationResult)

    def test_fails_missing_columns(self):
        df = make_mock_df(columns=["Time", "Amount"])  # missing Class + V cols
        validator = DataValidator(df)
        validator.check_required_columns()
        assert any("Missing required columns" in i for i in validator.issues)

    def test_fails_too_few_rows(self):
        df = make_mock_df(row_count=50)
        validator = DataValidator(df)
        validator.check_row_count(min_rows=1000)
        assert any("below minimum" in i for i in validator.issues)

    def test_fails_negative_amounts(self):
        df = make_mock_df(has_negatives=True)
        validator = DataValidator(df)
        validator.check_amount_validity()
        assert any("negative Amount" in i for i in validator.issues)

    def test_passes_with_sufficient_rows(self):
        df = make_mock_df(row_count=300_000)
        validator = DataValidator(df)
        validator.check_row_count(min_rows=1000)
        assert not any("below minimum" in i for i in validator.issues)