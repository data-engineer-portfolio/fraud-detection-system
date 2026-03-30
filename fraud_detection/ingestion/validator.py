# fraud_detection/ingestion/validator.py
import logging
from dataclasses import dataclass
from typing import List, Tuple
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, isnan, when, count

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    total_rows: int
    issues: List[str]

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"ValidationResult({status} | "
            f"rows={self.total_rows:,} | "
            f"issues={len(self.issues)})"
        )


class DataValidator:
    """
    Validates raw ingested data against expected schema
    and quality thresholds before writing to Delta Lake.
    """

    # Columns that must always be present
    REQUIRED_COLUMNS = [
        "Time", "Amount", "Class",
        *[f"V{i}" for i in range(1, 29)]   # V1 through V28
    ]

    # Fraud label must only be 0 or 1
    VALID_CLASS_VALUES = {0, 1}

    # Maximum tolerable null rate per column (5%)
    MAX_NULL_RATE = 0.05

    def __init__(self, df: DataFrame):
        self.df = df
        self.issues: List[str] = []

    def check_required_columns(self) -> "DataValidator":
        """Verify all expected columns are present."""
        actual = set(self.df.columns)
        missing = set(self.REQUIRED_COLUMNS) - actual
        if missing:
            self.issues.append(
                f"Missing required columns: {sorted(missing)}"
            )
        else:
            logger.info("✓ All required columns present")
        return self

    def check_row_count(self, min_rows: int = 1000) -> "DataValidator":
        """Ensure dataset is not suspiciously small."""
        count = self.df.count()
        if count < min_rows:
            self.issues.append(
                f"Row count {count:,} is below minimum threshold {min_rows:,}"
            )
        else:
            logger.info(f"✓ Row count: {count:,}")
        return self

    def check_null_rates(self) -> "DataValidator":
        """Flag columns with excessive null/NaN rates."""
        total = self.df.count()
        null_counts = self.df.select([
            count(
                when(col(c).isNull() | isnan(col(c)), c)
            ).alias(c)
            for c in self.REQUIRED_COLUMNS
            if c in self.df.columns
        ]).collect()[0].asDict()

        for col_name, null_count in null_counts.items():
            null_rate = null_count / total
            if null_rate > self.MAX_NULL_RATE:
                self.issues.append(
                    f"Column '{col_name}' has {null_rate:.1%} nulls "
                    f"(threshold: {self.MAX_NULL_RATE:.1%})"
                )

        if not any("null" in i for i in self.issues):
            logger.info("✓ Null rates within acceptable limits")
        return self

    def check_class_labels(self) -> "DataValidator":
        """Ensure Class column contains only 0 and 1."""
        if "Class" not in self.df.columns:
            return self

        distinct_classes = {
            row["Class"]
            for row in self.df.select("Class").distinct().collect()
        }
        invalid = distinct_classes - self.VALID_CLASS_VALUES
        if invalid:
            self.issues.append(
                f"Unexpected Class values found: {invalid}"
            )
        else:
            logger.info("✓ Class labels valid (0 and 1 only)")
        return self

    def check_amount_validity(self) -> "DataValidator":
        """Amount should never be negative."""
        if "Amount" not in self.df.columns:
            return self

        neg_count = self.df.filter("Amount < 0").count()
        if neg_count > 0:
            self.issues.append(
                f"Found {neg_count:,} rows with negative Amount values"
            )
        else:
            logger.info("✓ All Amount values are non-negative")
        return self

    def validate(self) -> ValidationResult:
        """Run all checks and return a ValidationResult."""
        logger.info("Starting data validation...")

        (
            self
            .check_required_columns()
            .check_row_count()
            .check_null_rates()
            .check_class_labels()
            .check_amount_validity()
        )

        passed = len(self.issues) == 0
        result = ValidationResult(
            passed=passed,
            total_rows=self.df.count(),
            issues=self.issues
        )

        if passed:
            logger.info(f"✓ Validation passed — {result.total_rows:,} rows")
        else:
            logger.warning(f"✗ Validation failed — {len(self.issues)} issues")
            for issue in self.issues:
                logger.warning(f"  → {issue}")

        return result
