# tests/conftest.py
"""
Pytest configuration — loaded automatically before any test runs.
Mocks all PySpark dependencies so tests run in CI without Spark.
"""
import sys
from unittest.mock import MagicMock

pyspark_mock = MagicMock()

PYSPARK_MODULES = [
    "pyspark",
    "pyspark.sql",
    "pyspark.sql.functions",
    "pyspark.sql.types",
    "pyspark.sql.window",
    "pyspark.sql.streaming",
    "pyspark.sql.connect",
    "pyspark.dbutils",
    "pyspark.ml",
    "pyspark.ml.feature",
    "pyspark.ml.classification",
]

for module in PYSPARK_MODULES:
    sys.modules[module] = pyspark_mock