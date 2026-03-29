# fraud_detection/features/engineer.py
import logging
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

from fraud_detection.config import ProjectConfig
from fraud_detection.features.transformers import (
    AmountTransformer,
    TimeTransformer,
    PCAInteractionTransformer,
    VelocityTransformer,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Orchestrates all feature engineering steps.

    Design principle — each transformer does ONE thing:
    - AmountTransformer    handles Amount features only
    - TimeTransformer      handles Time features only
    - PCAInteractionTransformer  handles V feature interactions only
    - VelocityTransformer  handles rolling window features only

    This makes each transformer independently testable and
    replaceable without touching the others.
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig):
        self.spark  = spark
        self.config = config

        # Register transformers in the order they run
        # Order matters — later transformers can use earlier features
        self.transformers = [
            AmountTransformer(),
            TimeTransformer(),
            PCAInteractionTransformer(),
            VelocityTransformer(),
        ]

    def run(self) -> DataFrame:
        """Execute full feature engineering pipeline."""
        logger.info("=" * 55)
        logger.info("Starting feature engineering pipeline")
        logger.info("=" * 55)

        # Read training data from Delta Lake
        train_table = self.config.data.full_table("train")
        logger.info(f"Reading from: {train_table}")
        df = self.spark.table(train_table)

        original_cols = len(df.columns)
        logger.info(
            f"Input : {df.count():,} rows, "
            f"{original_cols} columns")

        # Apply each transformer in sequence
        # Each one receives the DataFrame from the previous step
        # and returns an enriched DataFrame with new columns added
        for transformer in self.transformers:
            class_name = transformer.__class__.__name__
            logger.info(f"Running {class_name}...")
            df = transformer.transform(df)

        new_cols = len(df.columns)
        logger.info(
            f"Output: {df.count():,} rows, "
            f"{new_cols} columns")
        logger.info(
            f"New features added: {new_cols - original_cols}")

        # Save enriched feature table to Delta Lake
        self._save_features(df)

        # Log fraud vs legit comparison for key features
        self._log_feature_summary(df)

        logger.info("Feature engineering pipeline complete ✓")
        return df

    def _save_features(self, df: DataFrame) -> None:
        """Write feature table to Delta Lake."""
        feature_table = self.config.data.full_table("features")
        logger.info(f"Saving to: {feature_table}")

        (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(feature_table)
        )
        logger.info(f"✓ Saved: {feature_table}")

    def _log_feature_summary(self, df: DataFrame) -> None:
        """
        Compare key engineered features between fraud and legit.
        This is your EDA confirmation — engineered features
        should show stronger separation than raw features.
        """
        logger.info("=" * 55)
        logger.info("Feature Summary — Fraud vs Legit")
        logger.info("=" * 55)
        logger.info(
            f"{'Feature':<30} {'Fraud':>8} {'Legit':>8} {'Gap':>8}")
        logger.info("-" * 55)

        fraud = df.filter(col("Class") == 1)
        legit = df.filter(col("Class") == 0)

        features_to_check = [
            "amount_log",
            "amount_zscore",
            "amount_is_round",
            "is_night",
            "is_peak_fraud_hour",
            "pca_anomaly_score",
            "v14_abs",
            "tx_count_last_100s",
            "amount_vs_rolling_mean",
        ]

        for feature in features_to_check:
            if feature not in df.columns:
                continue
            try:
                fraud_mean = fraud.agg(
                    mean(col(feature))
                ).collect()[0][0] or 0.0

                legit_mean = legit.agg(
                    mean(col(feature))
                ).collect()[0][0] or 0.0

                gap = abs(fraud_mean - legit_mean)
                logger.info(
                    f"{feature:<30} "
                    f"{fraud_mean:>8.3f} "
                    f"{legit_mean:>8.3f} "
                    f"{gap:>8.3f}")
            except Exception:
                pass

        logger.info("=" * 55)