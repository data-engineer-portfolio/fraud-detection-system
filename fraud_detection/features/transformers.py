# fraud_detection/features/transformers.py
import logging
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, log1p, when, lit, floor,
    mean, stddev, count, percent_rank
)
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

class AmountTransformer:
    """
    Engineers fraud-relevant features from transaction Amount.

    Motivation from data analysis:
    - Fraud mean: €131 vs Legit mean: €88
    - Both distributions are heavily right-skewed (std > mean)
    - Fraud transactions max at €2,125 — threshold gaming behaviour
    - Zero amounts exist in both classes — need log1p not log
    """

    def transform(self, df: DataFrame) -> DataFrame:
        logger.info("Building amount features...")

        # Compute training set statistics ONCE
        # These must come from training data only — never test data
        # (data leakage prevention — we discussed this)
        stats = df.select(
            mean("Amount").alias("mean"),
            stddev("Amount").alias("std")
        ).collect()[0]

        amt_mean = stats["mean"]
        amt_std  = stats["std"] if stats["std"] else 1.0

        df = (
            df
            # log1p — safe log transform, handles Amount=0 cleanly
            # Compresses right skew: €25k becomes 10.1, €100 becomes 4.6
            .withColumn(
                "amount_log",
                log1p(col("Amount")))

            # Z-score — how many std deviations from population mean?
            # High positive value = unusually large amount = suspicious
            .withColumn(
                "amount_zscore",
                (col("Amount") - lit(amt_mean)) / lit(amt_std))

            # Bins — captures threshold-gaming behaviour
            # Fraudsters stay below detection thresholds
            .withColumn(
                "amount_bin",
                when(col("Amount") <= 10,    lit(0.0))
                .when(col("Amount") <= 50,   lit(1.0))
                .when(col("Amount") <= 100,  lit(2.0))
                .when(col("Amount") <= 500,  lit(3.0))
                .when(col("Amount") <= 1000, lit(4.0))
                .when(col("Amount") <= 5000, lit(5.0))
                .otherwise(lit(6.0)))

            # Round number flag — fraudsters often use round amounts
            # €100, €500, €1000 are suspicious patterns
            .withColumn(
                "amount_is_round",
                when(col("Amount") % 10 == 0, lit(1.0))
                .otherwise(lit(0.0)))

            # Percentile rank — where does this sit in the distribution?
            # 0.0 = smallest amount seen, 1.0 = largest amount seen
            .withColumn(
                "amount_percentile",
                percent_rank().over(
                    Window.orderBy("Amount")))
        )

        logger.info(
            "✓ Amount features: amount_log, amount_zscore, "
            "amount_bin, amount_is_round, amount_percentile")
        return df

class TimeTransformer:
    """
    Engineers fraud-relevant features from transaction Time.

    Motivation:
    - Raw Time is seconds since first transaction — not meaningful
    - Fraud spikes at night (1am-4am) when cardholders are asleep
    - Converting to hour_of_day gives the model a clean signal
    """

    SECONDS_IN_HOUR = 3600
    SECONDS_IN_DAY  = 86400

    def transform(self, df: DataFrame) -> DataFrame:
        logger.info("Building time features...")

        df = (
            df
            # Hour of day 0-23
            # floor(Time / 3600) gives total hours elapsed
            # % 24 wraps it back to 0-23
            .withColumn(
                "hour_of_day",
                (floor(col("Time") / self.SECONDS_IN_HOUR) % 24)
                .cast(DoubleType()))

            # Nighttime flag — 11pm to 6am
            # Fraudsters strike when cardholders are asleep
            .withColumn(
                "is_night",
                when(
                    (col("hour_of_day") >= 23) |
                    (col("hour_of_day") <= 5),
                    lit(1.0))
                .otherwise(lit(0.0)))

            # Peak fraud window — 1am to 4am
            # Tighter window than is_night, stronger signal
            .withColumn(
                "is_peak_fraud_hour",
                when(
                    (col("hour_of_day") >= 1) &
                    (col("hour_of_day") <= 4),
                    lit(1.0))
                .otherwise(lit(0.0)))

            # Day number in dataset (day 1, day 2, etc.)
            # Useful for detecting drift over time
            .withColumn(
                "day_of_dataset",
                floor(col("Time") / self.SECONDS_IN_DAY)
                .cast(DoubleType()))

            # Seconds into current day (0 to 86399)
            # More granular than hour_of_day
            .withColumn(
                "time_of_day_seconds",
                (col("Time") % self.SECONDS_IN_DAY)
                .cast(DoubleType()))
        )

        logger.info(
            "✓ Time features: hour_of_day, is_night, "
            "is_peak_fraud_hour, day_of_dataset, time_of_day_seconds")
        return df

class PCAInteractionTransformer:
    """
    Engineers interaction features from V1-V28 PCA components.

    Motivation from data analysis:
    - V14 fraud mean: -6.87 vs legit mean: 0.01
    - Squaring V14 gives large positive for fraud, near-zero for legit
    - Interaction terms capture patterns that individual features miss
    - V14, V4, V11 are historically the most discriminating
    """

    # Most fraud-predictive V features based on our EDA
    # V14 showed -6.87 separation — strongest single feature
    TOP_FEATURES = ["V1", "V3", "V4", "V10", "V11", "V14"]

    def transform(self, df: DataFrame) -> DataFrame:
        logger.info("Building PCA interaction features...")

        # Anomaly score — sum of squares of top fraud indicators
        # For fraud:  large V values → large squares → high score
        # For legit:  V values near 0 → small squares → low score
        # This is essentially a distance-from-origin measure
        anomaly_score = sum(
            col(v) * col(v)
            for v in self.TOP_FEATURES
        )
        df = df.withColumn("pca_anomaly_score", anomaly_score)

        # Pairwise interactions — V14 × V4 captures patterns
        # that neither V14 nor V4 captures individually
        df = (
            df
            .withColumn(
                "v14_v4_interaction",
                col("V14") * col("V4"))

            .withColumn(
                "v14_v11_interaction",
                col("V14") * col("V11"))

            .withColumn(
                "v4_v11_interaction",
                col("V4") * col("V11"))

            # Absolute values — magnitude matters, not direction
            # V14 = -6.87 and V14 = +6.87 are equally suspicious
            .withColumn(
                "v14_abs",
                when(col("V14") < 0,
                     col("V14") * lit(-1.0))
                .otherwise(col("V14")))

            .withColumn(
                "v4_abs",
                when(col("V4") < 0,
                     col("V4") * lit(-1.0))
                .otherwise(col("V4")))
        )

        logger.info(
            "✓ PCA features: pca_anomaly_score, v14_v4_interaction, "
            "v14_v11_interaction, v4_v11_interaction, v14_abs, v4_abs")
        return df

class VelocityTransformer:
    """
    Engineers velocity features using time-based rolling windows.

    Motivation:
    - Stolen cards get used rapidly before cardholder notices
    - Multiple transactions in short time = strongest fraud signal
    - We use population velocity (not per-card) because
      card IDs are anonymized in this dataset
    - In production with card IDs you would partition by card_id
    """

    # Window sizes in seconds to look back
    WINDOWS = [100, 500, 1000]

    def transform(self, df: DataFrame) -> DataFrame:
        logger.info("Building velocity features...")

        # Rolling transaction count for each window size
        # rangeBetween(-N, 0) = look back N seconds from current row
        # orderBy("Time") = windows are time-ordered
        for window_size in self.WINDOWS:
            window_spec = (
                Window
                .orderBy("Time")
                .rangeBetween(-window_size, 0)
            )

            df = df.withColumn(
                f"tx_count_last_{window_size}s",
                count("Amount")
                .over(window_spec)
                .cast(DoubleType())
            )

        # Rolling amount statistics in last 1000 seconds
        # These capture unusual amounts relative to recent activity
        amount_window = (
            Window
            .orderBy("Time")
            .rangeBetween(-1000, 0)
        )

        df = (
            df
            .withColumn(
                "rolling_mean_amount",
                mean("Amount").over(amount_window))

            .withColumn(
                "rolling_std_amount",
                stddev("Amount").over(amount_window))

            # Ratio: this transaction vs recent average
            # Value of 5.0 means this amount is 5x recent average
            # High ratio = suspicious spike in amount
            .withColumn(
                "amount_vs_rolling_mean",
                when(
                    col("rolling_mean_amount").isNotNull() &
                    (col("rolling_mean_amount") != 0),
                    col("Amount") / col("rolling_mean_amount"))
                .otherwise(lit(1.0)))
        )

        logger.info(
            "✓ Velocity features: tx_count windows 100/500/1000s, "
            "rolling_mean_amount, rolling_std_amount, "
            "amount_vs_rolling_mean")
        return df
