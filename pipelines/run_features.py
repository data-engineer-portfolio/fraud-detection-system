# pipelines/run_features.py
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

# Replace logging.basicConfig with our centralised setup
from fraud_detection.utils.logger import setup_logging, suppress_spark_warnings
from fraud_detection.config import ProjectConfig
from fraud_detection.features.engineer import FeatureEngineer

# Set up logging first — before anything else
setup_logging()
suppress_spark_warnings()

import logging
logger = logging.getLogger(__name__)


def main():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("FraudDetection-Features") \
        .getOrCreate()

    config = ProjectConfig.from_yaml("conf/config.yml")
    logger.info(f"Config loaded: {config}")

    engineer = FeatureEngineer(spark, config)
    engineer.run()


if __name__ == "__main__":
    main()
