# pipelines/run_features.py
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

from pyspark.sql import SparkSession
from fraud_detection.config import ProjectConfig
from fraud_detection.features.engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    spark = SparkSession.builder \
        .appName("FraudDetection-Features") \
        .getOrCreate()

    config = ProjectConfig.from_yaml("conf/config.yml")
    logger.info(f"Config loaded: {config}")

    engineer = FeatureEngineer(spark, config)
    engineer.run()


if __name__ == "__main__":
    main()