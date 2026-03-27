# pipelines/run_ingestion.py
import logging
import sys
import os

# Make the package importable when run from Databricks
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

from pyspark.sql import SparkSession
from fraud_detection.config import ProjectConfig
from fraud_detection.ingestion.loader import DataLoader

# Configure logging — shows up cleanly in Databricks notebook output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Initialising Spark session...")
    spark = SparkSession.builder \
        .appName("FraudDetection-Ingestion") \
        .getOrCreate()

    logger.info("Loading config...")
    config = ProjectConfig.from_yaml("conf/config.yml")
    logger.info(f"Config loaded: {config}")

    loader = DataLoader(spark, config)
    loader.run()


if __name__ == "__main__":
    main()