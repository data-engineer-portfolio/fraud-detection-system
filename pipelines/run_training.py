# pipelines/run_training.py
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

from fraud_detection.utils.logger import (
    setup_logging,
    suppress_spark_warnings,
)
from fraud_detection.config import ProjectConfig
from fraud_detection.training.trainer import FraudModelTrainer

setup_logging()
suppress_spark_warnings()

logger = logging.getLogger(__name__)


def main():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("FraudDetection-Training") \
        .getOrCreate()

    config = ProjectConfig.from_yaml("conf/config.yml")
    logger.info(f"Config: {config}")

    trainer = FraudModelTrainer(spark, config)
    winner  = trainer.run()

    logger.info(f"Winner: {winner}")
    logger.info(f"PR-AUC: {winner.pr_auc:.4f}")
    logger.info(f"Recall: {winner.recall:.4f}")


if __name__ == "__main__":
    main()