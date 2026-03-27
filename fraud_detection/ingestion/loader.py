# fraud_detection/ingestion/loader.py
import logging
import os
import shutil
import subprocess
from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

from fraud_detection.config import ProjectConfig
from fraud_detection.ingestion.validator import DataValidator

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles end-to-end data ingestion:
      1. Download raw CSV to local driver node
      2. Copy to DBFS
      3. Validate schema and quality
      4. Split into train / test
      5. Write all tables to Delta Lake
    """

    DOWNLOAD_URL = (
    "https://storage.googleapis.com/"
    "download.tensorflow.org/data/creditcard.csv"
    )
    BACKUP_URL = (
    "https://datahub.io/machine-learning/creditcard/"
    "r/creditcard.csv"
    )
    LOCAL_TMP = "/tmp/fraud-detection/"
    MIN_FILE_SIZE_MB = 100

    def __init__(self, spark: SparkSession, config: ProjectConfig):
        self.spark = spark
        self.config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full ingestion pipeline."""
        logger.info("=" * 50)
        logger.info("Starting ingestion pipeline")
        logger.info("=" * 50)

        self._setup_database()
        self._download_raw_data()
        self._copy_to_dbfs()

        df = self._read_csv()
        self._validate(df)

        df_train, df_test = self._split(df)

        self._write_delta(df,       self.config.data.tables["raw"])
        self._write_delta(df_train, self.config.data.tables["train"])
        self._write_delta(df_test,  self.config.data.tables["test"])

        self._log_summary(df, df_train, df_test)
        logger.info("Ingestion pipeline complete ✓")

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _setup_database(self) -> None:
        logger.info(f"Creating database: {self.config.data.database}")
        self.spark.sql(
            f"CREATE DATABASE IF NOT EXISTS {self.config.data.database}"
        )
        self.spark.sql(f"USE {self.config.data.database}")

    def _download_raw_data(self) -> None:
        os.makedirs(self.LOCAL_TMP, exist_ok=True)
        local_path = os.path.join(self.LOCAL_TMP, "creditcard.csv")

        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / 1_000_000
            if size_mb >= self.MIN_FILE_SIZE_MB:
                logger.info(f"File already downloaded ({size_mb:.1f} MB), skipping")
                return

        logger.info("Downloading dataset from primary source...")
        self._wget(self.DOWNLOAD_URL, local_path)

        size_mb = os.path.getsize(local_path) / 1_000_000
        if size_mb < self.MIN_FILE_SIZE_MB:
            logger.warning(
                f"File too small ({size_mb:.1f} MB), trying backup source..."
            )
            self._wget(self.BACKUP_URL, local_path)

        size_mb = os.path.getsize(local_path) / 1_000_000
        logger.info(f"✓ Downloaded: {size_mb:.1f} MB")

    def _wget(self, url: str, dest: str) -> None:
        result = subprocess.run(
            ["wget", "-q", "-O", dest, url],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Download failed from {url}:\n{result.stderr}"
            )

    def _copy_to_dbfs(self) -> None:
        dbfs_dir  = "/dbfs" + self.config.data.raw_path.rsplit("/", 1)[0]
        dbfs_path = "/dbfs" + self.config.data.raw_path
        os.makedirs(dbfs_dir, exist_ok=True)

        logger.info(f"Copying to DBFS: {self.config.data.raw_path}")
        shutil.copy2(
            os.path.join(self.LOCAL_TMP, "creditcard.csv"),
            dbfs_path
        )
        logger.info("✓ File copied to DBFS")

    def _read_csv(self) -> DataFrame:
        logger.info("Reading CSV into Spark DataFrame...")
        df = (
            self.spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(self.config.data.raw_path)
        )
        logger.info(f"✓ Loaded {df.count():,} rows, {len(df.columns)} columns")
        return df

    def _validate(self, df: DataFrame) -> None:
        validator = DataValidator(df)
        result = validator.validate()
        if not result.passed:
            raise ValueError(
                f"Data validation failed with {len(result.issues)} issues:\n"
                + "\n".join(f"  - {i}" for i in result.issues)
            )

    def _split(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        logger.info(
            f"Splitting data: "
            f"{self.config.data.train_ratio:.0%} train / "
            f"{1 - self.config.data.train_ratio:.0%} test"
        )
        df_train, df_test = df.randomSplit(
            [self.config.data.train_ratio,
             1 - self.config.data.train_ratio],
            seed=self.config.data.random_seed
        )
        return df_train, df_test

    def _write_delta(self, df: DataFrame, table_name: str) -> None:
        full_table = f"{self.config.data.database}.{table_name}"
        logger.info(f"Writing Delta table: {full_table}")
        (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(full_table)
        )
        logger.info(f"✓ Saved: {full_table}")

    def _log_summary(
        self,
        df: DataFrame,
        df_train: DataFrame,
        df_test: DataFrame
    ) -> None:
        total  = df.count()
        fraud  = df.filter(col("Class") == 1).count()
        legit  = total - fraud

        logger.info("=" * 50)
        logger.info("Ingestion Summary")
        logger.info("=" * 50)
        logger.info(f"  Total rows   : {total:>10,}")
        logger.info(f"  Fraud rows   : {fraud:>10,} ({100*fraud/total:.3f}%)")
        logger.info(f"  Legit rows   : {legit:>10,} ({100*legit/total:.3f}%)")
        logger.info(f"  Train rows   : {df_train.count():>10,}")
        logger.info(f"  Test rows    : {df_test.count():>10,}")
        logger.info(f"  Imbalance    : 1 fraud per {round(legit/fraud):,} legit")
        logger.info("=" * 50)
