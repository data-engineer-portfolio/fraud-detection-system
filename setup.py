# setup.py
from setuptools import setup, find_packages

setup(
    name="fraud-detection-system",
    version="1.0.0",
    author="Vidyullatha Polavarapu",
    description="Production-grade real-time fraud detection on Databricks",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # Only what the core package needs to import cleanly
        # Heavy ML deps (xgboost, mlflow, pyspark) live in
        # requirements.txt and are installed on Databricks directly
        "pyyaml>=6.0",
    ],
)
