# setup.py
from setuptools import setup, find_packages

setup(
    name="fraud-detection-system",
    version="1.0.0",
    author="Your Name",
    description="Production-grade real-time fraud detection on Databricks",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pyyaml>=6.0",
        "xgboost>=1.7",
        "lightgbm>=3.3",
        "scikit-learn>=1.2",
        "mlflow>=2.0",
        "imbalanced-learn>=0.10",
        "pandas>=1.5",
        "numpy>=1.23",
    ],
)