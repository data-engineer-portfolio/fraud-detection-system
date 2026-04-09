# fraud_detection/training/trainer.py
"""
Model training for fraud detection.

Trains XGBoost and LightGBM with proper imbalance handling,
tracks every experiment in MLflow, evaluates on held-out
validation set, and registers the best model.
"""
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from pyspark.sql import DataFrame, SparkSession

from fraud_detection.config import ProjectConfig
from fraud_detection.training.evaluator import (
    ModelEvaluator,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


# Columns that are NOT features — exclude from X
NON_FEATURE_COLS = [
    "Class",           # target label — never a feature
    "Time",            # raw seconds — we engineered better features
    "amount_percentile", # window function — can cause leakage
    "rolling_std_amount", # can be null for first rows
]


c