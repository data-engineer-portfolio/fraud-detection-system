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


class FraudModelTrainer:
    """
    Trains XGBoost and LightGBM fraud detection models.

    Design decisions:
    - SMOTE applied ONLY on training fold, never validation/test
    - scale_pos_weight computed from actual class distribution
    - MLflow autolog captures all params and metrics automatically
    - Both models tracked as separate MLflow runs for comparison
    - Best model registered in MLflow Model Registry
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig):
        self.spark  = spark
        self.config = config

    # ──────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────

    def run(self) -> EvaluationResult:
        """Execute full training pipeline."""
        logger.info("=" * 55)
        logger.info("Starting model training pipeline")
        logger.info("=" * 55)

        # Step 1 — load feature table
        X, y = self._load_features()

        # Step 2 — train/validation split
        X_train, X_val, y_train, y_val = self._split(X, y)

        # Step 3 — apply SMOTE on training data only
        X_train_bal, y_train_bal = self._apply_smote(
            X_train, y_train
        )

        # Step 4 — set up MLflow experiment
        self._setup_mlflow()

        # Step 5 — train both models, get evaluation results
        xgb_result = self._train_xgboost(
            X_train_bal, y_train_bal, X_val, y_val
        )
        lgb_result = self._train_lightgbm(
            X_train_bal, y_train_bal, X_val, y_val
        )

        # Step 6 — compare and register winner
        evaluator = ModelEvaluator("comparison")
        winner    = evaluator.compare(xgb_result, lgb_result)
        self._register_best_model(winner)

        logger.info("Training pipeline complete ✓")
        return winner

    # ──────────────────────────────────────────────────────
    # Private steps
    # ──────────────────────────────────────────────────────

    def _load_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load feature table from Delta Lake.
        Convert to Pandas for sklearn/xgboost compatibility.
        Drop non-feature columns to get clean X matrix.
        """
        feature_table = self.config.data.full_table("features")
        logger.info(f"Loading features from: {feature_table}")

        df_spark = self.spark.table(feature_table)

        # Drop columns that should never be features
        feature_cols = [
            c for c in df_spark.columns
            if c not in NON_FEATURE_COLS
        ]

        logger.info(f"Feature columns: {len(feature_cols)}")
        logger.info(f"Excluded columns: {NON_FEATURE_COLS}")

        # Convert to Pandas
        # Note: this is fine for 227k rows
        # For 100M+ rows you'd use Spark ML pipeline instead
        df_pandas = df_spark.select(
            feature_cols + ["Class"]
        ).toPandas()

        # Separate features from label
        X = df_pandas[feature_cols]
        y = df_pandas["Class"]

        # Log class distribution
        fraud_count = int(y.sum())
        legit_count = int((y == 0).sum())
        logger.info(
            f"Loaded: {len(X):,} rows | "
            f"Fraud: {fraud_count:,} | "
            f"Legit: {legit_count:,} | "
            f"Ratio: 1:{round(legit_count/fraud_count)}"
        )

        return X, y

    def _split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series]:
        """
        Split into training and validation sets.
        Stratified split ensures fraud cases in both sets.
        Uses 80/20 split with fixed seed for reproducibility.
        """
        logger.info("Splitting into train/validation...")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=self.config.data.random_seed,
            stratify=y,   # preserve fraud ratio in both splits
        )

        logger.info(
            f"Train: {len(X_train):,} rows "
            f"({int(y_train.sum())} fraud)"
        )
        logger.info(
            f"Val  : {len(X_val):,} rows "
            f"({int(y_val.sum())} fraud)"
        )
        return X_train, X_val, y_train, y_val

    def _apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to training data only.

        SMOTE = Synthetic Minority Oversampling TEchnique.
        Creates synthetic fraud cases by interpolating between
        existing fraud cases in feature space.

        CRITICAL: Never apply SMOTE to validation or test data.
        Applying it to test data would make results look better
        than real-world performance — data leakage.

        We target 10% fraud ratio after SMOTE — not 50/50.
        Extreme oversampling to 50/50 creates unrealistic
        synthetic samples and hurts real-world performance.
        """
        fraud_before = int(y_train.sum())
        legit_before = int((y_train == 0).sum())

        logger.info(
            f"Applying SMOTE... "
            f"(fraud before: {fraud_before:,})"
        )

        # sampling_strategy=0.1 means fraud will be 10% of legit
        # e.g. 182,000 legit → target 18,200 fraud samples
        smote = SMOTE(
            sampling_strategy=0.1,
            random_state=self.config.data.random_seed,
            k_neighbors=5,
        )

        X_balanced, y_balanced = smote.fit_resample(
            X_train, y_train
        )

        fraud_after = int(y_balanced.sum())
        logger.info(
            f"✓ SMOTE complete: "
            f"fraud {fraud_before:,} → {fraud_after:,} "
            f"(+{fraud_after - fraud_before:,} synthetic samples)"
        )

        return X_balanced, y_balanced

    def _setup_mlflow(self) -> None:
        """Configure MLflow experiment."""
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(
            self.config.model.experiment_name
        )
        logger.info(
            f"MLflow experiment: "
            f"{self.config.model.experiment_name}"
        )

    def _compute_scale_pos_weight(
        self,
        y_train: pd.Series,
    ) -> float:
        """
        Compute scale_pos_weight from actual training distribution.
        = number of legitimate / number of fraud cases
        Tells XGBoost how much more to penalise fraud misses.
        """
        fraud = int(y_train.sum())
        legit = int((y_train == 0).sum())
        weight = legit / fraud
        logger.info(
            f"scale_pos_weight = {legit:,} / {fraud:,} "
            f"= {weight:.1f}"
        )
        return weight

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
    ) -> EvaluationResult:
        """
        Train XGBoost with MLflow autologging.

        scale_pos_weight handles remaining imbalance after SMOTE.
        eval_metric=aucpr trains the model optimising for
        PR-AUC directly — consistent with our evaluation metric.
        """
        logger.info("Training XGBoost...")

        scale_pos_weight = self._compute_scale_pos_weight(
            y_train
        )

        params = {
            **self.config.model.params,
            "scale_pos_weight": scale_pos_weight,
            "use_label_encoder": False,
            "random_state": self.config.data.random_seed,
        }

        with mlflow.start_run(run_name="xgboost"):
            # autolog captures params, metrics, model automatically
            mlflow.xgboost.autolog(log_models=True)

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Get fraud probabilities for evaluation
            y_prob = model.predict_proba(X_val)[:, 1]

            # Evaluate
            evaluator = ModelEvaluator("XGBoost")
            result    = evaluator.evaluate(
                y_val.values, y_prob
            )

            # Log our custom metrics to MLflow
            mlflow.log_metrics(result.to_dict())
            mlflow.log_param(
                "smote_sampling_strategy", 0.1
            )

            # Log feature importance
            importance = dict(zip(
                X_train.columns,
                model.feature_importances_
            ))
            top_features = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            logger.info("Top 10 XGBoost features:")
            for feat, imp in top_features:
                logger.info(f"  {feat:<35} {imp:.4f}")

        return result

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
    ) -> EvaluationResult:
        """
        Train LightGBM as challenger model.

        LightGBM uses is_unbalance=True instead of
        scale_pos_weight — equivalent concept, different API.
        Leaf-wise growth often beats XGBoost on tabular data.
        """
        logger.info("Training LightGBM...")

        params = {
            "n_estimators":    self.config.model.params.get(
                "n_estimators", 300
            ),
            "max_depth":       self.config.model.params.get(
                "max_depth", 6
            ),
            "learning_rate":   self.config.model.params.get(
                "learning_rate", 0.05
            ),
            "subsample":       self.config.model.params.get(
                "subsample", 0.8
            ),
            "is_unbalance":    True,
            "metric":          "average_precision",
            "random_state":    self.config.data.random_seed,
            "verbosity":       -1,   # suppress LightGBM output
        }

        with mlflow.start_run(run_name="lightgbm"):
            mlflow.lightgbm.autolog(log_models=True)

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )

            y_prob = model.predict_proba(X_val)[:, 1]

            evaluator = ModelEvaluator("LightGBM")
            result    = evaluator.evaluate(
                y_val.values, y_prob
            )

            mlflow.log_metrics(result.to_dict())
            mlflow.log_param("smote_sampling_strategy", 0.1)

            # Log feature importance
            importance = dict(zip(
                X_train.columns,
                model.feature_importances_
            ))
            top_features = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            logger.info("Top 10 LightGBM features:")
            for feat, imp in top_features:
                logger.info(f"  {feat:<35} {imp:.4f}")

        return result

    def _register_best_model(
        self,
        winner: EvaluationResult,
    ) -> None:
        """
        Register the winning model in MLflow Model Registry.

        Model moves through: None → Staging → Production
        In Phase 4 we'll add the Staging → Production promotion
        with additional validation gates.
        """
        logger.info(
            f"Registering {winner.model_name} "
            f"in MLflow Model Registry..."
        )

        model_name = self.config.model.name

        # Find the run that produced the winning model
        experiment=mlflow.get_experiment_by_name(self.config.model.experiment_name)
        client = mlflow.tracking.MlflowClient()
        if experiment is None:
            logger.warning("Experiment not found-skipping registration")
            return
        
        run_name = winner.model_name.lower()

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(f"tags.mlflow.runName='{run_name}'"),
            order_by=["metrics.aucPR desc"],
            max_results=1,
        )    
        if not runs:
            logger.warning(
                f"No runs found for {run_name} — "
                f"skipping registration"
            )
            return

        run_id   = runs[0].info.run_id
        logger.info(f"Found run: {run_id}")
        artifacts = client.list_artifacts(run_id)
            logger.info(
                f"Artifacts in run: "
                f"{[a.path for a in artifacts]}"
            )
        model_path=None
        for candidate in [run_name,"model",winner.model_name]:
            matching = [
            a for a in artifacts
            if a.path == candidate and a.is_dir
            ]
            if matching:
                model_path = candidate
                break

        if model_path is None:
            # Fallback — use first directory artifact found
            dirs = [a for a in artifacts if a.is_dir]
            if dirs:
                model_path = dirs[0].path
                logger.info(
                    f"Using first artifact directory: {model_path}"
                )
        else:
                logger.warning(
                    "No model artifact found — skipping registration"
                )
                return

    model_uri = f"runs:/{run_id}/{model_path}"
    logger.info(f"Registering from URI: {model_uri}")

    try:
        mv = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        logger.info(
            f"✓ Registered: {model_name} "
            f"version {mv.version}"
        )

        # Transition to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )

        logger.info(
            f"✓ Promoted to Staging: "
            f"{model_name} v{mv.version}"
        )

    except Exception as e:
        logger.warning(
            f"Model registration failed: {e} — "
            f"model trained successfully, "
            f"registration can be done manually in MLflow UI"
        )
