# tests/test_trainer.py
"""
Unit tests for training module.
PySpark mocked via conftest.py.
ML libraries mocked to avoid installing them in CI.
"""
import sys
import numpy as np
from unittest.mock import MagicMock, patch

# Mock ML libraries for CI
for mod in [
    "mlflow", "mlflow.xgboost", "mlflow.lightgbm",
    "xgboost", "lightgbm",
    "imblearn", "imblearn.over_sampling",
    "sklearn", "sklearn.model_selection",
    "sklearn.metrics",
]:
    sys.modules[mod] = MagicMock()

from fraud_detection.training.evaluator import (   # noqa
    ModelEvaluator,
    EvaluationResult,
)


def make_eval_result(
    name: str,
    pr_auc: float,
    recall: float = 0.8,
) -> EvaluationResult:
    return EvaluationResult(
        model_name=      name,
        pr_auc=          pr_auc,
        roc_auc=         0.95,
        f1=              0.75,
        precision=       0.70,
        recall=          recall,
        threshold=       0.25,
        true_positives=  80,
        false_positives= 34,
        false_negatives= 20,
        true_negatives=  56813,
    )


class TestEvaluationResult:

    def test_instantiates(self):
        r = make_eval_result("XGBoost", 0.85)
        assert r is not None

    def test_repr_contains_model_name(self):
        r = make_eval_result("XGBoost", 0.85)
        assert "XGBoost" in repr(r)

    def test_repr_contains_pr_auc(self):
        r = make_eval_result("XGBoost", 0.85)
        assert "0.8500" in repr(r)

    def test_to_dict_contains_pr_auc(self):
        r = make_eval_result("XGBoost", 0.85)
        d = r.to_dict()
        assert "pr_auc" in d
        assert d["pr_auc"] == 0.85

    def test_to_dict_contains_all_keys(self):
        r    = make_eval_result("XGBoost", 0.85)
        d    = r.to_dict()
        keys = [
            "pr_auc", "roc_auc", "f1",
            "precision", "recall", "threshold",
            "true_positives", "false_positives",
            "false_negatives", "true_negatives",
        ]
        for key in keys:
            assert key in d

    def test_fraud_catch_rate(self):
        r = make_eval_result("XGBoost", 0.85)
        # tp=80, fn=20 → catch rate = 80/100 = 0.8
        assert r.fraud_catch_rate() == 0.8

    def test_false_alarm_rate(self):
        r = make_eval_result("XGBoost", 0.85)
        # fp=34, tn=56813 → rate = 34/56847
        expected = 34 / (34 + 56813)
        assert abs(r.false_alarm_rate() - expected) < 1e-6

    def test_fraud_catch_rate_zero_when_no_fraud(self):
        r = EvaluationResult(
            model_name="test", pr_auc=0.5, roc_auc=0.5,
            f1=0.0, precision=0.0, recall=0.0,
            threshold=0.5, true_positives=0,
            false_positives=0, false_negatives=0,
            true_negatives=1000,
        )
        assert r.fraud_catch_rate() == 0.0


class TestModelEvaluator:

    def test_instantiates(self):
        assert ModelEvaluator("XGBoost") is not None

    def test_compare_returns_higher_pr_auc(self):
        evaluator = ModelEvaluator("test")
        result_a  = make_eval_result("XGBoost",  0.85)
        result_b  = make_eval_result("LightGBM", 0.88)
        winner    = evaluator.compare(result_a, result_b)
        assert winner.model_name == "LightGBM"

    def test_compare_returns_a_when_higher(self):
        evaluator = ModelEvaluator("test")
        result_a  = make_eval_result("XGBoost",  0.90)
        result_b  = make_eval_result("LightGBM", 0.85)
        winner    = evaluator.compare(result_a, result_b)
        assert winner.model_name == "XGBoost"

    def test_compare_tiebreaker_is_recall(self):
        evaluator = ModelEvaluator("test")
        # Same PR-AUC — higher recall wins
        result_a  = make_eval_result("XGBoost",  0.85, recall=0.90)
        result_b  = make_eval_result("LightGBM", 0.85, recall=0.75)
        winner    = evaluator.compare(result_a, result_b)
        assert winner.model_name == "XGBoost"

    def test_compare_tiebreaker_recall_b_wins(self):
        evaluator = ModelEvaluator("test")
        result_a  = make_eval_result("XGBoost",  0.85, recall=0.70)
        result_b  = make_eval_result("LightGBM", 0.85, recall=0.85)
        winner    = evaluator.compare(result_a, result_b)
        assert winner.model_name == "LightGBM"