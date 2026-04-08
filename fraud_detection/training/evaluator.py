# fraud_detection/training/evaluator.py
"""
Model evaluation for fraud detection.

Computes precision, recall, F1, PR-AUC and finds
the optimal classification threshold on the PR curve.

Separated from trainer deliberately — evaluation logic
is reusable across XGBoost, LightGBM, and any future model.
"""
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Any

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Holds all evaluation metrics for one model run.
    Stored in MLflow and used for model comparison.
    """
    model_name:     str
    pr_auc:         float   # primary metric
    roc_auc:        float   # secondary metric
    f1:             float
    precision:      float
    recall:         float
    threshold:      float   # optimal threshold found by PR curve
    true_positives:  int    # fraud correctly caught
    false_positives: int    # legitimate flagged as fraud
    false_negatives: int    # fraud missed
    true_negatives:  int    # legitimate correctly cleared

    def __repr__(self) -> str:
        return (
            f"EvaluationResult({self.model_name} | "
            f"PR-AUC={self.pr_auc:.4f} | "
            f"F1={self.f1:.4f} | "
            f"Recall={self.recall:.4f} | "
            f"Precision={self.precision:.4f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for MLflow logging."""
        return {
            "pr_auc":          self.pr_auc,
            "roc_auc":         self.roc_auc,
            "f1":              self.f1,
            "precision":       self.precision,
            "recall":          self.recall,
            "threshold":       self.threshold,
            "true_positives":  self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives":  self.true_negatives,
        }

    def fraud_catch_rate(self) -> float:
        """
        Percentage of actual fraud cases caught.
        Same as recall — named explicitly for business reporting.
        """
        total_fraud = self.true_positives + self.false_negatives
        if total_fraud == 0:
            return 0.0
        return self.true_positives / total_fraud

    def false_alarm_rate(self) -> float:
        """
        Percentage of legitimate transactions wrongly flagged.
        A fraud team KPI — too many false alarms burns analyst time.
        """
        total_legit = self.false_positives + self.true_negatives
        if total_legit == 0:
            return 0.0
        return self.false_positives / total_legit


class ModelEvaluator:
    """
    Evaluates a trained model on the test set.

    Finds the optimal threshold on the Precision-Recall curve
    rather than using the default 0.5 — critical for imbalanced
    fraud detection where 0.5 is almost always wrong.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob:  np.ndarray,
    ) -> EvaluationResult:
        """
        Evaluate model predictions against ground truth.

        Args:
            y_true: actual labels (0=legit, 1=fraud)
            y_prob: predicted probabilities of fraud (0.0 to 1.0)

        Returns:
            EvaluationResult with all metrics computed
        """
        logger.info(f"Evaluating {self.model_name}...")

        # Core metrics
        pr_auc  = average_precision_score(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        # Find optimal threshold
        threshold = self._find_optimal_threshold(y_true, y_prob)

        # Apply threshold to get binary predictions
        y_pred = (y_prob >= threshold).astype(int)

        # Compute metrics at optimal threshold
        precision = precision_score(
            y_true, y_pred, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, zero_division=0
        )

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        result = EvaluationResult(
            model_name=     self.model_name,
            pr_auc=         pr_auc,
            roc_auc=        roc_auc,
            f1=             f1,
            precision=      precision,
            recall=         recall,
            threshold=      threshold,
            true_positives=  int(tp),
            false_positives= int(fp),
            false_negatives= int(fn),
            true_negatives=  int(tn),
        )

        self._log_result(result)
        return result

    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob:  np.ndarray,
    ) -> float:
        """
        Find the threshold that maximises F1 score on the PR curve.

        Why not use 0.5?
        At 0.173% fraud rate, probability scores cluster near 0.
        The model may assign 0.3 to fraud cases — still much higher
        than the 0.001 it assigns to obvious legitimate transactions.
        A 0.5 cutoff would miss all of them.

        We sweep all thresholds and pick the one with best F1.
        """
        precisions, recalls, thresholds = precision_recall_curve(
            y_true, y_prob
        )

        # F1 at each threshold point on the curve
        # Add small epsilon to avoid divide by zero
        f1_scores = (
            2 * precisions * recalls
            / (precisions + recalls + 1e-8)
        )

        # Best F1 index
        # Note: precision_recall_curve returns n+1 points
        # but only n thresholds — slice to match
        best_idx = np.argmax(f1_scores[:-1])
        optimal  = float(thresholds[best_idx])

        logger.info(
            f"  Optimal threshold: {optimal:.4f} "
            f"(F1={f1_scores[best_idx]:.4f} at this point)"
        )
        return optimal

    def _log_result(self, result: EvaluationResult) -> None:
        """Log evaluation results cleanly."""
        logger.info("=" * 55)
        logger.info(f"  {result.model_name} — Evaluation Results")
        logger.info("=" * 55)
        logger.info(f"  PR-AUC    : {result.pr_auc:.4f}  ← primary metric")
        logger.info(f"  ROC-AUC   : {result.roc_auc:.4f}")
        logger.info(f"  F1        : {result.f1:.4f}")
        logger.info(f"  Precision : {result.precision:.4f}")
        logger.info(f"  Recall    : {result.recall:.4f}")
        logger.info(f"  Threshold : {result.threshold:.4f}")
        logger.info("  ---")
        logger.info(f"  Fraud caught    : {result.true_positives}"
                    f" / {result.true_positives + result.false_negatives}"
                    f" ({result.fraud_catch_rate():.1%})")
        logger.info(f"  False alarms    : {result.false_positives}"
                    f" ({result.false_alarm_rate():.3%} of legit txns)")
        logger.info(f"  Fraud missed    : {result.false_negatives}")
        logger.info("=" * 55)

    def compare(
        self,
        result_a: EvaluationResult,
        result_b: EvaluationResult,
    ) -> EvaluationResult:
        """
        Compare two models and return the better one.
        Primary metric: PR-AUC.
        Tiebreaker: Recall (catching more fraud is the priority).
        """
        logger.info("Comparing models...")
        logger.info(
            f"  {result_a.model_name}: PR-AUC={result_a.pr_auc:.4f}"
        )
        logger.info(
            f"  {result_b.model_name}: PR-AUC={result_b.pr_auc:.4f}"
        )

        if result_a.pr_auc > result_b.pr_auc:
            winner = result_a
        elif result_b.pr_auc > result_a.pr_auc:
            winner = result_b
        else:
            # Tiebreaker — higher recall wins
            winner = (
                result_a
                if result_a.recall >= result_b.recall
                else result_b
            )

        logger.info(f"  Winner: {winner.model_name}")
        return winner