"""
Model Evaluator and Regression Benchmark.
=========================================
Computes comprehensive evaluation metrics: accuracy, per-class precision,
recall, F1, confusion matrix, false-positive rates, and regression checks
against the production benchmark.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .config import VALID_ALPHABET_CLASSES


class EvaluationReport:
    """Structured container for evaluation and regression metrics."""

    def __init__(
        self,
        overall_accuracy: float,
        precision_macro: float,
        recall_macro: float,
        f1_macro: float,
        per_class_metrics: Dict[str, Dict[str, float]],
        confusion_matrix: List[List[int]],
        total_samples: int,
        regression_detected: bool,
        regression_details: List[str],
        classes: List[str],
    ):
        self.overall_accuracy = overall_accuracy
        self.precision_macro = precision_macro
        self.recall_macro = recall_macro
        self.f1_macro = f1_macro
        self.per_class_metrics = per_class_metrics
        self.confusion_matrix = confusion_matrix
        self.total_samples = total_samples
        self.regression_detected = regression_detected
        self.regression_details = regression_details
        self.classes = classes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_accuracy": round(self.overall_accuracy, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "f1_macro": round(self.f1_macro, 4),
            "per_class_metrics": self.per_class_metrics,
            "confusion_matrix": self.confusion_matrix,
            "total_samples": self.total_samples,
            "regression_detected": self.regression_detected,
            "regression_details": self.regression_details,
            "classes": self.classes,
        }


class ModelEvaluator:
    """Performs rigorous statistical and regression evaluation of models."""

    def __init__(self, classes: Optional[List[str]] = None):
        self.classes = classes or VALID_ALPHABET_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_report: Optional[EvaluationReport] = None,
    ) -> EvaluationReport:
        """Compute complete multi-class metrics from true vs predicted label arrays."""
        num_classes = len(self.classes)
        cm = np.zeros((num_classes, num_classes), dtype=int)

        # Standardize labels to indices
        true_indices = []
        pred_indices = []
        for yt, yp in zip(y_true, y_pred):
            yt_str = str(yt).upper()
            yp_str = str(yp).upper()
            if yt_str in self.class_to_idx and yp_str in self.class_to_idx:
                ti = self.class_to_idx[yt_str]
                pi = self.class_to_idx[yp_str]
                cm[ti, pi] += 1
                true_indices.append(ti)
                pred_indices.append(pi)

        total = len(true_indices)
        if total == 0:
            return EvaluationReport(
                overall_accuracy=0.0,
                precision_macro=0.0,
                recall_macro=0.0,
                f1_macro=0.0,
                per_class_metrics={},
                confusion_matrix=cm.tolist(),
                total_samples=0,
                regression_detected=False,
                regression_details=["No matching evaluation samples"],
                classes=self.classes,
            )

        correct = int(np.trace(cm))
        overall_accuracy = correct / total

        per_class: Dict[str, Dict[str, float]] = {}
        precisions = []
        recalls = []
        f1s = []

        for i, c in enumerate(self.classes):
            tp = cm[i, i]
            fp = int(np.sum(cm[:, i]) - tp)
            fn = int(np.sum(cm[i, :]) - tp)
            tn = total - (tp + fp + fn)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            per_class[c] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "false_positive_rate": round(fpr, 4),
                "support": int(tp + fn),
            }

            if (tp + fn) > 0:  # Only count classes present in test set for macro average
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)

        macro_p = float(np.mean(precisions)) if precisions else 0.0
        macro_r = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        # Regression check against baseline
        regression_detected = False
        reg_details = []

        if baseline_report is not None:
            if overall_accuracy < baseline_report.overall_accuracy - 0.02:
                regression_detected = True
                reg_details.append(
                    f"Overall accuracy drop: {overall_accuracy:.3f} vs baseline {baseline_report.overall_accuracy:.3f}"
                )
            for c in self.classes:
                base_cls = baseline_report.per_class_metrics.get(c, {})
                base_f1 = base_cls.get("f1_score", 0.0)
                cur_f1 = per_class[c]["f1_score"]
                if base_cls.get("support", 0) > 0 and cur_f1 < base_f1 - 0.10:
                    regression_detected = True
                    reg_details.append(f"Significant class regression for '{c}': F1 {cur_f1:.2f} vs baseline {base_f1:.2f}")

        return EvaluationReport(
            overall_accuracy=overall_accuracy,
            precision_macro=macro_p,
            recall_macro=macro_r,
            f1_macro=macro_f1,
            per_class_metrics=per_class,
            confusion_matrix=cm.tolist(),
            total_samples=total,
            regression_detected=regression_detected,
            regression_details=reg_details,
            classes=self.classes,
        )

    def evaluate_keras_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        baseline_report: Optional[EvaluationReport] = None,
    ) -> EvaluationReport:
        """Run predictions on Keras model and evaluate."""
        probs = model.predict(X_test, verbose=0)
        pred_indices = np.argmax(probs, axis=1)
        y_pred = [self.classes[idx] if idx < len(self.classes) else "--" for idx in pred_indices]
        return self.evaluate_predictions(y_test, np.array(y_pred), baseline_report=baseline_report)
