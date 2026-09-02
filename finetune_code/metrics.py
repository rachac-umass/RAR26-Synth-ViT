import warnings

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

_NAN_METRICS = {
    "AUROC": float("nan"),
    "AUPRC": float("nan"),
    "PPV@90% Recall": float("nan"),
    "PPV": float("nan"),
    "Accuracy": float("nan"),
    "Sensitivity": float("nan"),
    "Specificity": float("nan"),
    "PPV+PPV@90% Recall": float("nan"),
    "PPV@90% Recall+AUPRC": float("nan"),
}


def compute_metrics(y_true, y_scores):
    """Computes AUROC, AUPRC, PPV@90% Recall, PPV, Accuracy, Sensitivity, Specificity,
    PPV+PPV@90% Recall, and PPV@90% Recall+AUPRC (sums of metric pairs, for use as
    combined model-selection criteria)."""

    if not np.all(np.isfinite(y_scores)):
        n_bad = np.sum(~np.isfinite(y_scores))
        warnings.warn(
            f"compute_metrics received {n_bad}/{len(y_scores)} non-finite scores "
            "(NaN/Inf) -- likely diverged training (exploding gradients). "
            "Returning NaN metrics for this fold instead of crashing.",
            RuntimeWarning,
        )
        return dict(_NAN_METRICS)

    # AUROC & AUPRC
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    # Compute Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Find PPV @ 90% Recall
    idx = np.where(recalls >= 0.9)[0][-1]
    ppv_at_0_9_recall = precisions[idx]

    # Convert scores to binary predictions (threshold at 0.5)
    y_pred = (y_scores >= 0.5).astype(int)

    # Compute Accuracy, Sensitivity, and Specificity
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "PPV@90% Recall": ppv_at_0_9_recall,
        "PPV": ppv,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV+PPV@90% Recall": ppv + ppv_at_0_9_recall,
        "PPV@90% Recall+AUPRC": ppv_at_0_9_recall + auprc,
    }