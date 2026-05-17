"""HHEM threshold calibration using RAGTruth labels.

Fits the threshold that maximises F1 on the calibration split, plots an ROC curve,
saves both to /data/results/.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def fit_threshold(
    scores: list[float],
    labels: list[int],
    output_dir: str = "/data/results",
) -> dict:
    """Find best threshold by F1, save curve, return summary.

    `labels`: 1 if the response is hallucinated (should be flagged), 0 if faithful.
    `scores`: HHEM faithfulness score in [0, 1], higher = more faithful.
    A response is flagged when score < threshold. So we want the threshold above
    which most scores are faithful (label=0).
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)

    # We flip the score: "flag-positive" if low score. roc_curve expects scores
    # that are HIGHER for positive class.
    flag_score = 1.0 - s
    fpr, tpr, thresholds = roc_curve(y, flag_score)
    auc = float(roc_auc_score(y, flag_score))

    # Best F1 over thresholds
    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for thr in thresholds:
        predicted_flag = (flag_score >= thr).astype(int)
        tp = int(((predicted_flag == 1) & (y == 1)).sum())
        fp = int(((predicted_flag == 1) & (y == 0)).sum())
        fn = int(((predicted_flag == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best["f1"]:
            # Convert the flag_score threshold back to the original-score threshold
            best = {
                "threshold": float(1.0 - thr),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ROC plot
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"HHEM (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("HHEM hallucination detection — ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration_curve.png", dpi=150)
    plt.close()

    summary = {**best, "auc": auc, "n_samples": int(len(y))}
    with open(f"{output_dir}/threshold.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
