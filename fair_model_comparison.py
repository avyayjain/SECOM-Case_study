"""
Lesson 21: Fair nested-CV comparison of Logistic Regression, Random Forest, and XGBoost.

Expected project layout:
SECOM-Case_study/
  data/
    secom.data
    secom_labels.data
  results/
    tables/

Run from the project root:
    python 21_fair_model_comparison.py

The script:
- Uses identical outer stratified folds for all models.
- Selects the F1 threshold using out-of-fold predictions from the outer training set only.
- Keeps preprocessing inside each pipeline/fold to prevent leakage.
- Saves fold-level and summary CSV files under results/tables/.

XGBoost uses the GPU when available. If CUDA setup fails, set USE_GPU_XGBOOST = False.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)

# ---------------------------- Configuration ----------------------------

RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4
USE_GPU_XGBOOST = True
N_JOBS = 4

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "secom.data"
LABEL_FILE = DATA_DIR / "secom_labels.data"


# ----------------------- Leakage-safe preprocessing --------------------

class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    """Drop columns that are entirely missing, learning the mask on fit data."""
    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=float)
        self.keep_mask_ = ~np.isnan(X_array).all(axis=0)
        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=float)
        return X_array[:, self.keep_mask_]


def make_models():
    models = {
        "Logistic Regression": Pipeline([
            ("drop_empty", DropAllMissingFeatures()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": Pipeline([
            ("drop_empty", DropAllMissingFeatures()),
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    try:
        from xgboost import XGBClassifier
        xgb_device = "cuda" if USE_GPU_XGBOOST else "cpu"
        models["XGBoost"] = Pipeline([
            ("drop_empty", DropAllMissingFeatures()),
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_lambda=1.0,
                scale_pos_weight=13.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device=xgb_device,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
            )),
        ])
    except ImportError:
        print("WARNING: xgboost is not installed; XGBoost will be skipped.")

    return models


def choose_f1_threshold(y_true, probabilities):
    """Choose a threshold from training-only OOF probabilities."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "average_precision": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "threshold": threshold,
        "faults_in_test": int(np.sum(y_true == 1)),
        "predicted_faults": int(np.sum(predictions == 1)),
    }


def main():
    if not DATA_FILE.exists() or not LABEL_FILE.exists():
        raise FileNotFoundError(
            "Could not find data/secom.data and data/secom_labels.data. "
            "Run this script from your SECOM-Case_study project root."
        )

    print("Loading SECOM dataset...")
    X = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, na_values=["NaN"])
    labels = pd.read_csv(LABEL_FILE, sep=r"\s+", header=None)
    # UCI labels: -1 = pass, +1 = fault. Convert fault to positive class 1.
    y = (labels.iloc[:, 0].to_numpy() == 1).astype(int)

    print(f"Dataset: {X.shape}")
    print(f"Faults: {int(y.sum())}")
    print(f"Fault rate: {y.mean():.2%}")
    print(f"GPU XGBoost requested: {USE_GPU_XGBOOST}")
    print(f"Outer folds: {OUTER_FOLDS}; inner folds: {INNER_FOLDS}")
    print("\nNote: XGBoost may print a device-mismatch prediction warning. "
          "This can slow prediction but does not automatically invalidate results.\n")

    models = make_models()
    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    rows = []
    overall_start = time.time()

    # Materialize splits once so every model gets exactly the same folds.
    outer_splits = list(outer_cv.split(X, y))

    for model_name, estimator in models.items():
        print(f"\n{'=' * 72}\nMODEL: {model_name}\n{'=' * 72}")

        for fold_number, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            fold_start = time.time()
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            inner_cv = StratifiedKFold(
                n_splits=INNER_FOLDS,
                shuffle=True,
                random_state=RANDOM_STATE + fold_number,
            )

            # Each inner validation prediction comes from a model whose fitting
            # and preprocessing used only that inner fold's training partition.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*mismatched devices.*",
                    category=UserWarning,
                )
                oof_probabilities = cross_val_predict(
                    clone(estimator),
                    X_train,
                    y_train,
                    cv=inner_cv,
                    method="predict_proba",
                    n_jobs=1,  # avoid parallel GPU models competing for 2 GB VRAM
                )[:, 1]

            threshold = choose_f1_threshold(y_train, oof_probabilities)

            fitted_model = clone(estimator)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*mismatched devices.*",
                    category=UserWarning,
                )
                fitted_model.fit(X_train, y_train)
                test_probabilities = fitted_model.predict_proba(X_test)[:, 1]

            metrics = calculate_metrics(y_test, test_probabilities, threshold)
            metrics.update({
                "model": model_name,
                "fold": fold_number,
                "fit_threshold_source": "inner out-of-fold predictions",
                "runtime_seconds": round(time.time() - fold_start, 2),
            })
            rows.append(metrics)

            print(
                f"Fold {fold_number}: threshold={threshold:.4f} | "
                f"AP={metrics['average_precision']:.3f} | "
                f"ROC-AUC={metrics['roc_auc']:.3f} | "
                f"Recall={metrics['recall']:.3f} | "
                f"F1={metrics['f1']:.3f} | "
                f"faults in test={metrics['faults_in_test']} | "
                f"time={metrics['runtime_seconds']}s"
            )

    fold_df = pd.DataFrame(rows)
    fold_path = RESULTS_DIR / "lesson21_fair_comparison_folds.csv"
    fold_df.to_csv(fold_path, index=False)

    metric_names = [
        "average_precision", "roc_auc", "balanced_accuracy",
        "precision", "recall", "f1", "threshold", "runtime_seconds",
    ]
    summary_rows = []
    for model_name, group in fold_df.groupby("model", sort=False):
        row = {"model": model_name}
        for metric in metric_names:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "lesson21_fair_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    display_cols = [
        "model",
        "average_precision_mean", "average_precision_std",
        "roc_auc_mean", "roc_auc_std",
        "balanced_accuracy_mean", "balanced_accuracy_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "f1_mean", "f1_std",
        "threshold_mean", "threshold_std",
    ]
    print("\n\nFAIR COMPARISON SUMMARY (mean ± standard deviation)")
    print(summary_df[display_cols].round(3).to_string(index=False))
    print(f"\nTotal runtime: {time.time() - overall_start:.1f}s")
    print(f"Fold-level results saved: {fold_path}")
    print(f"Summary saved: {summary_path}")
    print("\nInterpretation reminder:")
    print("- Average precision and ROC-AUC evaluate ranking across thresholds.")
    print("- Precision, recall, F1 and balanced accuracy use the inner-CV tuned threshold.")
    print("- Thresholds are selected without using the corresponding outer test fold.")
    print("- Five folds with few fault cases produce noisy estimates; don't overinterpret small gaps.")


if __name__ == "__main__":
    main()
