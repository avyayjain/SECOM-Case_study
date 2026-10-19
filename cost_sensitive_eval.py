"""
Lesson 22: Cost-sensitive model evaluation for the SECOM dataset.

Run from your project root:
    python 22_cost_sensitive_evaluation.py

This script repeats the same nested CV design for Logistic Regression,
Random Forest, and XGBoost. For each outer test fold, it:
1) Gets out-of-fold probabilities from the outer training data.
2) Chooses the threshold minimizing an assumed FN:FP cost ratio.
3) Evaluates that threshold on the untouched outer test fold.
4) Also evaluates a fixed threshold of 0.5 for reference.

Important:
- Costs are illustrative units, NOT actual semiconductor manufacturing costs.
- Cost-sensitive threshold selection is done using training-only OOF predictions.
- The same outer folds are used for every model and cost ratio.
- XGBoost uses CUDA when USE_GPU_XGBOOST=True.
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
)

# ---------------------------- Configuration ----------------------------

RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4
USE_GPU_XGBOOST = True
N_JOBS = 4

# Illustrative relative costs: one FN costs 5, 10, or 20 times one FP.
# Replace these with estimates from manufacturing stakeholders when available.
FN_TO_FP_COST_RATIOS = [5, 10, 20]

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "secom.data"
LABEL_FILE = DATA_DIR / "secom_labels.data"


class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=float)
        self.keep_mask_ = ~np.isnan(X_array).all(axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]


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
                device="cuda" if USE_GPU_XGBOOST else "cpu",
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
            )),
        ])
    except ImportError:
        print("WARNING: xgboost is not installed; XGBoost will be skipped.")

    return models


def choose_cost_threshold(y_true, probabilities, fn_cost, fp_cost=1.0):
    """
    Minimize mean empirical classification cost over candidate thresholds.
    Include thresholds outside the probability range so all-pass/all-fault
    predictions remain possible. This threshold is selected only on training OOF.
    """
    candidates = np.unique(np.concatenate([
        np.array([0.0, 0.5, 1.0]),
        probabilities,
        np.nextafter(probabilities, np.inf),
    ]))
    best_threshold = 0.5
    best_cost = float("inf")
    y_true = np.asarray(y_true)

    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        cost = (fn_cost * fn + fp_cost * fp) / len(y_true)
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)

    return best_threshold


def metrics_for_threshold(y_true, probabilities, threshold, fn_cost, fp_cost=1.0):
    pred = (probabilities >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    tn = int(np.sum((y_true == 0) & (pred == 0)))

    return {
        "average_precision": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "cost_per_sample": (fn_cost * fn + fp_cost * fp) / len(y_true),
        "total_cost": fn_cost * fn + fp_cost * fp,
        "test_samples": len(y_true),
        "faults_in_test": int(np.sum(y_true == 1)),
    }


def main():
    if not DATA_FILE.exists() or not LABEL_FILE.exists():
        raise FileNotFoundError(
            "Could not find data/secom.data and data/secom_labels.data. "
            "Run this script from your SECOM-Case_study project root."
        )

    X = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, na_values=["NaN"])
    labels = pd.read_csv(LABEL_FILE, sep=r"\s+", header=None)
    y = (labels.iloc[:, 0].to_numpy() == 1).astype(int)

    print(f"Dataset: {X.shape}")
    print(f"Faults: {int(y.sum())}; fault rate: {y.mean():.2%}")
    print(f"FN:FP cost ratios: {FN_TO_FP_COST_RATIOS}")
    print("Costs are hypothetical relative units, not real factory costs.")
    print(f"GPU XGBoost requested: {USE_GPU_XGBOOST}")

    models = make_models()
    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    outer_splits = list(outer_cv.split(X, y))
    rows = []
    started = time.time()

    for model_name, estimator in models.items():
        print(f"\n{'=' * 68}\nMODEL: {model_name}\n{'=' * 68}")

        for fold_number, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            inner_cv = StratifiedKFold(
                n_splits=INNER_FOLDS,
                shuffle=True,
                random_state=RANDOM_STATE + fold_number,
            )

            fold_started = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*mismatched devices.*")
                oof_probabilities = cross_val_predict(
                    clone(estimator),
                    X_train,
                    y_train,
                    cv=inner_cv,
                    method="predict_proba",
                    n_jobs=1,
                )[:, 1]

            fitted_model = clone(estimator)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*mismatched devices.*")
                fitted_model.fit(X_train, y_train)
                test_probabilities = fitted_model.predict_proba(X_test)[:, 1]

            for ratio in FN_TO_FP_COST_RATIOS:
                threshold = choose_cost_threshold(
                    y_train, oof_probabilities, fn_cost=ratio, fp_cost=1
                )
                metrics = metrics_for_threshold(
                    y_test, test_probabilities, threshold, fn_cost=ratio, fp_cost=1
                )
                metrics.update({
                    "model": model_name,
                    "fold": fold_number,
                    "fn_to_fp_cost_ratio": ratio,
                    "threshold_source": "inner out-of-fold predictions",
                    "runtime_seconds": round(time.time() - fold_started, 2),
                })
                rows.append(metrics)

            # Fixed 0.5 baseline for every model, every outer fold and cost ratio.
            for ratio in FN_TO_FP_COST_RATIOS:
                metrics = metrics_for_threshold(
                    y_test, test_probabilities, 0.5, fn_cost=ratio, fp_cost=1
                )
                metrics.update({
                    "model": model_name,
                    "fold": fold_number,
                    "fn_to_fp_cost_ratio": ratio,
                    "threshold_source": "fixed 0.5 reference",
                    "runtime_seconds": round(time.time() - fold_started, 2),
                })
                rows.append(metrics)

            print(
                f"Fold {fold_number}: test faults={int(y_test.sum())}; "
                f"training models + thresholds completed in "
                f"{time.time() - fold_started:.1f}s"
            )

    fold_df = pd.DataFrame(rows)
    fold_path = RESULTS_DIR / "lesson22_cost_sensitive_folds.csv"
    fold_df.to_csv(fold_path, index=False)

    summary = (
        fold_df.groupby(
            ["model", "fn_to_fp_cost_ratio", "threshold_source"], sort=False
        )
        .agg(
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", "std"),
            cost_per_sample_mean=("cost_per_sample", "mean"),
            cost_per_sample_std=("cost_per_sample", "std"),
            mean_fn=("fn", "mean"),
            mean_fp=("fp", "mean"),
        )
        .reset_index()
    )
    summary_path = RESULTS_DIR / "lesson22_cost_sensitive_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nCOST-SENSITIVE SUMMARY")
    print(summary.round(3).to_string(index=False))
    print(f"\nTotal runtime: {time.time() - started:.1f}s")
    print(f"Fold results saved: {fold_path}")
    print(f"Summary saved: {summary_path}")
    print("\nInterpretation:")
    print("- Lower cost_per_sample_mean is better under the assumed cost ratio.")
    print("- Compare tuned thresholds with fixed 0.5 references.")
    print("- These are relative illustrative costs, not monetary estimates.")
    print("- A model's chosen threshold can change substantially by fold.")


if __name__ == "__main__":
    main()
