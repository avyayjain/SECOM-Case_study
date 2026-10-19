
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import xgboost

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# ============================================================
# 1. Configuration
# ============================================================

RANDOM_STATE = 42
OUTER_FOLDS = 5
CALIBRATION_FOLDS = 3

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("scikit-learn:", sklearn.__version__)
print("XGBoost:", xgboost.__version__)


# ============================================================
# 2. Load the SECOM dataset
# ============================================================

X = pd.read_csv(
    DATA_DIR / "secom.data",
    sep=r"\s+",
    header=None,
    na_values="NaN",
)

labels = pd.read_csv(
    DATA_DIR / "secom_labels.data",
    sep=r"\s+",
    header=None,
)

# Original labels: -1 = pass, +1 = fault
y = (labels.iloc[:, 0].to_numpy() == 1).astype(int)

X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

print("\nDataset shape:", X.shape)
print("Fault samples:", int(y.sum()))
print("Passing samples:", int((y == 0).sum()))
print("Fault rate:", round(y.mean() * 100, 2), "%")


# ============================================================
# 3. Define models
# ============================================================

def make_model(model_name, y_train):
    """
    Create a fresh model using only the supplied training labels
    to calculate class imbalance.
    """

    if model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "XGBoost":
        positives = int(np.sum(y_train == 1))
        negatives = int(np.sum(y_train == 0))

        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=negatives / max(positives, 1),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
            n_jobs=4,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Fit imputation only on each training partition.
    # This also handles missing sensor readings.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier),
    ])


# ============================================================
# 4. Outer cross-validation
# ============================================================

models = ["Random Forest", "XGBoost"]
records = []

outer_cv = StratifiedKFold(
    n_splits=OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y), start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"\n{'=' * 60}")
    print(f"OUTER FOLD {fold}/{OUTER_FOLDS}")
    print("Training faults:", int(y_train.sum()))
    print("Test faults:", int(y_test.sum()))

    for model_name in models:
        print(f"\nModel: {model_name}")
        start = time.time()

        # ----------------------------------------------------
        # A. Ordinary, uncalibrated model
        # ----------------------------------------------------

        base_model = make_model(model_name, y_train)
        base_model.fit(X_train, y_train)

        raw_probabilities = base_model.predict_proba(X_test)[:, 1]

        # ----------------------------------------------------
        # B. Calibrated model
        #
        # Calibration CV runs only within the outer training
        # data. The outer test fold remains untouched.
        # ----------------------------------------------------

        calibration_cv = StratifiedKFold(
            n_splits=CALIBRATION_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + fold,
        )

        calibrated_model = CalibratedClassifierCV(
            estimator=make_model(model_name, y_train),
            method="sigmoid",
            cv=calibration_cv,
            ensemble=True,
        )

        calibrated_model.fit(X_train, y_train)

        calibrated_probabilities = (
            calibrated_model.predict_proba(X_test)[:, 1]
        )

        # ----------------------------------------------------
        # C. Evaluate both probability sets on the same test fold
        # ----------------------------------------------------

        for calibration_status, probabilities in [
            ("Uncalibrated", raw_probabilities),
            ("Sigmoid calibrated", calibrated_probabilities),
        ]:
            record = {
                "fold": fold,
                "model": model_name,
                "calibration": calibration_status,
                "test_faults": int(y_test.sum()),
                "fault_rate": float(y_test.mean()),
                "mean_predicted_probability": float(
                    probabilities.mean()
                ),
                "brier_score": brier_score_loss(
                    y_test, probabilities
                ),
                "log_loss": log_loss(
                    y_test,
                    probabilities,
                    labels=[0, 1],
                ),
                "average_precision": average_precision_score(
                    y_test, probabilities
                ),
                "roc_auc": roc_auc_score(
                    y_test, probabilities
                ),
            }

            records.append(record)

            print(
                f"  {calibration_status:20s} | "
                f"Brier={record['brier_score']:.4f} | "
                f"Log loss={record['log_loss']:.4f} | "
                f"AP={record['average_precision']:.4f} | "
                f"ROC-AUC={record['roc_auc']:.4f}"
            )

        print(f"  Elapsed: {time.time() - start:.1f} seconds")


# ============================================================
# 5. Save fold-level results and summary
# ============================================================

fold_results = pd.DataFrame(records)

fold_path = TABLE_DIR / "lesson23_calibration_folds.csv"
fold_results.to_csv(fold_path, index=False)

metric_columns = [
    "brier_score",
    "log_loss",
    "average_precision",
    "roc_auc",
    "mean_predicted_probability",
]

summary = (
    fold_results
    .groupby(["model", "calibration"])[metric_columns]
    .agg(["mean", "std"])
    .round(4)
)

summary_path = TABLE_DIR / "lesson23_calibration_summary.csv"
summary.to_csv(summary_path)

print("\n\nSUMMARY: mean and standard deviation across outer folds")
print(summary.to_string())

print("\nSaved fold results:", fold_path)
print("Saved summary:", summary_path)


# ============================================================
# 6. Plot reliability diagrams from outer-fold predictions
#
# We will rerun the outer folds to collect each sample's
# held-out probabilities. Each sample is predicted by a model
# that did not train on that sample.
# ============================================================

oof_records = []

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y), start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    for model_name in models:
        base_model = make_model(model_name, y_train)
        base_model.fit(X_train, y_train)

        raw_probabilities = base_model.predict_proba(X_test)[:, 1]

        calibration_cv = StratifiedKFold(
            n_splits=CALIBRATION_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + fold,
        )

        calibrated_model = CalibratedClassifierCV(
            estimator=make_model(model_name, y_train),
            method="sigmoid",
            cv=calibration_cv,
            ensemble=True,
        )

        calibrated_model.fit(X_train, y_train)

        calibrated_probabilities = (
            calibrated_model.predict_proba(X_test)[:, 1]
        )

        for i, original_idx in enumerate(test_idx):
            oof_records.append({
                "sample_index": int(original_idx),
                "actual": int(y_test[i]),
                "model": model_name,
                "uncalibrated_probability": float(
                    raw_probabilities[i]
                ),
                "calibrated_probability": float(
                    calibrated_probabilities[i]
                ),
            })

oof = pd.DataFrame(oof_records)

oof_path = TABLE_DIR / "lesson23_calibration_oof_predictions.csv"
oof.to_csv(oof_path, index=False)

# Reliability diagrams
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, model_name in zip(axes, models):
    subset = oof[oof["model"] == model_name]
    actual = subset["actual"].to_numpy()

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for column, label in [
        ("uncalibrated_probability", "Uncalibrated"),
        ("calibrated_probability", "Sigmoid calibrated"),
    ]:
        observed_rate, mean_probability = calibration_curve(
            actual,
            subset[column].to_numpy(),
            n_bins=5,
            strategy="quantile",
        )

        ax.plot(
            mean_probability,
            observed_rate,
            marker="o",
            label=label,
        )

    ax.set_title(model_name)
    ax.set_xlabel("Mean predicted fault probability")
    ax.set_ylabel("Observed fault rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle("SECOM: Probability Calibration")
fig.tight_layout()

figure_path = FIGURE_DIR / "lesson23_reliability_diagrams.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()

print("\nSaved reliability diagram:", figure_path)
print("Saved out-of-fold probabilities:", oof_path)
print("\nLesson 23 complete.")
