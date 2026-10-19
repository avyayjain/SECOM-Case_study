
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def load_data():
    X = pd.read_csv(
        DATA_DIR / "secom.data",
        sep=r"\s+",
        header=None,
    )
    labels = pd.read_csv(
        DATA_DIR / "secom_labels.data",
        sep=r"\s+",
        header=None,
    )

    y = labels.iloc[:, 0].map({-1: 0, 1: 1}).to_numpy()
    X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

    # Remove features that are completely missing globally.
    X = X.dropna(axis=1, how="all")

    return X, y


def make_model(y_train, seed):
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))

    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        scale_pos_weight=negatives / max(positives, 1),
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=4,
    )


def fit_and_predict(X_train, y_train, X_eval, seed):
    # Fit imputation only on this training partition.
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(X_train).astype(np.float32)
    eval_values = imputer.transform(X_eval).astype(np.float32)

    model = make_model(y_train, seed)
    model.fit(train_values, y_train, verbose=False)

    # Use the model's normal prediction interface. XGBoost
    # may emit a device-mismatch warning, but falls back safely.
    probabilities = model.predict_proba(eval_values)[:, 1]

    return probabilities, model


def choose_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    # The final precision/recall point has no matching threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = (
        2 * precision * recall
        / np.maximum(precision + recall, 1e-12)
    )

    if len(thresholds) == 0:
        return 0.5

    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index])


def main():
    X, y = load_data()

    print("XGBoost:", xgb.__version__)
    print("Dataset:", X.shape)
    print("Faults:", int(y.sum()))
    print("Fault rate:", f"{100 * y.mean():.2f}%")
    print("GPU: CUDA via XGBoost")

    outer_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )

    records = []
    all_start = time.perf_counter()

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        fold_start = time.perf_counter()

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Inner out-of-fold predictions for threshold selection.
        inner_cv = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=SEED + fold,
        )

        oof_probabilities = np.zeros(len(train_idx), dtype=float)

        for inner_train_idx, inner_valid_idx in inner_cv.split(
            X_train, y_train
        ):
            inner_probs, _ = fit_and_predict(
                X_train.iloc[inner_train_idx],
                y_train[inner_train_idx],
                X_train.iloc[inner_valid_idx],
                seed=SEED + fold,
            )
            oof_probabilities[inner_valid_idx] = inner_probs

        threshold = choose_f1_threshold(
            y_train, oof_probabilities
        )

        # Refit on the full outer training partition.
        test_probabilities, _ = fit_and_predict(
            X_train, y_train, X_test, seed=SEED + fold
        )

        for method, cutoff in [
            ("default_0.5", 0.5),
            ("tuned_f1", threshold),
        ]:
            predictions = (test_probabilities >= cutoff).astype(int)

            records.append({
                "fold": fold,
                "method": method,
                "threshold": cutoff,
                "average_precision": average_precision_score(
                    y_test, test_probabilities
                ),
                "roc_auc": roc_auc_score(
                    y_test, test_probabilities
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y_test, predictions
                ),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "seconds": round(
                    time.perf_counter() - fold_start, 2
                ),
            })

        print(
            f"Fold {fold}: threshold={threshold:.4f}; "
            f"faults in test={int(y_test.sum())}; "
            f"time={time.perf_counter() - fold_start:.1f}s"
        )

    results = pd.DataFrame(records)
    results.to_csv(
        RESULTS_DIR / "gpu_xgboost_threshold_cv.csv",
        index=False,
    )

    metric_columns = [
        "average_precision",
        "roc_auc",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "threshold",
    ]

    print("\nMEAN RESULTS BY METHOD")
    print(
        results.groupby("method")[metric_columns]
        .agg(["mean", "std"])
        .round(3)
        .to_string()
    )

    print(
        "\nTotal runtime:",
        round(time.perf_counter() - all_start, 1),
        "seconds",
    )
    print(
        "Saved:",
        RESULTS_DIR / "gpu_xgboost_threshold_cv.csv",
    )


if __name__ == "__main__":
    main()
