
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# False-negative cost / false-positive cost
COST_RATIOS = [5, 10, 20]

# --------------------------------------------------
# 2. Load the SECOM dataset
# --------------------------------------------------

X = pd.read_csv(
    DATA_DIR / "secom.data",
    sep=r"\s+",
    header=None,
    na_values="?"
)

labels = pd.read_csv(
    DATA_DIR / "secom_labels.data",
    sep=r"\s+",
    header=None
)

# SECOM labels: -1 = pass, +1 = fault
y = (labels.iloc[:, 0] == 1).astype(int)

print("Dataset shape:", X.shape)
print("Faulty observations:", int(y.sum()))
print("Passing observations:", int((y == 0).sum()))

# --------------------------------------------------
# 3. Remove columns that are entirely missing
#    This transformation is fitted inside each fold.
# --------------------------------------------------

class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        return self

    def transform(self, X):
        return X.iloc[:, self.keep_mask_]


# --------------------------------------------------
# 4. Define candidate models
# --------------------------------------------------

def make_logistic_all():
    return Pipeline([
        ("drop_empty", DropAllMissingFeatures()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", __import__(
            "sklearn.preprocessing", fromlist=["StandardScaler"]
        ).StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        ))
    ])


def make_logistic_100():
    return Pipeline([
        ("drop_empty", DropAllMissingFeatures()),
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(
            score_func=lambda X, y: mutual_info_classif(
                X, y, random_state=RANDOM_STATE
            ),
            k=100
        )),
        ("scale", __import__(
            "sklearn.preprocessing", fromlist=["StandardScaler"]
        ).StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        ))
    ])


def make_random_forest():
    return Pipeline([
        ("drop_empty", DropAllMissingFeatures()),
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])


models = {
    "Logistic - 100 features": make_logistic_100,
    "Random Forest - all": make_random_forest,
}

# --------------------------------------------------
# 5. Helpers: cost and threshold selection
# --------------------------------------------------

def calculate_cost(y_true, y_pred, fn_cost, fp_cost):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    total_cost = fn * fn_cost + fp * fp_cost

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "total_cost": total_cost,
        "cost_per_observation": total_cost / len(y_true),
    }


def choose_threshold_by_cost(y_true, probabilities, fn_cost, fp_cost):
    # Select using inner training predictions only.
    thresholds = np.arange(0.01, 0.501, 0.01)

    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        cost = calculate_cost(
            y_true, predictions, fn_cost, fp_cost
        )["cost_per_observation"]

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold


# --------------------------------------------------
# 6. Nested cross-validation
# --------------------------------------------------

outer_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE
)

fold_results = []

for model_name, model_factory in models.items():
    print(f"\nEvaluating {model_name}")

    for fold_number, (train_idx, valid_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        inner_cv = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=100 + fold_number
        )

        # OOF predictions are generated only within this outer
        # training partition. Parallelize CPU work here.
        inner_probabilities = cross_val_predict(
            model_factory(),
            X_train,
            y_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]

        fitted_model = model_factory()
        fitted_model.fit(X_train, y_train)

        valid_probabilities = fitted_model.predict_proba(X_valid)[:, 1]

        for ratio in COST_RATIOS:
            threshold = choose_threshold_by_cost(
                y_train,
                inner_probabilities,
                fn_cost=ratio,
                fp_cost=1
            )

            predictions = (
                valid_probabilities >= threshold
            ).astype(int)

            costs = calculate_cost(
                y_valid,
                predictions,
                fn_cost=ratio,
                fp_cost=1
            )

            fold_results.append({
                "model": model_name,
                "fold": fold_number,
                "FN_cost_ratio": ratio,
                "threshold": threshold,
                **costs,
                "average_precision": average_precision_score(
                    y_valid, valid_probabilities
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y_valid, predictions
                ),
                "precision": precision_score(
                    y_valid, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_valid, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_valid, predictions, zero_division=0
                ),
            })

            print(
                f"  Fold {fold_number}, FN cost {ratio}:1"
                f" | threshold={threshold:.2f}"
                f" | FN={costs['FN']}"
                f" | FP={costs['FP']}"
                f" | cost={costs['total_cost']}"
            )

# --------------------------------------------------
# 7. Summarize results across outer folds
# --------------------------------------------------

fold_df = pd.DataFrame(fold_results)

summary = (
    fold_df.groupby(["model", "FN_cost_ratio"])
    .agg(
        mean_cost=("cost_per_observation", "mean"),
        std_cost=("cost_per_observation", "std"),
        mean_FN=("FN", "mean"),
        mean_FP=("FP", "mean"),
        mean_TP=("TP", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_AP=("average_precision", "mean"),
        mean_threshold=("threshold", "mean"),
    )
    .reset_index()
)

fold_path = RESULTS_DIR / "cost_sensitive_fold_results.csv"
summary_path = RESULTS_DIR / "cost_sensitive_summary.csv"

fold_df.to_csv(fold_path, index=False)
summary.to_csv(summary_path, index=False)

print("\nCOST-SENSITIVE SUMMARY")
print(summary.round(3).to_string(index=False))

print("\nSaved results:")
print(fold_path)
print(summary_path)
