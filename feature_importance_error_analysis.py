
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
THRESHOLD = 0.12
N_REPEATS = 5

# --------------------------------------------------
# 2. Load SECOM
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

# 0 = passing, 1 = faulty
y = (labels.iloc[:, 0] == 1).astype(int)

# Keep original row and feature indices for interpretation.
X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

print("Dataset:", X.shape)
print("Faulty:", int(y.sum()))
print("Passing:", int((y == 0).sum()))

# --------------------------------------------------
# 3. Leakage-safe preprocessing
# --------------------------------------------------

class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        self.kept_columns_ = X.columns[self.keep_mask_].tolist()
        return self

    def transform(self, X):
        return X.loc[:, self.kept_columns_]


pipeline = Pipeline([
    ("drop_empty", DropAllMissingFeatures()),
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )),
])

# --------------------------------------------------
# 4. Split data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

pipeline.fit(X_train, y_train)

probabilities = pipeline.predict_proba(X_test)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

print("\nTEST SET PERFORMANCE")
print("Average precision:", round(
    average_precision_score(y_test, probabilities), 3
))
print("Threshold:", THRESHOLD)
print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Passing", "Faulty"],
    zero_division=0,
))

# --------------------------------------------------
# 5. Built-in Random Forest importance
# --------------------------------------------------

rf = pipeline.named_steps["model"]
kept_features = pipeline.named_steps["drop_empty"].kept_columns_

impurity_importance = pd.DataFrame({
    "feature": kept_features,
    "impurity_importance": rf.feature_importances_,
}).sort_values("impurity_importance", ascending=False)

impurity_importance.to_csv(
    OUTPUT_DIR / "rf_impurity_importance.csv", index=False
)

print("\nTOP 20 FEATURES — IMPURITY IMPORTANCE")
print(impurity_importance.head(20).to_string(index=False))

# --------------------------------------------------
# 6. Permutation importance on held-out data
# --------------------------------------------------
# This asks: how much does AP deteriorate when one feature
# is randomly shuffled? It measures predictive usefulness,
# not causality.

print("\nCalculating permutation importance...")

permutation = permutation_importance(
    pipeline,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

permutation_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": permutation.importances_mean,
    "importance_std": permutation.importances_std,
}).sort_values("importance_mean", ascending=False)

permutation_df.to_csv(
    OUTPUT_DIR / "rf_permutation_importance.csv", index=False
)

print("\nTOP 20 FEATURES — PERMUTATION IMPORTANCE")
print(permutation_df.head(20).round(5).to_string(index=False))

# --------------------------------------------------
# 7. Analyze false positives and false negatives
# --------------------------------------------------

test_cases = pd.DataFrame({
    "original_row_index": X_test.index,
    "actual_label": y_test.to_numpy(),
    "fault_probability": probabilities,
    "predicted_label": predictions,
})

test_cases["error_type"] = np.select(
    [
        (test_cases["actual_label"] == 1)
        & (test_cases["predicted_label"] == 0),

        (test_cases["actual_label"] == 0)
        & (test_cases["predicted_label"] == 1),
    ],
    [
        "False negative",
        "False positive",
    ],
    default="Correct",
)

test_cases = test_cases.sort_values(
    "fault_probability", ascending=False
)

test_cases.to_csv(
    OUTPUT_DIR / "rf_test_predictions_and_errors.csv",
    index=False,
)

false_negatives = test_cases[
    test_cases["error_type"] == "False negative"
]
false_positives = test_cases[
    test_cases["error_type"] == "False positive"
]

print("\nERROR COUNTS")
print(test_cases["error_type"].value_counts().to_string())

print("\nFALSE NEGATIVES — MISSED FAULTS")
print(false_negatives.to_string(index=False))

print("\nFALSE POSITIVES — FALSE ALARMS")
print(false_positives.head(20).to_string(index=False))

print("\nSaved files to:", OUTPUT_DIR)
