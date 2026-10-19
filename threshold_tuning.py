
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)


# 1. Remove features that are completely missing in the training fold
class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        return self

    def transform(self, X):
        return X.iloc[:, self.keep_mask_]


# 2. Load the dataset
X = pd.read_csv(
    Path("data") / "secom.data",
    sep=r"\s+",
    header=None,
)

labels = pd.read_csv(
    Path("data") / "secom_labels.data",
    sep=r"\s+",
    header=None,
).iloc[:, 0]

# Convert -1 (pass) to 0 and +1 (fault) to 1
y = labels.map({-1: 0, 1: 1})

if y.isna().any():
    raise ValueError("Unexpected target labels.")


# 3. Keep a final test set separate
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# 4. Build the model pipeline
model = Pipeline([
    ("drop_empty", DropAllMissingFeatures()),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    ),
])


# 5. Generate predictions for training rows using cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
)[:, 1]


# 6. Compare different decision thresholds
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50]

results = []

for threshold in thresholds:
    predictions = (oof_probabilities >= threshold).astype(int)

    results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_train, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_train, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_train, predictions, zero_division=0
        ),
    })

results_df = pd.DataFrame(results)

print("\nThreshold comparison using training cross-validation:")
print(results_df.round(3).to_string(index=False))


# 7. Select the threshold with the highest cross-validated F1
best_row = results_df.loc[results_df["f1"].idxmax()]
best_threshold = float(best_row["threshold"])

print("\nSelected threshold:", best_threshold)
print("Validation precision:", round(best_row["precision"], 3))
print("Validation recall:", round(best_row["recall"], 3))
print("Validation F1:", round(best_row["f1"], 3))


# 8. Fit on all training data and evaluate on the test set
model.fit(X_train, y_train)

test_probabilities = model.predict_proba(X_test)[:, 1]

# Standard threshold
default_predictions = (test_probabilities >= 0.50).astype(int)

# Threshold selected without using test labels
tuned_predictions = (
    test_probabilities >= best_threshold
).astype(int)

for name, predictions in [
    ("Default threshold (0.50)", default_predictions),
    (f"Tuned threshold ({best_threshold:.2f})", tuned_predictions),
]:
    print(f"\n--- {name} ---")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Pass", "Faulty"],
        zero_division=0,
    ))

    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(y_test, predictions), 3),
    )
