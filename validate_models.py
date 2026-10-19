
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        return self

    def transform(self, X):
        return X.iloc[:, self.keep_mask_]


# Load dataset
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

y = labels.map({-1: 0, 1: 1})

if y.isna().any():
    raise ValueError("Unexpected target labels.")


# Preserve the same untouched test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


def make_logistic():
    return Pipeline([
        ("drop_empty", DropAllMissingFeatures()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ])


def make_forest():
    return Pipeline([
        ("drop_empty", DropAllMissingFeatures()),
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])


models = {
    "Logistic Regression": make_logistic(),
    "Random Forest": make_forest(),
}

# First: compare model ranking performance using training folds
print("CROSS-VALIDATION ON TRAINING DATA")

for name, model in models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "average_precision": "average_precision",
            "recall": "recall",
            "f1": "f1",
            "balanced_accuracy": "balanced_accuracy",
        },
        error_score="raise",
    )

    print(f"\n{name}")
    for metric in [
        "average_precision",
        "recall",
        "f1",
        "balanced_accuracy",
    ]:
        values = scores[f"test_{metric}"]
        print(
            f"{metric}: mean={values.mean():.3f}, "
            f"std={values.std():.3f}"
        )


# Second: use out-of-fold probabilities to select a threshold
# This step uses training labels only.
print("\nTHRESHOLD SELECTION FROM TRAINING FOLDS")

selected_thresholds = {}

for name, model in models.items():
    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    rows = []

    for threshold in np.arange(0.02, 0.301, 0.01):
        predictions = (oof_probabilities >= threshold).astype(int)

        rows.append({
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

    table = pd.DataFrame(rows)
    best = table.loc[table["f1"].idxmax()]
    selected_thresholds[name] = float(best["threshold"])

    print(f"\n{name}")
    print(table.sort_values("f1", ascending=False).head(5).round(3))
    print("Selected threshold:", round(best["threshold"], 2))


# Third: fit on training data, evaluate the test set once per model
print("\nFINAL TEST EVALUATION")

for name, factory in [
    ("Logistic Regression", make_logistic),
    ("Random Forest", make_forest),
]:
    model = factory()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    threshold = selected_thresholds[name]
    predictions = (probabilities >= threshold).astype(int)

    print(f"\n{name}")
    print("Training-selected threshold:", round(threshold, 2))
    print("Test average precision:",
          round(average_precision_score(y_test, probabilities), 3))
    print("Test precision:",
          round(precision_score(y_test, predictions, zero_division=0), 3))
    print("Test recall:",
          round(recall_score(y_test, predictions, zero_division=0), 3))
    print("Test F1:",
          round(f1_score(y_test, predictions, zero_division=0), 3))
    print("Test balanced accuracy:",
          round(balanced_accuracy_score(y_test, predictions), 3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
