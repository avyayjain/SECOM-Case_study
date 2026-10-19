
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    """Remove columns that are entirely missing in the training fold."""

    def fit(self, X, y=None):
        # True for each column with at least one observed value.
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        return self

    def transform(self, X):
        # Keep the exact columns selected during fit().
        return X.iloc[:, self.keep_mask_]


DATA_DIR = Path("data")

X = pd.read_csv(
    DATA_DIR / "secom.data",
    sep=r"\s+",
    header=None
)

y_original = pd.read_csv(
    DATA_DIR / "secom_labels.data",
    sep=r"\s+",
    header=None
).iloc[:, 0]

y = y_original.map({-1: 0, 1: 1})

if y.isna().any():
    raise ValueError("Unexpected or missing target labels.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Training faulty cases:", int(y_train.sum()))
print("Test faulty cases:", int(y_test.sum()))


model = Pipeline([
    ("drop_empty", DropAllMissingFeatures()),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
    "average_precision": "average_precision",
}

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    error_score="raise"
)

print("\nFive-fold cross-validation results:")

for metric in scoring:
    scores = cv_results[f"test_{metric}"]

    print(
        f"{metric}: "
        f"mean={scores.mean():.3f}, "
        f"std={scores.std():.3f}, "
        f"fold_scores={np.round(scores, 3)}"
    )


model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nFinal baseline test results:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Pass (0)", "Faulty (1)"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4)
)
