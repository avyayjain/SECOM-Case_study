
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        return self

    def transform(self, X):
        return X.iloc[:, self.keep_mask_]


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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

forest = Pipeline([
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

forest.fit(X_train, y_train)

probabilities = forest.predict_proba(X_test)[:, 1]

print("Average precision:",
      round(average_precision_score(y_test, probabilities), 3))

print("\nProbability distribution:")
print(pd.Series(probabilities).describe().round(4))

print("\nHighest 15 predicted fault probabilities:")
print(np.sort(probabilities)[-15:][::-1].round(4))

print("\nThreshold comparison on the test set (diagnostic only):")

for threshold in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    predictions = (probabilities >= threshold).astype(int)

    print(
        f"threshold={threshold:.2f} | "
        f"predicted_faults={predictions.sum():3d} | "
        f"precision={precision_score(y_test, predictions, zero_division=0):.3f} | "
        f"recall={recall_score(y_test, predictions, zero_division=0):.3f} | "
        f"F1={f1_score(y_test, predictions, zero_division=0):.3f}"
    )
