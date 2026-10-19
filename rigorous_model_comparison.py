
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. Load sensor measurements and labels
X = pd.read_csv(
    "data/secom.data",
    sep=r"\s+",
    header=None,
    na_values="NaN",
)

labels = pd.read_csv(
    "data/secom_labels.data",
    sep=r"\s+",
    header=None,
)

# Convert original labels: +1 = fault, -1 = pass.
y = (labels.iloc[:, 0] == 1).astype(int)


# 2. Create the same stratified train/test split as before.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

print("Training observations:", len(X_train))
print("Test observations reserved for final evaluation:", len(X_test))
print("Training faults:", int(y_train.sum()))
print("Test faults:", int(y_test.sum()))


# 3. Define a transformer that removes entirely empty columns.
class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        return self

    def transform(self, X):
        return X.iloc[:, self.keep_mask_]


def mutual_information_score(X, y):
    return mutual_info_classif(X, y, random_state=42)


# 4. Build Logistic Regression pipelines.
def build_logistic(k=None):
    steps = [
        ("drop_empty", DropAllMissingFeatures()),
        ("impute", SimpleImputer(strategy="median")),
    ]

    if k is not None:
        steps.append(
            ("select", SelectKBest(
                score_func=mutual_information_score,
                k=k,
            ))
        )

    steps.extend([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        )),
    ])

    return Pipeline(steps)


# 5. Build Random Forest and dummy baseline pipelines.
def build_forest():
    return Pipeline([
        ("drop_empty", DropAllMissingFeatures()),
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )),
    ])


def build_dummy():
    return Pipeline([
        ("drop_empty", DropAllMissingFeatures()),
        ("impute", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])


models = {
    "Dummy baseline": build_dummy(),
    "Logistic Regression - all": build_logistic(),
    "Logistic Regression - 100 features": build_logistic(k=100),
    "Random Forest - all": build_forest(),
}


# 6. Use identical stratified folds for every model.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": "f1",
}

rows = []

for name, model in models.items():
    print(f"\nEvaluating {name}...", flush=True)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    row = {"model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std(ddof=1)

    rows.append(row)


# 7. Print and save mean and standard deviation.
results = pd.DataFrame(rows)

columns = [
    "model",
    "average_precision_mean",
    "average_precision_std",
    "roc_auc_mean",
    "roc_auc_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "recall_mean",
    "recall_std",
    "f1_mean",
    "f1_std",
]

print("\nRIGOROUS CROSS-VALIDATION COMPARISON")
print(results[columns].round(3).to_string(index=False))

Path("results/tables").mkdir(parents=True, exist_ok=True)

results[columns].to_csv(
    "results/tables/rigorous_model_comparison.csv",
    index=False,
)

print("\nSaved to results/tables/rigorous_model_comparison.csv")
print("The held-out test set was not used in cross-validation.")
