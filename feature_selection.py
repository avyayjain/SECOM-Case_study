
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. Load the dataset prepared in Lesson 1.
DATA_DIR = Path("data")

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

# UCI labels: -1 = pass, +1 = fault.
y = (labels.iloc[:, 0] == 1).astype(int)


# 2. Remove features that are entirely missing.
# This transformer learns the columns to keep during fit().
class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        return self

    def transform(self, X):
        return X.iloc[:, self.keep_mask_]


# 3. Keep mutual-information scoring reproducible.
def mutual_information_score(X, y):
    return mutual_info_classif(X, y, random_state=42)


# 4. Build a pipeline.
# k=None means retain every remaining feature.
def build_model(k=None):
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


# 5. Compare models using the same cross-validation folds.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "average_precision": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": "f1",
}

feature_counts = [None, 25, 50, 100, 200]
results = []

for k in feature_counts:
    model = build_model(k)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    results.append({
        "features": "All" if k is None else k,
        "average_precision_mean":
            scores["test_average_precision"].mean(),
        "average_precision_std":
            scores["test_average_precision"].std(),
        "balanced_accuracy_mean":
            scores["test_balanced_accuracy"].mean(),
        "recall_mean": scores["test_recall"].mean(),
        "f1_mean": scores["test_f1"].mean(),
    })

results_df = pd.DataFrame(results)

print("\nFEATURE SELECTION: CROSS-VALIDATION RESULTS")
print(results_df.round(3).to_string(index=False))

Path("results/tables").mkdir(parents=True, exist_ok=True)
results_df.to_csv(
    "results/tables/feature_selection_results.csv",
    index=False,
)

print("\nSaved results to results/tables/feature_selection_results.csv")
