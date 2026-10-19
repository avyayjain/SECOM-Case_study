
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline


# 1. Load the dataset.
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
y = (labels.iloc[:, 0] == 1).astype(int)


# 2. Remove entirely empty features inside each training fold.
class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        return self

    def transform(self, X):
        return X.iloc[:, self.keep_mask_]


def mi_score(X, y):
    return mutual_info_classif(X, y, random_state=42)


# 3. Build a Random Forest pipeline.
def build_model(k=None):
    steps = [
        ("drop_empty", DropAllMissingFeatures()),
        ("impute", SimpleImputer(strategy="median")),
    ]

    if k is not None:
        steps.append(
            ("select", SelectKBest(score_func=mi_score, k=k))
        )

    steps.append(
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ))
    )

    return Pipeline(steps)


# 4. Preserve the same 80/20 stratified split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

all_results = []

# Compare every usable feature against the top 100.
for name, k in [("All features", None), ("Top 100", 100)]:
    print(f"\nTraining configuration: {name}", flush=True)

    model = build_model(k)

    # Each training observation is predicted by a model that
    # did not train on that observation.
    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    # Select threshold by training out-of-fold F1.
    threshold_rows = []

    for threshold in np.arange(0.02, 0.301, 0.01):
        predictions = (oof_probabilities >= threshold).astype(int)

        threshold_rows.append({
            "threshold": threshold,
            "f1": f1_score(
                y_train, predictions, zero_division=0
            ),
        })

    threshold_df = pd.DataFrame(threshold_rows)
    best = threshold_df.sort_values(
        ["f1", "threshold"],
        ascending=[False, False],
    ).iloc[0]

    threshold = float(best["threshold"])

    # Fit on all training data and evaluate on the held-out test set.
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result = {
        "model": name,
        "threshold": threshold,
        "average_precision": average_precision_score(
            y_test, probabilities
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
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
    }
    all_results.append(result)

    print("Selected threshold:", round(threshold, 2))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(pd.Series(result).to_string())

# 5. Save the comparison.
results = pd.DataFrame(all_results)
Path("results/tables").mkdir(parents=True, exist_ok=True)
results.to_csv(
    "results/tables/forest_feature_selection.csv",
    index=False,
)

print("\nFINAL COMPARISON")
print(results.round(3).to_string(index=False))
print("\nSaved results/tables/forest_feature_selection.csv")
