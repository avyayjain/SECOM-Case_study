
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler


# 1. Load data
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


# 2. Remove columns that are completely missing.
class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        return self

    def transform(self, X):
        return X.iloc[:, self.keep_mask_]


def mutual_information_score(X, y):
    return mutual_info_classif(X, y, random_state=42)


# 3. Build the model. Feature selection is learned inside each fold.
def build_model(k):
    return Pipeline([
        ("drop_empty", DropAllMissingFeatures()),
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(
            score_func=mutual_information_score,
            k=k,
        )),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        )),
    ])


# 4. Create one untouched test set.
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

results = []

# 5. Select a threshold using out-of-fold training predictions.
for k in [25, 100, 200]:
    model = build_model(k)

    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold_results = []

    for threshold in np.arange(0.05, 0.51, 0.01):
        predictions = (
            oof_probabilities >= threshold
        ).astype(int)

        threshold_results.append({
            "threshold": threshold,
            "f1": f1_score(
                y_train, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_train, predictions, zero_division=0
            ),
            "precision": precision_score(
                y_train, predictions, zero_division=0
            ),
        })

    threshold_df = pd.DataFrame(threshold_results)

    # Select the threshold with the highest training OOF F1.
    best_row = threshold_df.sort_values(
        ["f1", "threshold"],
        ascending=[False, False],
    ).iloc[0]

    best_threshold = float(best_row["threshold"])

    # 6. Fit on all training data; evaluate only once on test data.
    model.fit(X_train, y_train)
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    results.append({
        "features": k,
        "threshold": best_threshold,
        "test_average_precision": average_precision_score(
            y_test, test_probabilities
        ),
        "test_precision": precision_score(
            y_test, test_predictions, zero_division=0
        ),
        "test_recall": recall_score(
            y_test, test_predictions, zero_division=0
        ),
        "test_f1": f1_score(
            y_test, test_predictions, zero_division=0
        ),
        "test_balanced_accuracy": balanced_accuracy_score(
            y_test, test_predictions
        ),
        "test_confusion_matrix": confusion_matrix(
            y_test, test_predictions
        ).tolist(),
    })

    print(f"\nModel using {k} features")
    print(f"Training-selected threshold: {best_threshold:.2f}")
    print("Test confusion matrix:")
    print(confusion_matrix(y_test, test_predictions))
    print(f"Test precision: {precision_score(y_test, test_predictions, zero_division=0):.3f}")
    print(f"Test recall: {recall_score(y_test, test_predictions, zero_division=0):.3f}")
    print(f"Test F1: {f1_score(y_test, test_predictions, zero_division=0):.3f}")
    print(f"Test balanced accuracy: {balanced_accuracy_score(y_test, test_predictions):.3f}")
    print(f"Test average precision: {average_precision_score(y_test, test_probabilities):.3f}")


# 7. Save results for our final comparison.
results_df = pd.DataFrame(results)
Path("results/tables").mkdir(parents=True, exist_ok=True)
results_df.to_csv(
    "results/tables/selected_model_test_results.csv",
    index=False,
)

print("\nFINAL COMPARISON")
print(results_df.round(3).to_string(index=False))
print("\nSaved to results/tables/selected_model_test_results.csv")
