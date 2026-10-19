
from pathlib import Path

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    average_precision_score,
)


class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        return self

    def transform(self, X):
        return X.iloc[:, self.keep_mask_]


# Load data
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


# Use exactly the same split as before
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# Model A: Logistic Regression
logistic_model = Pipeline([
    ("drop_empty", DropAllMissingFeatures()),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )),
])


# Model B: Random Forest
forest_model = Pipeline([
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
    "Logistic Regression": logistic_model,
    "Random Forest": forest_model,
}


# Train and evaluate
results = []

for name, model in models.items():
    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    # Use the same 0.30 threshold for both models
    predictions = (probabilities >= 0.30).astype(int)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Pass", "Faulty"],
        zero_division=0,
    ))

    balanced_acc = balanced_accuracy_score(y_test, predictions)
    avg_precision = average_precision_score(y_test, probabilities)

    results.append({
        "model": name,
        "threshold": 0.30,
        "fault_recall": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )["1"]["recall"],
        "balanced_accuracy": balanced_acc,
        "average_precision": avg_precision,
    })


print("\nSUMMARY")
print(pd.DataFrame(results).round(3).to_string(index=False))
