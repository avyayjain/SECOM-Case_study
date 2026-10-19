
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)

# Load the data saved in Lesson 1.
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

# Convert original labels: -1 = pass, +1 = faulty.
y = y_original.map({-1: 0, 1: 1})

# Check that the input and target are consistent.
assert len(X) == len(y), "X and y have different row counts."
assert y.notna().all(), "Some target labels are invalid."
assert X.shape[0] == 1567, "Unexpected number of observations."

print("Dataset shape:", X.shape)
print("Target counts:")
print(y.value_counts().sort_index())


# Reserve 20% of the observations for final testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining observations:", len(X_train))
print("Test observations:", len(X_test))

print("\nTraining target proportions:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nTest target proportions:")
print(y_test.value_counts(normalize=True).sort_index())


# Identify columns that contain no observed training values.
all_missing_in_train = X_train.isna().all(axis=0)

columns_to_drop = X_train.columns[all_missing_in_train]

print("\nFeatures entirely missing in training:",
      len(columns_to_drop))

# Remove those same columns from both sets.
X_train_clean = X_train.drop(columns=columns_to_drop)
X_test_clean = X_test.drop(columns=columns_to_drop)

print("Training shape after dropping empty features:",
      X_train_clean.shape)

print("Test shape after dropping empty features:",
      X_test_clean.shape)


# Baseline 1: always predict the most common training class.
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train_clean, y_train)

# Baseline 2: impute, scale, and fit logistic regression.
logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])

logistic_model.fit(X_train_clean, y_train)

print("\nBoth baseline models have been trained.")


def evaluate_model(name, model, X_eval, y_eval):
    predictions = model.predict(X_eval)

    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)

    print("Accuracy:",
          round(accuracy_score(y_eval, predictions), 4))

    print("Balanced accuracy:",
          round(balanced_accuracy_score(y_eval, predictions), 4))

    print("Precision:",
          round(precision_score(
              y_eval, predictions, zero_division=0
          ), 4))

    print("Recall:",
          round(recall_score(
              y_eval, predictions, zero_division=0
          ), 4))

    print("F1 score:",
          round(f1_score(
              y_eval, predictions, zero_division=0
          ), 4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_eval, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_eval,
        predictions,
        labels=[0, 1],
        target_names=["Pass (0)", "Faulty (1)"],
        zero_division=0
    ))


# Use the same untouched test set for both comparisons.
evaluate_model(
    "Majority-class dummy baseline",
    dummy_model,
    X_test_clean,
    y_test
)

evaluate_model(
    "Balanced logistic regression",
    logistic_model,
    X_test_clean,
    y_test
)
