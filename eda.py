
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# SECTION 1: Load the files downloaded in Lesson 1

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

X = pd.read_csv(
    DATA_DIR / "secom.data",
    sep=r"\s+",
    header=None
)

y_table = pd.read_csv(
    DATA_DIR / "secom_labels.data",
    sep=r"\s+",
    header=None
)

y_original = y_table.iloc[:, 0]


# SECTION 2: Validate and convert the target labels

print("Original label counts:")
print(y_original.value_counts(dropna=False))

if y_original.isna().any():
    raise ValueError("Some target labels are missing.")

if not y_original.isin([-1, 1]).all():
    raise ValueError("Unexpected target labels found.")

# Original convention: -1 = pass, +1 = faulty
# Our convention: 0 = pass, 1 = faulty
y = y_original.map({-1: 0, 1: 1})

if len(X) != len(y):
    raise ValueError("Feature and target row counts do not match.")

print("\nConverted label counts:")
print(y.value_counts().sort_index())

print("\nTarget proportions:")
print(y.value_counts(normalize=True).sort_index())


# SECTION 3: Understand missing values

missing_by_feature = X.isna().sum()
missing_by_row = X.isna().sum(axis=1)

total_cells = X.shape[0] * X.shape[1]
total_missing = int(missing_by_feature.sum())
missing_percentage = 100 * total_missing / total_cells

print("\nTotal observations:", X.shape[0])
print("Total features:", X.shape[1])
print("Total missing values:", total_missing)
print(f"Percentage missing: {missing_percentage:.2f}%")

print("\nRows with the most missing values:")
print(missing_by_row.sort_values(ascending=False).head(10))

print("\nFeatures with the most missing values:")
print(missing_by_feature.sort_values(ascending=False).head(15))


# SECTION 4: Create a feature-level missingness report

missing_report = pd.DataFrame({
    "feature": X.columns,
    "missing_count": missing_by_feature.values,
    "missing_percentage": (
        missing_by_feature.values / len(X) * 100
    )
})

missing_report = missing_report.sort_values(
    "missing_percentage",
    ascending=False
)

missing_report.to_csv(
    TABLES_DIR / "missingness_by_feature.csv",
    index=False
)

row_report = pd.DataFrame({
    "row_index": range(len(X)),
    "missing_count": missing_by_row.values,
    "missing_percentage": missing_by_row.values
    / X.shape[1] * 100
})

row_report.to_csv(
    TABLES_DIR / "missingness_by_row.csv",
    index=False
)


# SECTION 5: Plot the target distribution

label_counts = y.value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(
    ["Pass (0)", "Faulty (1)"],
    [label_counts.get(0, 0), label_counts.get(1, 0)]
)
plt.title("SECOM: Target Class Distribution")
plt.xlabel("Recorded outcome")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
plt.close()


# SECTION 6: Plot missingness across features

plt.figure(figsize=(10, 5))
plt.hist(missing_by_feature / len(X) * 100, bins=30)
plt.title("Missing Values Across Sensor Features")
plt.xlabel("Percentage of observations missing")
plt.ylabel("Number of features")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_missingness.png", dpi=150)
plt.close()


# SECTION 7: Plot missingness across rows

plt.figure(figsize=(10, 5))
plt.hist(missing_by_row, bins=30)
plt.title("Missing Values Per Observation")
plt.xlabel("Number of missing sensor measurements")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "row_missingness.png", dpi=150)
plt.close()


# SECTION 8: Summarise the findings

print("\nFeatures with no missing values:",
      int((missing_by_feature == 0).sum()))

print("Features missing all values:",
      int((missing_by_feature == len(X)).sum()))

print("Rows with no missing values:",
      int((missing_by_row == 0).sum()))

print("\nSaved reports to:", RESULTS_DIR.resolve())
print("EDA completed.")
