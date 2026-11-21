
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# SECTION 1: Configuration and directories
# ============================================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ROWS = 1567
EXPECTED_FEATURES = 590

# Exploratory threshold only. No observations are removed here.
THREE_SIGMA = 3

# Correlation threshold used to identify potentially redundant sensors.
CORRELATION_THRESHOLD = 0.80


# ============================================================================
# SECTION 2: Load and validate the dataset
# ============================================================================

print("=" * 65)
print("SECOM DATA EXPLORATION AND DATA QUALITY ANALYSIS")
print("=" * 65)

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

if y_table.shape[1] < 1:
    raise ValueError("The label file does not contain a target column.")

# First column contains the class label.
# The second column, when present, contains the observation timestamp.
y_original = y_table.iloc[:, 0]

if len(X) != len(y_original):
    raise ValueError("Sensor and target row counts do not match.")

if X.shape[0] != EXPECTED_ROWS:
    raise ValueError(f"Expected {EXPECTED_ROWS} observations.")

if X.shape[1] != EXPECTED_FEATURES:
    raise ValueError(f"Expected {EXPECTED_FEATURES} sensor features.")

if y_original.isna().any():
    raise ValueError("Target labels contain missing values.")

if not y_original.isin([-1, 1]).all():
    raise ValueError("Unexpected target labels found.")

# Original convention: -1 = pass, +1 = faulty.
# Analysis convention: 0 = pass, 1 = faulty.
y = y_original.map({-1: 0, 1: 1})

# Preserve the original row order and index.
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print("\nDataset validation: PASSED")
print("Sensor data shape:", X.shape)
print("Target data shape:", y.shape)

print("\nOriginal target label counts:")
print(y_original.value_counts().sort_index())

print("\nConverted target label counts:")
print(y.value_counts().sort_index())


# ============================================================================
# SECTION 3: General dataset description
# ============================================================================

print("\n" + "=" * 65)
print("1. GENERAL DATA DESCRIPTION")
print("=" * 65)

n_rows, n_features = X.shape

total_cells = n_rows * n_features
total_missing = int(X.isna().sum().sum())

missing_percentage = (
    total_missing / total_cells * 100
)

pass_count = int((y == 0).sum())
fault_count = int((y == 1).sum())

fault_percentage = fault_count / len(y) * 100
pass_percentage = pass_count / len(y) * 100

print("Total observations:", n_rows)
print("Total sensor features:", n_features)
print("Total sensor measurements:", total_cells)
print("Total missing values:", total_missing)
print(f"Overall missing percentage: {missing_percentage:.2f}%")

print("\nPass observations:", pass_count)
print("Fault observations:", fault_count)
print(f"Pass percentage: {pass_percentage:.2f}%")
print(f"Fault percentage: {fault_percentage:.2f}%")

print("\nData types:")
print(X.dtypes.value_counts())


# ============================================================================
# SECTION 4: Target distribution and class imbalance
# ============================================================================

print("\n" + "=" * 65)
print("2. TARGET DISTRIBUTION")
print("=" * 65)

label_counts = y.value_counts().sort_index()
label_proportions = y.value_counts(normalize=True).sort_index()

target_report = pd.DataFrame({
    "class": ["Pass", "Faulty"],
    "label": [0, 1],
    "count": [
        label_counts.get(0, 0),
        label_counts.get(1, 0)
    ],
    "percentage": [
        label_proportions.get(0, 0) * 100,
        label_proportions.get(1, 0) * 100
    ]
})

target_report.to_csv(
    TABLES_DIR / "target_distribution.csv",
    index=False
)

print(target_report.to_string(index=False))

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["Pass (0)", "Faulty (1)"],
    [
        label_counts.get(0, 0),
        label_counts.get(1, 0)
    ]
)

plt.title("SECOM: Target Class Distribution")
plt.xlabel("Recorded outcome")
plt.ylabel("Number of observations")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(bar.get_height())}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "target_distribution.png",
    dpi=150
)

# Keep the existing handover filename as well.
plt.savefig(
    FIGURES_DIR / "class_distribution.png",
    dpi=150
)

plt.close()


# ============================================================================
# SECTION 5: Missing-value analysis
# ============================================================================

print("\n" + "=" * 65)
print("3. MISSING-VALUE ANALYSIS")
print("=" * 65)

missing_by_feature = X.isna().sum()
missing_by_row = X.isna().sum(axis=1)

missing_percentage_by_feature = (
    missing_by_feature / n_rows * 100
)

missing_percentage_by_row = (
    missing_by_row / n_features * 100
)

print("\nTotal missing sensor values:", total_missing)
print(f"Overall missing percentage: {missing_percentage:.2f}%")

print("\nTop 15 features with the most missing values:")

print(
    missing_by_feature
    .sort_values(ascending=False)
    .head(15)
)

print("\nTop 10 observations with the most missing values:")

print(
    missing_by_row
    .sort_values(ascending=False)
    .head(10)
)

# Feature-level report.
missing_report = pd.DataFrame({
    "feature": X.columns,
    "missing_count": missing_by_feature.values,
    "missing_percentage": missing_percentage_by_feature.values,
    "non_missing_count": n_rows - missing_by_feature.values
})

missing_report = missing_report.sort_values(
    "missing_percentage",
    ascending=False
)

missing_report.to_csv(
    TABLES_DIR / "missingness_by_feature.csv",
    index=False
)

# Observation-level report.
row_report = pd.DataFrame({
    "row_index": range(n_rows),
    "missing_count": missing_by_row.values,
    "missing_percentage": missing_percentage_by_row.values
})

row_report.to_csv(
    TABLES_DIR / "missingness_by_row.csv",
    index=False
)

# Count completely missing and completely observed features.
all_missing_features = int(
    (missing_by_feature == n_rows).sum()
)

no_missing_features = int(
    (missing_by_feature == 0).sum()
)

all_missing_rows = int(
    (missing_by_row == n_features).sum()
)

no_missing_rows = int(
    (missing_by_row == 0).sum()
)

print("\nFeatures with no missing values:", no_missing_features)
print("Features missing all values:", all_missing_features)
print("Rows with no missing values:", no_missing_rows)
print("Rows missing all sensor values:", all_missing_rows)


# ============================================================================
# SECTION 6: Missingness by target class
# ============================================================================

print("\n" + "=" * 65)
print("4. MISSINGNESS BY TARGET CLASS")
print("=" * 65)

class_missingness = []

for class_label, class_name in [(0, "Pass"), (1, "Faulty")]:

    class_mask = y == class_label
    class_X = X.loc[class_mask]

    class_observations = len(class_X)

    class_missing_cells = int(
        class_X.isna().sum().sum()
    )

    class_total_cells = class_X.shape[0] * class_X.shape[1]

    class_missing_percentage = (
        class_missing_cells / class_total_cells * 100
        if class_total_cells > 0 else 0
    )

    average_missing_per_observation = (
        class_X.isna().sum(axis=1).mean()
    )

    class_missingness.append({
        "class": class_name,
        "label": class_label,
        "observations": class_observations,
        "missing_cells": class_missing_cells,
        "missing_percentage": class_missing_percentage,
        "average_missing_per_observation":
            average_missing_per_observation
    })

class_missingness_report = pd.DataFrame(class_missingness)

print(class_missingness_report.to_string(index=False))

class_missingness_report.to_csv(
    TABLES_DIR / "missingness_by_class.csv",
    index=False
)


# ============================================================================
# SECTION 7: Statistical description of sensor features
# ============================================================================

print("\n" + "=" * 65)
print("5. SENSOR STATISTICAL DESCRIPTION")
print("=" * 65)

# These statistics ignore missing values.
descriptive_statistics = X.describe().T

descriptive_statistics["missing_count"] = missing_by_feature
descriptive_statistics["missing_percentage"] = (
    missing_percentage_by_feature
)

descriptive_statistics["skewness"] = X.skew()

descriptive_statistics["variance"] = X.var()

descriptive_statistics.to_csv(
    TABLES_DIR / "descriptive_statistics.csv",
    index_label="feature"
)

print("\nFirst 15 sensor features:")
print(descriptive_statistics.head(15))

# Identify features with the highest absolute skewness.
skewness_report = (
    descriptive_statistics[["skewness"]]
    .assign(
        absolute_skewness=lambda df: df["skewness"].abs()
    )
    .sort_values(
        "absolute_skewness",
        ascending=False
    )
)

skewness_report.to_csv(
    TABLES_DIR / "feature_skewness.csv",
    index_label="feature"
)

print("\nTop 10 features by absolute skewness:")
print(skewness_report.head(10))


# ============================================================================
# SECTION 8: Exploratory 3-sigma outlier analysis
# ============================================================================

print("\n" + "=" * 65)
print("6. EXPLORATORY OUTLIER ANALYSIS (3-SIGMA)")
print("=" * 65)

# Calculate mean and standard deviation using available observations.
feature_means = X.mean()
feature_stds = X.std(ddof=0)

lower_bounds = feature_means - THREE_SIGMA * feature_stds
upper_bounds = feature_means + THREE_SIGMA * feature_stds

# Identify values outside mean +/- 3 standard deviations.
outlier_mask = (
    X.lt(lower_bounds) | X.gt(upper_bounds)
)

# Missing values are not outliers.
outlier_mask = outlier_mask.fillna(False)

outlier_counts = outlier_mask.sum()

non_missing_counts = X.notna().sum()

outlier_report = pd.DataFrame({
    "feature": X.columns,
    "outlier_count": outlier_counts.values,
    "non_missing_count": non_missing_counts.values,
    "outlier_percentage": (
        outlier_counts.values
        / non_missing_counts.replace(0, np.nan).values
        * 100
    )
})

outlier_report["outlier_percentage"] = (
    outlier_report["outlier_percentage"].fillna(0)
)

outlier_report = outlier_report.sort_values(
    "outlier_count",
    ascending=False
)

outlier_report.to_csv(
    TABLES_DIR / "outlier_3sigma_by_feature.csv",
    index=False
)

total_outliers = int(outlier_counts.sum())

print("Total potential 3-sigma outlier measurements:", total_outliers)

print("\nTop 15 features by outlier count:")
print(outlier_report.head(15).to_string(index=False))

# Important: This is an exploratory count.
# No values are removed or capped during EDA.


# ============================================================================
# SECTION 9: Correlation and redundant feature analysis
# ============================================================================

print("\n" + "=" * 65)
print("7. SENSOR CORRELATION ANALYSIS")
print("=" * 65)

# Pairwise Pearson correlations use available observations.
correlation_matrix = X.corr(method="pearson")

# Extract unique pairs from the upper triangle.
correlation_values = correlation_matrix.to_numpy()

upper_triangle = np.triu(
    np.ones(correlation_values.shape, dtype=bool),
    k=1
)

row_indices, column_indices = np.where(upper_triangle)

correlation_pairs = []

for row_index, column_index in zip(row_indices, column_indices):

    correlation = correlation_values[row_index, column_index]

    if np.isfinite(correlation):
        correlation_pairs.append({
            "feature_1": X.columns[row_index],
            "feature_2": X.columns[column_index],
            "correlation": correlation,
            "absolute_correlation": abs(correlation)
        })

correlation_report = pd.DataFrame(
    correlation_pairs,
    columns=[
        "feature_1",
        "feature_2",
        "correlation",
        "absolute_correlation"
    ]
)

if not correlation_report.empty:

    correlation_report = correlation_report.sort_values(
        "absolute_correlation",
        ascending=False
    )

    strong_correlations = correlation_report[
        correlation_report["absolute_correlation"]
        >= CORRELATION_THRESHOLD
    ]

else:
    strong_correlations = correlation_report.copy()

correlation_report.to_csv(
    TABLES_DIR / "sensor_correlations.csv",
    index=False
)

strong_correlations.to_csv(
    TABLES_DIR / "high_correlations.csv",
    index=False
)

print("Total valid feature pairs:", len(correlation_report))

print(
    f"Pairs with absolute correlation >= {CORRELATION_THRESHOLD}:",
    len(strong_correlations)
)

print("\nTop 15 correlated sensor pairs:")

print(
    correlation_report.head(15).to_string(index=False)
)

# Save a correlation heatmap for the 30 features with the highest variance.
# This is a visual sample, not a feature-selection decision.
top_variance_features = (
    X.var()
    .sort_values(ascending=False)
    .head(30)
    .index
)

if len(top_variance_features) > 1:

    top_correlation = X[top_variance_features].corr()

    plt.figure(figsize=(12, 10))

    plt.imshow(
        top_correlation,
        aspect="auto",
        interpolation="nearest",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(label="Pearson correlation")

    plt.title("Correlation Heatmap: 30 Highest-Variance Sensors")
    plt.xlabel("Sensor feature")
    plt.ylabel("Sensor feature")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "sensor_correlation_heatmap.png",
        dpi=150
    )

    plt.close()


# ============================================================================
# SECTION 10: Visualise missingness
# ============================================================================

print("\n" + "=" * 65)
print("8. MISSINGNESS VISUALISATIONS")
print("=" * 65)

# Feature-level missingness histogram.
plt.figure(figsize=(10, 5))

plt.hist(
    missing_percentage_by_feature,
    bins=30
)

plt.title("Missing Values Across Sensor Features")
plt.xlabel("Percentage of observations missing")
plt.ylabel("Number of features")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "feature_missingness.png",
    dpi=150
)

# Preserve the handover's existing filename.
plt.savefig(
    FIGURES_DIR / "missingness_histogram.png",
    dpi=150
)

plt.close()


# Row-level missingness histogram.
plt.figure(figsize=(10, 5))

plt.hist(
    missing_by_row,
    bins=30
)

plt.title("Missing Values Per Observation")
plt.xlabel("Number of missing sensor measurements")
plt.ylabel("Number of observations")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "row_missingness.png",
    dpi=150
)

plt.close()


# ============================================================================
# SECTION 11: Consolidated data-quality summary
# ============================================================================

print("\n" + "=" * 65)
print("9. DATA QUALITY SUMMARY")
print("=" * 65)

quality_summary = pd.DataFrame([
    {
        "metric": "Total observations",
        "value": n_rows
    },
    {
        "metric": "Total sensor features",
        "value": n_features
    },
    {
        "metric": "Pass observations",
        "value": pass_count
    },
    {
        "metric": "Fault observations",
        "value": fault_count
    },
    {
        "metric": "Fault percentage",
        "value": round(fault_percentage, 4)
    },
    {
        "metric": "Total missing sensor values",
        "value": total_missing
    },
    {
        "metric": "Missing sensor percentage",
        "value": round(missing_percentage, 4)
    },
    {
        "metric": "Features with no missing values",
        "value": no_missing_features
    },
    {
        "metric": "Features missing all values",
        "value": all_missing_features
    },
    {
        "metric": "Rows with no missing values",
        "value": no_missing_rows
    },
    {
        "metric": "Total potential 3-sigma outliers",
        "value": total_outliers
    },
    {
        "metric": "Highly correlated feature pairs",
        "value": len(strong_correlations)
    }
])

quality_summary.to_csv(
    TABLES_DIR / "data_quality_summary.csv",
    index=False
)

print(quality_summary.to_string(index=False))


# ============================================================================
# SECTION 12: Final output
# ============================================================================

print("\n" + "=" * 65)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nSaved reports to:", RESULTS_DIR.resolve())

print("\nGenerated tables:")
for file_path in sorted(TABLES_DIR.glob("*.csv")):
    print(" -", file_path.name)

print("\nGenerated figures:")
for file_path in sorted(FIGURES_DIR.glob("*.png")):
    print(" -", file_path.name)

print("\nNo observations or sensor features were modified.")
print("EDA is exploratory; preprocessing decisions come next.")
