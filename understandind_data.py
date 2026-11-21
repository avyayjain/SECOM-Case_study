
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")

DATA_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"

SENSOR_FILE = DATA_DIR / "secom.data"
LABEL_FILE = DATA_DIR / "secom_labels.data"

EXPECTED_ROWS = 1567
EXPECTED_FEATURES = 590

VALID_LABELS = {-1, 1}


# ---------------------------------------------------------------------------
# Step 1: Download and extract the SECOM dataset
# ---------------------------------------------------------------------------

def download_and_extract_data():
    """
    Download the SECOM dataset from the UCI repository and extract
    the sensor measurements and corresponding labels.

    Inputs:
        None.

    Outputs:
        Saves secom.data and secom_labels.data inside the data directory.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading SECOM dataset from UCI...")

    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    print("Download completed successfully.")

    with ZipFile(BytesIO(response.content)) as archive:

        file_names = archive.namelist()

        print("\nFiles available in ZIP:")
        for file_name in file_names:
            print(file_name)

        data_file = next(
            (
                name for name in file_names
                if name.endswith("secom.data")
            ),
            None
        )

        labels_file = next(
            (
                name for name in file_names
                if name.endswith("secom_labels.data")
            ),
            None
        )

        if data_file is None or labels_file is None:
            raise FileNotFoundError(
                "Could not find both SECOM sensor and label files in ZIP."
            )

        SENSOR_FILE.write_bytes(archive.read(data_file))
        LABEL_FILE.write_bytes(archive.read(labels_file))

    print("\nDataset files extracted successfully.")
    print(f"Sensor file: {SENSOR_FILE}")
    print(f"Label file: {LABEL_FILE}")


# ---------------------------------------------------------------------------
# Step 2: Load sensor measurements and labels
# ---------------------------------------------------------------------------


def load_data():
    """
    Load sensor measurements and corresponding target labels.

    The SECOM label file contains two columns:
    - Column 0: Class label (-1 = pass, +1 = fault)
    - Column 1: Observation timestamp

    Only the first column is used as the target.
    """

    if not SENSOR_FILE.exists():
        raise FileNotFoundError(f"Missing sensor file: {SENSOR_FILE}")

    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"Missing label file: {LABEL_FILE}")

    # Load sensor measurements.
    X = pd.read_csv(
        SENSOR_FILE,
        sep=r"\s+",
        header=None
    )

    # Load the complete label file.
    y_table = pd.read_csv(
        LABEL_FILE,
        sep=r"\s+",
        header=None
    )

    if y_table.shape[1] < 1:
        raise ValueError("The label file does not contain a target column.")

    # Extract only the actual class label.
    y = y_table.iloc[:, 0].astype(int)

    # Preserve the original row order.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y



# ---------------------------------------------------------------------------
# Step 3: Validate dataset structure and row alignment
# ---------------------------------------------------------------------------

def validate_data(X, y):
    """
    Validate the dimensions, labels, and row alignment of the dataset.

    Inputs:
        X: Sensor measurement DataFrame.
        y: Original target label Series.

    Outputs:
        Raises an error if the dataset does not meet expected conditions.
        Otherwise, prints the validation results.
    """

    print("\n========== DATA VALIDATION ==========")

    print("Sensor data shape:", X.shape)
    print("Label data shape:", y.shape)

    # Check expected dimensions.
    if X.shape[0] != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} sensor rows, found {X.shape[0]}."
        )

    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} sensor features, found {X.shape[1]}."
        )

    if len(y) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} labels, found {len(y)}."
        )

    # Check that every sensor observation has a corresponding label.
    if len(X) != len(y):
        raise ValueError(
            "Sensor and label row counts do not match. "
            "Row alignment cannot be guaranteed."
        )

    print("\nDimension validation: PASSED")
    print("Sensor-label row count validation: PASSED")

    # Check target values.
    observed_labels = set(y.dropna().unique())

    if y.isna().any():
        raise ValueError("The target label column contains missing values.")

    if not observed_labels.issubset(VALID_LABELS):
        raise ValueError(
            f"Unexpected target labels found: {observed_labels}"
        )

    if observed_labels != VALID_LABELS:
        raise ValueError(
            f"Expected both target classes {VALID_LABELS}, "
            f"but found {observed_labels}."
        )

    print("Target label validation: PASSED")

    print("\nOriginal target label distribution:")
    print(y.value_counts().sort_index())

    # Missing-value analysis.
    missing_values = int(X.isna().sum().sum())

    print("\nMissing sensor values:", missing_values)

    missing_percentage = (
        missing_values / X.size
    ) * 100

    print(f"Missing sensor percentage: {missing_percentage:.2f}%")

    print("\nDataset validation completed successfully.")


# ---------------------------------------------------------------------------
# Step 4: Display initial dataset information
# ---------------------------------------------------------------------------

def describe_data(X, y):
    """
    Display the initial structure and sample records of the dataset.

    Inputs:
        X: Sensor measurement DataFrame.
        y: Original target label Series.

    Outputs:
        Prints dataset dimensions, sample records, data types,
        target distribution, and missing-value information.
    """

    print("\n========== INITIAL DATA DESCRIPTION ==========")

    print("\nSensor data shape:", X.shape)
    print("Label data shape:", y.shape)

    print("\nFirst five sensor observations:")
    print(X.head())

    print("\nFirst five original labels:")
    print(y.head())

    print("\nSensor data types:")
    print(X.dtypes.value_counts())

    print("\nOriginal target label counts:")
    print(y.value_counts(dropna=False))

    print("\nMissing values per sensor (first 20):")
    print(X.isna().sum().head(20))


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    """
    Execute the initial SECOM data collection and validation workflow.
    """

    download_and_extract_data()

    X, y = load_data()

    validate_data(X, y)

    describe_data(X, y)

    print("\nSECOM data collection stage completed.")


if __name__ == "__main__":
    main()
