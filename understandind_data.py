from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import pandas as pd
import requests

# 1. Create a folder to store the dataset.
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# 2. Download the SECOM dataset from the UCI repository.
DATA_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"

response = requests.get(DATA_URL, timeout=60)
response.raise_for_status()

# 3. Open the downloaded ZIP file in memory.
with ZipFile(BytesIO(response.content)) as archive:
    print("Files inside the ZIP:")
    print(archive.namelist())

    # Find the two files we need.
    data_file = next(
        name for name in archive.namelist()
        if name.endswith("secom.data")
    )
    labels_file = next(
        name for name in archive.namelist()
        if name.endswith("secom_labels.data")
    )

    # Save copies locally so we can inspect them later.
    (DATA_DIR / "secom.data").write_bytes(
        archive.read(data_file)
    )
    (DATA_DIR / "secom_labels.data").write_bytes(
        archive.read(labels_file)
    )

# 4. Read the sensor measurements.
X = pd.read_csv(
    DATA_DIR / "secom.data",
    sep=r"\s+",
    header=None
)

# 5. Read the target labels.
y_table = pd.read_csv(
    DATA_DIR / "secom_labels.data",
    sep=r"\s+",
    header=None
)

y = y_table.iloc[:, 0]

# 6. Display the basic structure of the dataset.
print("\nSensor data shape:", X.shape)
print("Label table shape:", y_table.shape)

print("\nFirst five rows of sensor data:")
print(X.head())

print("\nFirst five original labels:")
print(y.head())

print("\nOriginal label counts:")
print(y.value_counts(dropna=False))

print("\nNumber of missing sensor values:", X.isna().sum().sum())
