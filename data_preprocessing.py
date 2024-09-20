import pandas as pd
import numpy as np
import math

"""
USER-SETTINGS
"""

# Exceptions for Columns that shall not be dropped from the Datasets can be set here:
COLUMNS_DROP_EXCEPTIONS = [
    "Others_surgery_type_text:",
    "Enterectomy"
]

# Values in Cells are mapped according to this dict
CELL_MAPPING = {
    ",": ".",
    "No": "0",
    "Yes": "1",
    "Male": "0",
    "Female": "1",
    "Extensive (Two or more segments)": "2",
    "Segmental (One segment)": "1"
}

# Class Imbalance Ratio
CLASS_IMBALANCE_RATIO = 0.95

# Variance Drop
REL_DROP_THRESHOLD = 0.05

# Count Drop
COUNT_DROP_THRESHOLD = 0.3


data_common = pd.read_csv("data/sub_data/data_common.csv")
data_personal = pd.read_csv("data/sub_data/data_personal.csv")
data_imaging = pd.read_csv("data/sub_data/data_imaging.csv")
data_endo = pd.read_csv("data/sub_data/data_endo.csv")
data_lab_endo = pd.read_csv("data/sub_data/data_lab_endo.csv")
data_surgical = pd.read_csv("data/sub_data/data_surgical.csv")
data_lab_surgical = pd.read_csv("data/sub_data/data_lab_surgical.csv")

"""
Data-Structure for all Sub-Datasets
"""
data = {
    "data_common": data_common,
    "data_personal": data_personal,
    "data_imaging": data_imaging,
    "data_endo": data_endo,
    "data_lab_endo": data_lab_endo,
    "data_surgical": data_surgical,
    "data_lab_surgical": data_lab_surgical
}

for key in data.keys():
    data[key] = data[key].dropna(how="all", axis=1)  # Drop all-NaN-Columns
    data[key] = data[key].drop(["Repeat Instrument",
                                "Repeat Instance",
                                "Unnamed: 0"],
                               axis=1,
                               errors="ignore")      # Drop irrelevant Columns
    data[key] = data[key].set_index("Record ID")     # Set Index to Record ID (Indexing from Dataset)
    data[key] = data[key].dropna(how="all")          # Drop empty rows (There should be none)

# Check for any inf's in Datasets
for key in data.keys():
    if data[key].isin([np.inf, -1 * np.inf]).any().any():
        raise Exception(f"Dataset {key} contains inf's!")

# Delete Columns where Column-Name contains "other"
for key in data.keys():
    current_columns = data[key].columns
    for column in current_columns:
        value_counts = data[key][column].value_counts()         # Calculate Value Counts
        num_categories = len(value_counts)                      # Calculate number of categories in Column
        num_entries = data[key][column].count()                 # Calculate number of non-null entries

        # Don't drop user-set Exceptions
        if column in COLUMNS_DROP_EXCEPTIONS:
            continue

        # Drop other-Columns
        if "other" in column.lower() and num_categories > 2:
            current_columns = current_columns.drop(column)

        # Drop complete-Columns
        if "complete" in column.lower():
            current_columns = current_columns.drop(column)

        # Drop if high Class-Imbalance
        if num_categories == 2 and "No" in value_counts and "Yes" in value_counts:
            # Calculate No-/Yes-Counts
            no_ratio = value_counts["No"] / num_entries
            yes_ratio = value_counts["Yes"] / num_entries

            # Drop if CLASS_IMBALANCE_RATIO is surpassed
            if max(no_ratio, yes_ratio) > CLASS_IMBALANCE_RATIO:
                current_columns = current_columns.drop(column)

    data[key] = data[key][current_columns]

"""
Apply Cell-Mapping
"""

def map_cell(cell_value):
    return_value = str(cell_value)
    for target in CELL_MAPPING.keys():
        replacement = CELL_MAPPING[target]
        return_value = return_value.replace(target, replacement)
    return return_value

for key in data.keys():
    # Apply Cell-Mapping
    data[key] = data[key].apply(lambda column: [map_cell(cell) for cell in column])

    # Cast Values to Float, ignore errors
    data[key] = data[key].apply(pd.DataFrame.astype, dtype=float, errors="ignore")

    # Drop non-numeric Columns
    to_drop = data[key].select_dtypes(exclude=np.number).columns
    data[key] = data[key].drop(to_drop, axis=1)


"""
Drop due to Variance
"""

for key in data.keys():
    variance_data = pd.DataFrame(data={
        "std": data[key].std(),
        "count": data[key].count(),
        "mean": data[key].mean(),
        "std div mean": np.abs(data[key].std() / data[key].mean())
    })

    # Drop Values with low Standard-Deviation / Mean
    column_mask_std = variance_data["std div mean"] > REL_DROP_THRESHOLD
    columns_std = data[key].columns[column_mask_std]

    # Drop Values with low relative non-Null Count
    column_mask_count = variance_data["count"] / len(data[key].index) > COUNT_DROP_THRESHOLD
    columns_count = data[key].columns[column_mask_count]

    data[key] = data[key][columns_count.intersection(columns_std)]

"""
Export Preprocessed Data
"""
for key in data.keys():
    data[key] = data[key].astype(float)
    data[key].to_csv(f"data/prep_data/{key}.csv")