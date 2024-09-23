import pandas as pd
import numpy as np

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

"""
Load Data
"""
data_common = pd.read_csv("data/sub_data/data_common.csv")
data_personal = pd.read_csv("data/sub_data/data_personal.csv")
data_imaging = pd.read_csv("data/sub_data/data_imaging.csv")
data_endo = pd.read_csv("data/sub_data/data_endo.csv")
data_lab_endo = pd.read_csv("data/sub_data/data_lab_endo.csv")
data_surgical = pd.read_csv("data/sub_data/data_surgical.csv")
data_lab_surgical = pd.read_csv("data/sub_data/data_lab_surgical.csv")

data = {
    "data_common": data_common,
    "data_personal": data_personal,
    "data_imaging": data_imaging,
    "data_endo": data_endo,
    "data_lab_endo": data_lab_endo,
    "data_surgical": data_surgical,
    "data_lab_surgical": data_lab_surgical
}

"""
Preprocessing
"""
for key in data.keys():
    data[key] = data[key].dropna(how="all", axis=1)  # Drop all-NaN-Columns
    data[key] = data[key].drop(["Repeat Instrument",
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
    """
    Applies all Replacement-Rules specified in CELL_MAPPING to cell_value.
    :param cell_value: Value to apply the Replacement-Rules to
    :return: cell_value with applied Rules
    """
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
    to_drop = data[key].select_dtypes(exclude=np.number).columns    # Get non-numeric Columns
    to_drop = to_drop.difference(COLUMNS_DROP_EXCEPTIONS)           # Apply Exceptions
    data[key] = data[key].drop(to_drop, axis=1)                     # Drop Columns


"""
Export Preprocessed Data
"""
for key in data.keys():
    data[key].to_csv(f"data/prep_data/{key}.csv")