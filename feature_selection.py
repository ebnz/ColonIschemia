import pandas as pd
import numpy as np
from sklearn import cluster, metrics, feature_selection
from sklearn.utils import shuffle
from scipy.stats import chi2_contingency as c2
from scipy.stats import pearsonr as pr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


"""
USER-SETTINGS
"""

# All specified Columns in this list won't be manipulated (e.g. Imputing) or dropped
# Adding them will make the Feature-Selection ignore them
# Structure: [(DATASET_NAME, COLUMN_NAME), ...]
EXCEPTIONS = [
    ("data_surgical", "Ischämie?"),
    ("data_endo", "Findings compatible with ischemia"),
    ("data_surgical", "Enterectomy"),
    ("data_surgical", "Colectomy"),
    ("data_surgical", "Exploration"),
    ("data_surgical", "Others_surgery_type_text:")
]

# Variance Drop
REL_DROP_THRESHOLD = 0.05

# Count Drop
COUNT_DROP_THRESHOLD = 0.3

# Min. Correlation-Coefficient at which two Columns get merged
MIN_MERGE_CORRELATION = 0.7

# p-Value Threshold
P_VALUE_THRESHOLD = 0.5

"""
Load Data
"""

data_common = pd.read_csv("data/prep_data/data_common.csv")
data_personal = pd.read_csv("data/prep_data/data_personal.csv")
data_imaging = pd.read_csv("data/prep_data/data_imaging.csv")
data_endo = pd.read_csv("data/prep_data/data_endo.csv")
data_lab_endo = pd.read_csv("data/prep_data/data_lab_endo.csv")
data_surgical = pd.read_csv("data/prep_data/data_surgical.csv")
data_lab_surgical = pd.read_csv("data/prep_data/data_lab_surgical.csv")

data_common["Repeat Instance"] = 1
data_personal["Repeat Instance"] = 1

data_common = data_common.set_index(["Record ID", "Repeat Instance"])
data_personal = data_personal.set_index(["Record ID", "Repeat Instance"])
data_imaging = data_imaging.set_index(["Record ID", "Repeat Instance"])
data_endo = data_endo.set_index(["Record ID", "Repeat Instance"])
data_lab_endo = data_lab_endo.set_index(["Record ID", "Repeat Instance"])
data_surgical = data_surgical.set_index(["Record ID", "Repeat Instance"])
data_lab_surgical = data_lab_surgical.set_index(["Record ID", "Repeat Instance"])

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
Combine Repeat Instances
"""
for dataset_name in data.keys():
    data[dataset_name] = data[dataset_name].groupby("Record ID").aggregate("first")

"""
Copy Columns from EXCEPTIONS
"""

EXCEPTIONS_DICT = {}

for dataset_name, column in EXCEPTIONS:
    EXCEPTIONS_DICT[(dataset_name, column)] = data[dataset_name][column]
    data[dataset_name] = data[dataset_name].drop(column, axis=1)

"""
Feature-Selection
"""

"""
Drop due to Variance
"""

# variance_data holds data on UNIMPUTED sub-data
variance_data = {}

for key in data.keys():
    variance_data[key] = pd.DataFrame(data={
        "std": data[key].std(),
        "count": data[key].count(),
        "mean": data[key].mean(),
        "std div mean": np.abs(data[key].std() / data[key].mean())
    })

    # Drop Values with low Standard-Deviation / Mean
    column_mask_std = variance_data[key]["std div mean"] > REL_DROP_THRESHOLD
    columns_std = data[key].columns[column_mask_std]

    # Drop Values with low relative non-Null Count
    column_mask_count = variance_data[key]["count"] / len(data[key].index) > COUNT_DROP_THRESHOLD
    columns_count = data[key].columns[column_mask_count]

    data[key] = data[key][columns_count.intersection(columns_std)]

"""
Imputation
"""

def mice(data, m) -> pd.DataFrame:
    """
    Multiple Imputation by Chained Equations (MICE) algorithm for imputing missing values in the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame containing missing values.
        m (int): Number of imputations to perform.

    Returns:
        list: List of DataFrames containing imputed values for each iteration of the MICE algorithm.
    """
    imp_dfs = []
    for i in range(m):
        imp = IterativeImputer(
            missing_values=np.nan,
            random_state=i,
            min_value=0,
            sample_posterior=True)
        imp_dfs.append(
            pd.DataFrame(
                imp.fit_transform(data),
                columns=data.columns,
                index=data.index))  # Set the index properly

    return imp_dfs

for dataset_name in data:
    data[dataset_name] = mice(data[dataset_name], 1)[0]


"""
Clustering Correlation Drop
"""

# ToDo: Correlation-Matrices save
clusters = {}

for key in data.keys():
    # Calculate Correlation-Matrix
    correlation_matrix = data[key].corr()

    # Convert Correlation-Matrix to Distance-Matrix
    # Correlation-Matrix 0 -> Low Similarity, 1 -> High Similarity
    # Distance-Matrix 0 -> High Similarity, 1 -> Low Similarity
    distance_matrix = 1 - np.abs(correlation_matrix.to_numpy())

    # Replace NaN-Correlation with Distance 1
    distance_matrix = np.nan_to_num(distance_matrix, nan=1)
    # Replace Diagonal with Distance 0
    np.fill_diagonal(distance_matrix, 0)

    # Generate Clustering with AgglomerativeClustering
    clustering = cluster.AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - MIN_MERGE_CORRELATION,
        metric="precomputed",
        linkage="complete"
    )
    cluster_ids = clustering.fit_predict(distance_matrix)
    index = correlation_matrix.index

    # Extract Clusters
    clusters[key] = []
    num_clusters = clustering.n_clusters_       # Number of found Clusters
    for i in range(num_clusters):
        column_indices = np.where(cluster_ids == i)
        column_names = index[column_indices]
        clusters[key].append(column_names)

# For each cluster, use the column with the most non-NaN values
for key in clusters.keys():
    current_dataset = data[key]

    for single_cluster in clusters[key]:
        sorted_non_nan_counts = variance_data[key]["count"][single_cluster].sort_values(ascending=False)
        highest_non_nan_count_column_name = sorted_non_nan_counts.index[0]
        columns_to_drop = sorted_non_nan_counts.drop(highest_non_nan_count_column_name).index
        data[key] = data[key].drop(columns_to_drop, axis=1)

"""
p-Values
"""

for dataset_name in data.keys():
    non_null_indices = EXCEPTIONS_DICT[("data_surgical", "Ischämie?")].dropna(how="all").index

    X = data[dataset_name].loc[non_null_indices]
    y = EXCEPTIONS_DICT[("data_surgical", "Ischämie?")].loc[non_null_indices]

    for column in data[dataset_name].columns:
        table = pd.crosstab(y, X[column])
        chi2_stats = c2(table)
        if chi2_stats.pvalue > P_VALUE_THRESHOLD:
            data[dataset_name] = data[dataset_name].drop(column, axis=1)

"""
Insert Columns from EXCEPTIONS
"""
for dataset_name, column in EXCEPTIONS_DICT:
    data[dataset_name][column] = EXCEPTIONS_DICT[(dataset_name, column)]

"""
Merge Data
"""
data_merged = pd.DataFrame(index=data["data_common"].index)
for dataset_name in data.keys():
    data_merged = data_merged.join(data[dataset_name], validate="one_to_one")

"""
Shuffle Data
"""
data_merged = shuffle(data_merged)

"""
Export Data
"""
# All data (e.g. for Semi-Supervised Learning)
data_merged.to_csv("data/complete_data/data_complete.csv")

# Only data with GT-Labels (Ischämie?-Column not null)
gt_data = data_merged[~pd.isnull(data_merged["Ischämie?"])]
gt_data.to_csv("data/complete_data/data_complete_gt.csv")
