import ehrapy as ep
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

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

data_common["Repeat Instance"] = 1
data_personal["Repeat Instance"] = 1

data_common["Repeat Instrument"] = ""
data_personal["Repeat Instrument"] = ""

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
For each Repeat ID with multiple Repeat Instances, combine the Entries preferring the first occurrence of a Value
"""
for dataset_name in data.keys():
    data[dataset_name] = data[dataset_name].groupby("Record ID").aggregate("first")
    data[dataset_name] = data[dataset_name].reset_index().set_index("Record ID")
    data[dataset_name].drop(["Unnamed: 0", "Repeat Instance", "Repeat Instrument"], axis=1, inplace=True)

"""
Cell-Mappings
"""
CELL_MAPPING = {
    ",": ".",
    "No": "0",
    "Yes": "1",
    "Male": "0",
    "Female": "1",
    "Extensive (Two or more segments)": "2",
    "Segmental (One segment)": "1"
}

COLUMNS_DROP_EXCEPTIONS = [
    "Others_surgery_type_text:",
    "Enterectomy",
    "Ischämie?"
]

def map_cell(cell_value):
    """
    Applies all Replacement-Rules specified in CELL_MAPPING to cell_value.
    :param cell_value: Value to apply the Replacement-Rules to
    :return: cell_value with applied Rules
    """
    if pd.isnull(cell_value):
        return cell_value
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
Merge Data
Merge all Sub-Datasets back together
"""
data_merged = pd.DataFrame(index=data["data_common"].index)
for dataset_name in data.keys():
    data_merged = data_merged.join(data[dataset_name], validate="one_to_one")

"""
Shuffle Data
"""
data_merged = shuffle(data_merged)

data_merged = data_merged[~pd.isnull(data_merged["Ischämie?"])]

data_merged["Ischämie?"] = data_merged["Ischämie?"].astype(dtype=str, errors="raise")

data_merged.drop("Height/weight/BMI nicht vorhanden", axis=1, inplace=True)

adata = ep.ad.df_to_anndata(data_merged, index_column="Record ID", columns_obs_only=["Ischämie?"])

ep.ad.infer_feature_types(adata)
ep.ad.replace_feature_types(
    adata,
    ["Amount of PC units day of surgery",
    "Amount of PRBC units 1 day before surgery",
    "Amount of PRBC units day of endoscopy",
    "Number of CT scans (before endoscopy)",
    "Number of X-rays (before endoscopy)"],
    "numeric"
)

ep.ad.feature_type_overview(adata)

adata = ep.pp.encode(adata, autodetect=True)

#ep.pl.violin(adata, keys=["Age", "Weight ", "BMI", "Gender"], groupby="Ischämie?")

obs_metric, var_metrics = ep.pp.qc_metrics(adata)

adata = adata[::, var_metrics["missing_values_pct"] <= 70]

ep.pp.simple_impute(adata, strategy="median")
#ep.pp.knn_impute(adata, n_neighbours=5)

obs_metric, var_metrics = ep.pp.qc_metrics(adata)

print(var_metrics["missing_values_pct"].max())

# Plot PCA
#ep.pp.pca(adata, n_comps=5)
#ep.pl.pca(adata, color="Ischämie?", components=["1,2", "3,4"])

ep.pp.neighbors(adata, n_pcs=10)
ep.tl.umap(adata)

ep.pl.umap(
    adata,
    color=[
        "Ischämie?",
        "Gender"
    ]
)