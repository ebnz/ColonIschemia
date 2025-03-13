# Detecting Colon Ischemia from Clinical Data

<!-- TOC -->
* [Detecting Colon Ischemia from Clinical Data](#detecting-colon-ischemia-from-clinical-data)
  * [What is this Repository?](#what-is-this-repository)
    * [Structure of the `data/`-Directory](#structure-of-the-data-directory)
  * [How to use this Repository](#how-to-use-this-repository)
    * [Subdata-Generation](#subdata-generation)
    * [Data Preprocessing](#data-preprocessing)
    * [Feature-Selection](#feature-selection)
<!-- TOC -->

## What is this Repository?

This Project uses Machine-Learning to detect a possible diagnosis of [Colon Ischemia](https://en.wikipedia.org/wiki/Intestinal_ischemia) from a Hospital's Patient Record. 
To achieve this, we apply multiple Feature-Selection Strategies to (dis-)select Features in the Dataset which may not be relevant to the Problem. 
This repository features multiple Classification-Algorithms, such as XGBoost, LightGBM, RandomForest and TabTransformer to classify the clinical records. 

We mainly focus on three different Classification-Tasks: 
* Predict the disorder of Colon Ischemia 
  * Column "Ischämie?" in surgical data
* Predict the reasonability of Endoscopy for giving a diagnosis on Colon Ischemia
  * Columns "Ischämie?" == "Findings compatible with ischemia" in surgical/endoscopy data
* Predict the severeness of Colon Ischemia
  * Columns ("Enterectomy" $\vee$ "Colectomy") $\vee$ ("Exploration" $\wedge$ "Others_surgery_type_text:" $\notin$ ["Abdo Mac", "AbdoMAC", "AbdoVac"])

### Structure of the `data/`-Directory

| Name of File / Directory | Explanation                                                |
|--------------------------|------------------------------------------------------------|
| `data/raw_data.csv`      | Raw Data as obtained from Data Source                      |
| `data/data.csv`          | Renamed Columns (e.g. Duplicate Column Names)              |
| `data/sub_data/`         | Split up data into chunks (data_common, data_imaging, ...) |
| `data/prep_data/`        | Preprocessed data, ready for Training                      |
| `data/complete_data/`    | Reassembled data                                           |


## How to use this Repository

1. Create a file `data/raw_data.csv` which contains the Raw Clinical Record Data. 
2. Run `pip3 install -r requirements.txt` to install all the dependencies. 
2. Run `python3 data_preprocessing/generate_subdata.py` to generate the Sub-Data. 
3. Run `python3 data_preprocessing/data_preprocessing.py` to fill in Conditional Columns. 
4. Run `python3 data_preprocessing/feature_selection.py` to filter out the (hopefully!) irrelevant Features. 
5. Run `python3 train/train_xgboost.py` (or other Training-Script) to Train a Model. 
6. View the directory `plots/` to review the Training-Results. 
7. Fine-Tune the Models, mutating the Parameters in the Script's Sections called `USER-SETTINGS`. 

| Folder-Name        | Contents                                       |
|--------------------|------------------------------------------------|
| data_preprocessing | Scripts for Preprocessing the Dataset          |
| train              | Scripts for Training of the Models             |
| plots              | Plots with information on the Training-Process |
| data               | Datasets                                       |


### Subdata-Generation

`data/raw_data.csv` contains all the data spread over multiple Rows. To organize this data in a useful manner, 
the script `data_preprocessing/generate_subdata.py` generates Sub-Datasets for the following Categories: 

| Sub-Dataset       | Contents                                            |
|-------------------|-----------------------------------------------------|
| data_common       | Reasons for admission, Comorbidities, Premedication |
| data_personal     | Age, Gender, Height, Weight                         |
| data_imaging      | Computertomography-Data                             |
| data_endo         | Endoscopy-Data (Type, Indication, Findings)         |
| data_lab_endo     | Lab-Data made from blood taken before Endoscopy     |
| data_surgical     | Surgery-Data (Duration, Approach, Findings)         |
| data_lab_surgical | Lab-Data made from blood taken before Surgery       |

The generated Sub-Datasets are saved in the directory `data/sub_data/`. 

### Data Preprocessing

### Feature-Selection



