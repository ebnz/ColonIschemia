import pandas as pd
import numpy as np
from sklearn import model_selection, feature_selection, neural_network, metrics
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

"""
USER_SETTINGS farther down in script
"""

"""
Load Data
"""
data = pd.read_csv("data/complete_data/data_complete_gt.csv")
data = data.set_index("Record ID")

preferred_columns_x = ['ARDS',
                     'Adipositas',
                     'Anti-diabetic medication',
                     'Anticoagulation and platelet inhibitors',
                     'Antidepressants',
                     'Antipsychotics',
                     'Diuretics',
                     'Dyslipidemia',
                     'GIT_comorbidities',
                     'HPB_comorbidities',
                     'Heart insufficency',
                     'Hypothyreosis',
                     'Infection',
                     'Lipid-lowering agents',
                     'Metabolic & endocrinological',
                     'NSAIDs',
                     'Opioids',
                     'Other_cardiovascular_comorbidities',
                     'Others_comorbidities_general',
                     'Others_premedication',
                     'Others_reason_for_admission',
                     'PPI',
                     'Renal',
                     'Sepsis',
                     'Stroke',
                     'Age',
                     'Height',
                     'CT',
                     'Radiological findings?',
                     'Abnormal wall enhancement',
                     'Ascites',
                     'Distension',
                     'Total colonoscopy',
                     'Enteroscopy',
                     'Upper/Lower GI bleeding',
                     'Endoscopic findings?',
                     'Petechiae/Small mucosal erosions',
                     'Ulceration of (sub-)mucosa',
                     'Cyanotic membrane',
                     'Mucosal edema',
                     'Necrosis_endoscopic_findings',
                     'Mucosal inflammation',
                     'Bilirubin day of endoscopy',
                     'Urea day of endoscopy',
                     'Lactate day of endoscopy',
                     'Bicarbonate day of endoscopy',
                     'LDH day of endoscopy',
                     'CPK day of endoscopy',
                     'ALP day of endoscopy',
                     'GGT day of endoscopy',
                     'Hemoglobin day of endoscopy',
                     'Platelet count day of endoscopy',
                     'Oxygen saturation day of endoscopy',
                     'pO2 day of endoscopy',
                     'pCO2 day of endoscopy',
                     'Heart rate day of endoscopy',
                     'ITBVI day of endoscopy',
                     'Volume balance (24hrs) day of endoscopy',
                     'Dialysis/hemofiltration_endoscopy',
                     'Ventilation 3 days before endoscopy',
                     'Dialysis/hemofiltration 3 days before endoscopy']

preferred_columns_y = ['Ischämie?',
                     'Findings compatible with ischemia',
                     'Enterectomy',
                     'Colectomy',
                     'Exploration',
                     'Others_surgery_type_text:']

# Define Training Labels
X = data[preferred_columns_x]

"""
USER-SETTINGS
"""
# Define Train-, Validation-Split Sizes
# VALIDATION_SIZE is relative to size of Complete Dataset minus Datarows of Test-Set
TEST_SIZE = 0.23
VALIDATION_SIZE = 0.23

# Max Epochs for NN Training
MAX_ITER_NN = 300

# Feature-Selection Values
acc_fs = []
f1_fs = []
prec_fs = []
rec_fs = []

# Hyperparameter Opt. Values
acc_hp = []
f1_hp = []
prec_hp = []
rec_hp = []

# Dict for Feature-Usage
selected_feature_frequency = {}

for RANDOM_STATE in range(10):
    # Define Training Target
    y = data["Ischämie?"]
    #y = data["Ischämie?"] == data["Findings compatible with ischemia"]
    #y = (data["Enterectomy"].astype(bool) | data["Colectomy"].astype(bool))
    #y = (data["Enterectomy"].astype(bool) | data["Colectomy"].astype(bool)) | (data["Exploration"].astype(bool) & ~(data["Others_surgery_type_text:"].isin(["Abdo Mac", "AbdoMAC", "AbdoVac"])).astype(bool))

    # Number of Features to Select
    N_FEATURES_TO_SELECT = 4

    """Feature-Selection"""
    # early_stopping_rounds in Feature-Selection
    EARLY_STOPPING_ROUNDS_FEATURE_SELECTION = 3

    # Initial Hyperparameters for Feature-Selection
    LEARNING_RATE_INIT = 0.15
    MAX_DEPTH_INIT = 3

    """Hyperparameter-Selection"""
    # Number of Runs of the Hyperparameter-Optimization
    HP_OPTIMIZATION_ITERATIONS = 20

    # Hyperparameters and their Values to optimize
    HYPERPARAM_SPACE = {
            'learning_rate_init': np.linspace(0.01, 0.2, 10),
            'hidden_layer_sizes': [
                (4, 4, 4),
                (8, 4),
                (4, 4)
            ]
    }

    """
    Generate Dataset-Splits
    """
    # Define Train- and Test-Splits
    # Validation set is handles automatically during early stopping
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                        test_size=TEST_SIZE, random_state=RANDOM_STATE)

    """
    Model-based Feature-Selection
    """
    # Define Model and SequentialFeatureSelector for selecting best Features from Dataset
    model = neural_network.MLPClassifier(
        hidden_layer_sizes=(8, 4),
        learning_rate_init=LEARNING_RATE_INIT,
        max_iter=MAX_ITER_NN,
        random_state=RANDOM_STATE
    )

    selector = feature_selection.SequentialFeatureSelector(
        model,
        n_features_to_select=N_FEATURES_TO_SELECT,
        cv=3
    )

    # Run Feature-Selection and obtain selected Features
    selector.fit(X_train, y_train)
    selected_features = selector.get_feature_names_out()
    X_selected = X_train[selected_features]

    # Performance after Feature-Selection
    model = neural_network.MLPClassifier(
        hidden_layer_sizes=(8, 4),
        learning_rate_init=LEARNING_RATE_INIT,
        early_stopping=True,
        n_iter_no_change=EARLY_STOPPING_ROUNDS_FEATURE_SELECTION,
        validation_fraction=0.23,
        max_iter=MAX_ITER_NN,
        random_state=RANDOM_STATE
    )

    # Sample weights sadly not supported :/
    model.fit(
        X_selected,
        y_train
    )

    # Obtain Results using the Test-Dataset and plot Loss on Validation-Dataset
    y_pred = model.predict(X_test[selected_features])
    acc_fs.append(metrics.accuracy_score(y_test, y_pred))
    f1_fs.append(metrics.f1_score(y_test, y_pred))
    prec_fs.append(metrics.precision_score(y_test, y_pred))
    rec_fs.append(metrics.recall_score(y_test, y_pred))


    """
    Hyperparameter-Selection
    """
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train,
                                                                      test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)

    # Define Model and RandomizedSearchCV with Hyperparameter-Space for Hyperparameter-Optimization
    model = neural_network.MLPClassifier(
        early_stopping=True,
        n_iter_no_change=EARLY_STOPPING_ROUNDS_FEATURE_SELECTION,
        validation_fraction=0.23,
        max_iter=MAX_ITER_NN,
        random_state=RANDOM_STATE
    )

    selector = model_selection.RandomizedSearchCV(
        model,
        HYPERPARAM_SPACE,
        n_iter=HP_OPTIMIZATION_ITERATIONS,
        cv=3,
        random_state=RANDOM_STATE
    )

    selector.fit(
        X_train[selected_features],
        y_train
    )

    # Performance
    model = neural_network.MLPClassifier(
        early_stopping=True,
        n_iter_no_change=EARLY_STOPPING_ROUNDS_FEATURE_SELECTION,
        validation_fraction=0.23,
        max_iter=MAX_ITER_NN,
        random_state=RANDOM_STATE,
        **selector.best_params_
    )
    model.fit(
        X_train[selected_features],
        y_train
    )

    y_pred = model.predict(X_test[selected_features])
    acc_hp.append(metrics.accuracy_score(y_test, y_pred))
    f1_hp.append(metrics.f1_score(y_test, y_pred))
    prec_hp.append(metrics.precision_score(y_test, y_pred))
    rec_hp.append(metrics.recall_score(y_test, y_pred))

    for feature in selected_features:
        if feature in selected_feature_frequency.keys():
            selected_feature_frequency[feature] += 1
        else:
            selected_feature_frequency[feature] = 1

print("Selected Features Frequency: ")
print(selected_feature_frequency)

print("Feature Selection: ")
print(f"ACC: Mean: {np.mean(acc_fs)} | "
      f"+-: {np.std(acc_fs)}")
print(f"F1: Mean: {np.mean(f1_fs)} | "
      f"+-: {np.std(f1_fs)}")
print(f"PREC: Mean: {np.mean(prec_fs)} | "
      f"+-: {np.std(prec_fs)}")
print(f"REC: Mean: {np.mean(rec_fs)} | "
      f"+-: {np.std(rec_fs)}")

print("HP Optimization: ")
print(f"ACC: Mean: {np.mean(acc_hp)} | "
      f"+-: {np.std(acc_hp)}")
print(f"F1: Mean: {np.mean(f1_hp)} | "
      f"+-: {np.std(f1_hp)}")
print(f"PREC: Mean: {np.mean(prec_hp)} | "
      f"+-: {np.std(prec_hp)}")
print(f"REC: Mean: {np.mean(rec_hp)} | "
      f"+-: {np.std(rec_hp)}")
