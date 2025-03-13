import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, semi_supervised

import xgboost as xgb

"""
USER_SETTINGS farther down in script
"""

"""
Load Data
"""
data = pd.read_csv("../data/complete_data/data_complete.csv")
data = data.set_index("Record ID")

preferred_columns_x = ['ARDS',
                       'Adipositas',
                       'Endoscopic findings?',
                       'Anti-diabetic medication']

preferred_columns_y = ['Ischämie?',
                     'Findings compatible with ischemia',
                     'Enterectomy',
                     'Colectomy',
                     'Exploration',
                     'Others_surgery_type_text:']

data_gt = data[~data["Ischämie?"].isnull()]

# Define Training Labels
X = data[preferred_columns_x]
X_gt = data_gt[preferred_columns_x]

"""
USER-SETTINGS
"""
# Define Train-, Test-Split Sizes
TEST_SIZE = 0.23

# Vanilla Values
acc_vanilla = []
f1_vanilla = []
prec_vanilla = []
rec_vanilla = []

# Semi-Supervised Values
acc_semi = []
f1_semi = []
prec_semi = []
rec_semi = []

for RANDOM_STATE in range(50):
    # Define Training Target
    y = data["Ischämie?"]
    y_gt = data_gt["Ischämie?"]

    """
    Generate Dataset-Splits
    """
    # Define Train-, and Test-Splits

    _, X_test, _, y_test = model_selection.train_test_split(X_gt, y_gt,
                                                            test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Drop Test-/Val-Data from Training Set
    X_train_semi_superv, y_train_semi_superv = X.drop(X_test.index), y.drop(y_test.index)

    y_train_vanilla = y_train_semi_superv.dropna()
    X_train_vanilla = X_train_semi_superv.loc[y_train_vanilla.index]

    y_train_semi_superv.fillna(-1, inplace=True)


    """
    Train Semi-Supervised
    """
    print("Using Vanilla Approach")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=3,
        learning_rate=0.15,
        subsample=0.8,
        seed=RANDOM_STATE
    )

    model.fit(
        X_train_vanilla,
        y_train_vanilla
    )

    y_pred = model.predict(X_test)
    acc_vanilla.append(metrics.accuracy_score(y_test, y_pred))
    f1_vanilla.append(metrics.f1_score(y_test, y_pred))
    prec_vanilla.append(metrics.precision_score(y_test, y_pred))
    rec_vanilla.append(metrics.recall_score(y_test, y_pred))

    print("Using Semi-Supervised Approach")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=3,
        learning_rate=0.15,
        subsample=0.8,
        seed=RANDOM_STATE
    )

    semi_supervised_classifier = semi_supervised.SelfTrainingClassifier(
        model,
        criterion="k_best",
        k_best=20,
        verbose=True
    )

    semi_supervised_classifier.fit(
        X_train_semi_superv,
        y_train_semi_superv
    )

    y_pred = semi_supervised_classifier.predict(X_test)
    acc_semi.append(metrics.accuracy_score(y_test, y_pred))
    f1_semi.append(metrics.f1_score(y_test, y_pred))
    prec_semi.append(metrics.precision_score(y_test, y_pred))
    rec_semi.append(metrics.recall_score(y_test, y_pred))

print("Vanilla Performance: ")
print(f"ACC: Mean: {np.mean(acc_vanilla)} | "
      f"+-: {np.std(acc_vanilla)}")
print(f"F1: Mean: {np.mean(f1_vanilla)} | "
      f"+-: {np.std(f1_vanilla)}")
print(f"PREC: Mean: {np.mean(prec_vanilla)} | "
      f"+-: {np.std(prec_vanilla)}")
print(f"REC: Mean: {np.mean(rec_vanilla)} | "
      f"+-: {np.std(rec_vanilla)}")

print("Semi-Supervised Performance: ")
print(f"ACC: Mean: {np.mean(acc_semi)} | "
      f"+-: {np.std(acc_semi)}")
print(f"F1: Mean: {np.mean(f1_semi)} | "
      f"+-: {np.std(f1_semi)}")
print(f"PREC: Mean: {np.mean(prec_semi)} | "
      f"+-: {np.std(prec_semi)}")
print(f"REC: Mean: {np.mean(rec_semi)} | "
      f"+-: {np.std(rec_semi)}")
