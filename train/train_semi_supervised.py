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
                       'Distension',
                       'Endoscopic findings?',
                       'CPK day of endoscopy']

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
TEST_SIZE = 0.3

# Define Random State
RANDOM_STATE = 64

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
    learning_rate=0.35,
    subsample=1.0,
    seed=RANDOM_STATE
)

model.fit(
    X_train_vanilla,
    y_train_vanilla
)

y_pred = model.predict(X_test)
test_accuracy = metrics.accuracy_score(y_test, y_pred)
test_f1 = metrics.f1_score(y_test, y_pred)
test_precision = metrics.precision_score(y_test, y_pred)
test_recall = metrics.recall_score(y_test, y_pred)

print(f"ACC:  {test_accuracy}")
print(f"F1:   {test_f1}")
print(f"PREC: {test_precision}")
print(f"REC:  {test_recall}")


print("Using Semi-Supervised Approach")

model = xgb.XGBClassifier(
    objective="binary:logistic",
    max_depth=3,
    learning_rate=0.19,
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
test_accuracy = metrics.accuracy_score(y_test, y_pred)
test_f1 = metrics.f1_score(y_test, y_pred)
test_precision = metrics.precision_score(y_test, y_pred)
test_recall = metrics.recall_score(y_test, y_pred)

print(f"ACC:  {test_accuracy}")
print(f"F1:   {test_f1}")
print(f"PREC: {test_precision}")
print(f"REC:  {test_recall}")