import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection, metrics, neural_network

import xgboost as xgb
import shap

"""
USER_SETTINGS farther down in script
"""

"""
Load Data
"""
data = pd.read_csv("../data/complete_data/data_complete_gt.csv")
data = data.set_index("Record ID")

preferred_columns_x = ['ARDS',
                       'Adipositas',
                       'Endoscopic findings?',
                       'Anti-diabetic medication']

# Define Training Labels
X = data[preferred_columns_x]

"""
USER-SETTINGS
"""
# Define Train-, Test-Split Sizes
TEST_SIZE = 0.23

# Define Training Target
y_ischemia = data["Ischämie?"]

RANDOM_STATE = 0

"""
Generate Dataset-Splits
"""
# Define Train-, and Test-Splits
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_ischemia,
                                                                    test_size=TEST_SIZE, random_state=RANDOM_STATE)

"""
SHAP-Analysis XGBoost
Task 1: Detecting Colon Ischemia
"""
model = xgb.XGBClassifier(
    objective="binary:logistic",
    max_depth=3,
    learning_rate=0.15,
    subsample=0.8,
    seed=RANDOM_STATE
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
prec = metrics.precision_score(y_test, y_pred)
rec = metrics.recall_score(y_test, y_pred)

print("Performance: ")
print(f"ACC: {accuracy}")
print(f"F1: {f1}")
print(f"PREC: {prec}")
print(f"REC: {rec}")

replacements = {
    "Endoscopic findings?": "Endoscopic findings"
}
columns = [replacements[item] if item in replacements.keys() else item for item in X_test.columns.tolist()]

shap_values = shap.TreeExplainer(model).shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=columns, show=False)
plt.xlabel("mean(|SHAP value|) (average impact on model output)")
plt.savefig("../plots/shap_impact_detecting_ischemia.png")
plt.clf()

"""
SHAP-Analysis Neural Network
Task 2: Endoscopic Expressiveness
"""
preferred_columns_x = ['ARDS',
                       'Adipositas',
                       'GIT_comorbidities',
                       'Sepsis']

# Define Training Labels
X = data[preferred_columns_x]

# Define Training Target
y_endo_expressiveness = (data["Ischämie?"] == data["Findings compatible with ischemia"])

# Define Train-, and Test-Splits
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_endo_expressiveness,
                                                                    test_size=TEST_SIZE, random_state=RANDOM_STATE)

model = neural_network.MLPClassifier(
        early_stopping=True,
        n_iter_no_change=3,
        validation_fraction=0.23,
        max_iter=300,
        random_state=42,
        learning_rate_init=0.03111111,
        hidden_layer_sizes=(8, 4)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
prec = metrics.precision_score(y_test, y_pred)
rec = metrics.recall_score(y_test, y_pred)

print("Performance: ")
print(f"ACC: {accuracy}")
print(f"F1: {f1}")
print(f"PREC: {prec}")
print(f"REC: {rec}")

replacements = {
    "ARDS": "Reason for Admission: ARDS",
    "Sepsis": "Reason for Admission: Sepsis",
    "GIT_comorbidities": "Comorbidities: Gastrointestinal"
}
columns = [replacements[item] if item in replacements.keys() else item for item in X_test.columns.tolist()]

explainer = shap.KernelExplainer(model.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[:, :, 1], plot_type="bar", feature_names=columns, show=False)
plt.xlabel("mean(|SHAP value|) (average impact on model output)")
plt.xticks([0.005, 0.01, 0.015, 0.02])
plt.savefig("../plots/shap_impact_endoscopy_expressiveness.png")
plt.clf()
