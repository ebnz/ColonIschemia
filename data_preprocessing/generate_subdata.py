import copy
import pandas as pd
from itertools import product

data = pd.read_csv("data/data.csv", dtype=str)

"""
Split data into subdata-Categories
"""
"""
data_common
"""
# Get all columns where Repeat Instrument is empty (no clinical data in this row)
data_common = data[data["Repeat Instrument"].isnull()]

# Drop all Columns only containing NaN
data_common.dropna(how="all", axis=1, inplace=True)

"""
data_personal
"""
# Split up data_common into data_common and data_personal
data_personal_columns = pd.Index([
    "Record ID",
    "Age",
    "Gender",
    "Height",
    "Weight ",
    "BMI",
    "Height/weight/BMI nicht vorhanden",
    "Complete?"])
data_common_columns = data_common.columns.difference(data_personal_columns)

# Record ID is Dataset-Indexing, must be in both datasets
data_common_columns = data_common_columns.append(pd.Index(["Record ID"]))

data_personal = data_common[data_personal_columns]
data_common = data_common[data_common_columns]

"""
data_imaging
"""
data_imaging = data[data["Repeat Instrument"] == "Imaging"]
data_imaging.dropna(how="all", axis=1, inplace=True)


"""
data_endo
"""
data_endo = data[data["Repeat Instrument"] == "Endoscopic data and findings"]
data_endo.dropna(how="all", axis=1, inplace=True)

"""
data_lab_endo
"""
data_lab_endo = data[data["Repeat Instrument"] == "Lab Data (endoscopy)"]
data_lab_endo.dropna(how="all", axis=1, inplace=True)

"""
data_surgical
"""
data_surgical = data[data["Repeat Instrument"] == "Surgical data and findings"]
data_surgical.dropna(how="all", axis=1, inplace=True)

"""
data_lab_surgical
"""
data_lab_surgical = data[data["Repeat Instrument"] == "Lab data (surgery)"]
data_lab_surgical.dropna(how="all", axis=1, inplace=True)

"""
Fill in Conditional Columns
"""
def apply_implicit_data_rule(df, conditional_column, implied_columns, conditional_column_value, implied_columns_value):
    """
    Applies an implicit Data Rule "If Column conditional_column == conditional_column_value,
    then set implied_columns = implied_columns_value"
    Operates __in-place__
    Parameters:
    df: pd.DataFrame - The DataFrame to operate on
    conditional_column: str - Conditional Column that has to be met
    implied_columns: list - Columns that are mutated if condition is met
    conditional_column_value: str - Value that the Conditional Column has to equal
    implied_columns_value: str - Value to which the implied Columns are set if condition is met
    """
    for implied_column in implied_columns:
        # Rows where the Condition holds True
        row_mask = df[conditional_column] == conditional_column_value
        df[implied_column] = df[implied_column].mask(row_mask, implied_columns_value)

"""
Comorbidities? = No
=>
Chronic infection, Substance abuse, HPB_comorbidities, GIT_comorbidities, Cardiovascular, Metabolic & endocrinological, 
Pulmonary, Renal, Autoimmune & rheumatic, Neurological & psychatric, Transplantation, Others = No
"""
conditional_column = "Comorbidities?"
implied_columns = ["Chronic infection", "Substance abuse", "HPB_comorbidities", "GIT_comorbidities", "Cardiovascular", "Metabolic & endocrinological", "Pulmonary", "Renal", "Autoimmune & rheumatic", "Neurological & psychatric", "Transplantation", "Others"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Chronic infection = No 
=>
Hepatitis B, Hepatitis C, HIV, TBC, Other_chronic_infection = No
"""
conditional_column = "Chronic infection"
implied_columns = ['Hepatitis B', 'Hepatitis C', 'HIV', 'TBC', 'Other_chronic_infection']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Substance abuse = No
=>
Nicotine, Alcohol, Medication (e.g. benzodiazepine, opioids), Drugs (e.g. cocaine, heroin), Others_substance_abuse = No
"""
conditional_column = "Substance abuse"
implied_columns = ['Nicotine', 'Alcohol', 'Medication (e.g. benzodiazepine, opioids)', 'Drugs (e.g. cocaine, heroin)', 'Others_substance_abuse']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
HPB_comorbidities = No 
=>
Hepatic steatosis, Haemochromatosis/Morbus Wilson, Pancreatitis, Biliary (cholelithiasis etc.), PBC, PSC, Others_HPB_comorbidities, Liver cirrhosis? = No
"""
conditional_column = "HPB_comorbidities"
implied_columns = ['Hepatic steatosis',
 'Haemochromatosis/Morbus Wilson',
 'Pancreatitis',
 'Biliary (cholelithiasis etc.)',
 'PBC',
 'PSC',
 'Others_HPB_comorbidities',
 'Liver cirrhosis?']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
GIT_comorbidities = No
=>
Reflux disease, Gastritis, Coeliac disease, Inflammatory bowel disease, Diverticulitis, Other_GIT_comorbidities = No
"""
conditional_column = "GIT_comorbidities"
implied_columns = ['Reflux disease',
 'Gastritis',
 'Coeliac disease',
 'Inflammatory bowel disease',
 'Diverticulitis',
 'Other_GIT_comorbidities']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Cardiovascular = No
=>
Atrial fibrillation, Arterial hypertension, Coronary artery disease, Heart insufficency, Heart valve disease, Peripheral artery disease, Chronic venous insufficiency, Other_cardiovascular_comorbidities = No
"""
conditional_column = "Cardiovascular"
implied_columns = ['Atrial fibrillation',
 'Arterial hypertension',
 'Coronary artery disease',
 'Heart insufficency',
 'Heart valve disease',
 'Peripheral artery disease',
 'Chronic venous insufficiency',
 'Other_cardiovascular_comorbidities']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Metabolic & endocrinological = No
=>
Hypothyreosis, Hyperthyreosis, Diabetes mellitus, Disorders of hypothalamic-pituitary-adrenal axis, Malnutrition, Vitamins & trace elements deficiency, Adipositas, Dyslipidemia, Metabolic syndrome, Other_metabolic_endocrino_comorbidities = No
"""
conditional_column = "Metabolic & endocrinological"
implied_columns = ['Hypothyreosis',
 'Hyperthyreosis',
 'Diabetes mellitus',
 'Disorders of hypothalamic-pituitary-adrenal axis',
 'Malnutrition',
 'Vitamins & trace elements deficiency',
 'Adipositas',
 'Dyslipidemia',
 'Metabolic syndrome',
 'Other_metabolic_endocrino_comorbidities']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Pulmonary = No
=>
COPD, Chronic bronchitis, Bronchial asthma, Cystic fibrosis, Silicosis, Other_pulmonary_comorbidities = No
"""
conditional_column = "Pulmonary"
implied_columns = ['COPD',
 'Chronic bronchitis',
 'Bronchial asthma',
 'Cystic fibrosis',
 'Silicosis',
 'Other_pulmonary_comorbidities']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Renal = No
=>
Glomerular diseases, Tubulointerstitial diseases, Renal insufficiency, Cystic kidney disease, Urolithiasis, Other_renal_urogenital_comorbidities, Dialysis? = No
"""
conditional_column = "Renal"
implied_columns = ['Glomerular diseases',
 'Tubulointerstitial diseases',
 'Renal insufficiency',
 'Cystic kidney disease',
 'Urolithiasis',
 'Other_renal_urogenital_comorbidities',
 'Dialysis?']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Autoimmune & rheumatic = No
=>
Allergy, Collagenosis, Sarcoidosis, Bechterew's disease, Other_autoimmune_rheumatic_comorbidities = No
"""
conditional_column = "Autoimmune & rheumatic"
implied_columns = ['Allergy',
 'Collagenosis',
 'Sarcoidosis',
 "Bechterew's disease",
 'Other_autoimmune_rheumatic_comorbidities']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Neurological & psychatric = No
=>
Epilepsy, Parkinson's disease, Dementia, Multiple sclerosis, Depression, Psychosis/Schizophrenia, Anxiety disorder, Eating disorder, Other_neurological_psychiatric_comorbidities = No
"""
conditional_column = "Neurological & psychatric"
implied_columns = ['Epilepsy',
 "Parkinson's disease",
 'Dementia',
 'Multiple sclerosis',
 'Depression',
 'Psychosis/Schizophrenia',
 'Anxiety disorder',
 'Eating disorder',
 'Other_neurological_psychiatric_comorbidities']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Transplantation = No
=>
Kidney, Liver, Heart, Lung, Stem cells, Other_transplantations = No
"""
conditional_column = "Transplantation"
implied_columns = ['Kidney_transplantation', 'Liver', 'Heart', 'Lung_transplantation', 'Stem cells', 'Other_transplantations']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)


"""
Premedication? = No
=>
Unknown_premedication, Anticoagulation and platelet inhibitors, Antihypertensive medication, Diuretics, 
Lipid-lowering agents, NSAIDs, PPI, Opioids, Anti-diabetic medication, Insulin, Corticosteroids, 
Immunosuppressive agents, Bronchodilators, Antidepressants, Antipsychotics, Anticonvulsants, Others = No
"""
conditional_column = "Premedication?"
implied_columns = ["Unknown_premedication", "Anticoagulation and platelet inhibitors", "Antihypertensive medication", "Diuretics", "Lipid-lowering agents", "NSAIDs", "PPI", "Opioids", "Anti-diabetic medication", "Insulin", "Corticosteroids", "Immunosuppressive agents", "Bronchodilators", "Antidepressants", "Antipsychotics", "Anticonvulsants", "Others"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Malignancy = No
=>
GIT_malignancy, HPB_malignancy, Urogenital_malignancy, Lung, Breast, Brain, Hemato, Skin, Others_malignancy = No
"""
conditional_column = "Malignancy"
implied_columns = ["GIT_malignancy", "HPB_malignancy", "Urogenital_malignancy", "Lung_malignancy", "Breast", "Brain", "Hemato", "Skin", "Others_malignancy"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
GIT_malignancy = No
=>
Oesophageal, Gastric, Small intestine, Colon, Sigma, Rectum = No
"""
conditional_column = "GIT_malignancy"
implied_columns = ["Esophageal", "Gastric", "Small intestine", "Colon", "Sigma", "Rectum_malignancy"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
HPB_malignancy = No => Hepatic, Pancreatic, Biliary = No
"""
conditional_column = "HPB_malignancy"
implied_columns = ["Hepatic", "Pancreatic", "Biliary"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Urogenital_malignancy = No => Kidney, Urinary tract & bladder, Prostate, Testicles, Cervix, Endometrium, Ovary = No
"""
conditional_column = "Urogenital_malignancy"
implied_columns = ["Kidney_malignancy", "Urinary tract & bladder", "Prostate", "Testicles", "Cervix", "Endometrium", "Ovary"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_common, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
IMAGING
"""
"""
Was an abdominal imaging performed prior to this endoscopy [start_endo]? = No
=>
CT, X-ray, Number of CT scans (before endoscopy), Number of X-rays (before endoscopy) = No
"""
conditional_column = "Was an abdominal imaging performed prior to this endoscopy [start_endo]?"
implied_columns = ["CT", "X-ray"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_imaging, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
CT = "No"
=>
Number of CT scans (before endoscopy) = 0.0
"""
conditional_column = "CT"
implied_columns = ["Number of CT scans (before endoscopy)"]

conditional_column_value = "No"
implied_columns_value = "0.0"

apply_implicit_data_rule(data_imaging, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
X-ray = "No"
=>
Number of X-rays (before endoscopy) = 0.0
"""
conditional_column = "X-ray"
implied_columns = ["Number of X-rays (before endoscopy)"]

conditional_column_value = "No"
implied_columns_value = "0.0"

apply_implicit_data_rule(data_imaging, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Radiological findings? = No
=>
Abnormal wall enhancement, Pneumatosis intestinalis, Ogilvie syndrome, Portal venous gas, Atherosclerosis, Perforation_radiological_findings, Ascites, Distension, Coprostasis, Other_radiological_findings = No
"""
conditional_column = "Radiological findings?"
implied_columns = ["Abnormal wall enhancement", "Pneumatosis intestinalis", "Ogilvie syndrome", "Portal venous gas", "Atherosclerosis", "Perforation_radiological_findings", "Ascites", "Distension", "Coprostasis", "Other_radiological_findings"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_imaging, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
ENDO
"""
"""
Endoscopic findings? = No
=>
Transmural ischemia, Petechiae/Small mucosal erosions, Haemorrhagic submucosal nodules, Ulceration of (sub-)mucosa, Cyanotic membrane, Mucosal edema, Pseudopolyps, Ulceration with possible perforation, Necrosis_endoscopic_findings, Mucosal inflammation, Others_endoscopic_findings = No
"""
conditional_column = "Endoscopic findings?"
implied_columns = ["Transmural ischemia", "Petechiae/Small mucosal erosions", "Haemorrhagic submucosal nodules", "Ulceration of (sub-)mucosa", "Cyanotic membrane", "Mucosal edema", "Pseudopolyps", "Ulceration with possible perforation", "Necrosis_endoscopic_findings", "Mucosal inflammation", "Others_endoscopic_findings"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_endo, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
LAB ENDO
"""
"""
Dialysis/hemofiltration_endoscopy = No
=>
Dialysis/hemofiltration x days before/after endoscopy = No
"""
conditional_column = "Dialysis/hemofiltration_endoscopy"
implied_columns = ['Dialysis/hemofiltration 3 days before endoscopy',
 'Dialysis/hemofiltration 2 days before endoscopy',
 'Dialysis/hemofiltration 1 day before endoscopy',
 'Dialysis/hemofiltration day of endoscopy',
 'Dialysis/hemofiltration 1 day after endoscopy',
 'Dialysis/hemofiltration 2 days after endoscopy',
 'Dialysis/hemofiltration 3 days after endoscopy']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_lab_endo, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Ventilation_endoscopy = No
=>
ECMO_endoscopy, Other_Ventilation, Ventilation x days before/after endoscopy, ECMO x days before/after endoscopy NaN = No
"""
conditional_column = "Ventilation_endoscopy"
implied_columns = ["ECMO_endoscopy", "Other_Ventilation_endoscopy",
 'Ventilation 3 days before endoscopy',
 'Ventilation 2 days before endoscopy',
 'Ventilation 1 day before endoscopy',
 'Ventilation day of endoscopy',
 'Ventilation 1 day after endoscopy',
 'Ventilation 2 days after endoscopy',
 'Ventilation 3 days after endoscopy',
 'ECMO 3 days before endoscopy',
 'ECMO 2 days before endoscopy',
 'ECMO 1 day before endoscopy',
 'ECMO day of endoscopy',
 'ECMO 1 day after endoscopy',
 'ECMO 2 days after endoscopy',
 'ECMO 3 days after endoscopy']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_lab_endo, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Blood transfusions before/after endoscopy? [start_endo] = No
=>
PRBC_endoscopy, FFP_endoscopy, Platelet concentrate (PC)_endoscopy = No 
"""
conditional_column = "Blood transfusions before/after endoscopy? [start_endo]"
implied_columns = ["PRBC_endoscopy", "FFP_endoscopy", "Platelet concentrate (PC)_endoscopy"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_lab_endo, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Blood transfusions before/after endoscopy? [start_endo] = No
=>
Sum of PRBC units before endoscopy, Sum of PRBC units after endoscopy, Total sum of PRBC before/after endoscopy, Sum of FFP units before endoscopy, Sum of FFP units after endoscopy, Total sum of FFP before/after endoscopy, Sum of PC units before endoscopy, Sum of PC units after endoscopy, Total sum of PC before/after endoscopy
"""
conditional_column = "Blood transfusions before/after endoscopy? [start_endo]"
implied_columns = ['Amount of PRBC units 3 days before endoscopy',
 'Amount of PRBC units 2 days before endoscopy',
 'Amount of PRBC units 1 day before endoscopy',
 'Amount of PRBC units day of endoscopy',
 'Amount of PRBC units 1 day after endoscopy',
 'Amount of PRBC units 2 days after endoscopy',
 'Amount of PRBC units 3 days after endoscopy',
 'Amount of FFP units 3 days before endoscopy',
 'Amount of FFP units 2 days before endoscopy',
 'Amount of FFP units 1 day before endoscopy',
 'Amount of FFP units day of endoscopy',
 'Amount of FFP units 1 day after endoscopy',
 'Amount of FFP units 2 days after endoscopy',
 'Amount of FFP units 3 days after endoscopy',
 'Amount of PC units 3 days before endoscopy',
 'Amount of PC units 2 days before endoscopy',
 'Amount of PC units 1 day before endoscopy',
 'Amount of PC units day of endoscopy',
 'Amount of PC units 2 days after endoscopy',
 'Amount of PC units 3 days after endoscopy']

conditional_column_value = "No"
implied_columns_value = "0.0"

apply_implicit_data_rule(data_lab_endo, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
PRBC_endoscopy = No
=>
Sum of PRBC units before endoscopy, Sum of PRBC units after endoscopy, Total sum of PRBC before/after endoscopy = 0
"""
conditional_column = "PRBC_endoscopy"
implied_columns = ["Sum of PRBC units before endoscopy", "Sum of PRBC units after endoscopy", "Total sum of PRBC before/after endoscopy"]

conditional_column_value = "No"
implied_columns_value = "0.0"

apply_implicit_data_rule(data_lab_endo, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
FFP_endoscopy = No
=>
Sum of FFP units before endoscopy, Sum of FFP units after endoscopy, Total sum of FFP before/after endoscopy = 0
"""
conditional_column = "FFP_endoscopy"
implied_columns = ["Sum of FFP units before endoscopy", "Sum of FFP units after endoscopy", "Total sum of FFP before/after endoscopy"]

conditional_column_value = "No"
implied_columns_value = "0.0"

apply_implicit_data_rule(data_lab_endo, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Platelet concentrate (PC)_endoscopy = No
=>
Sum of PC units before endoscopy, Sum of PC units after endoscopy, Total sum of PC before/after endoscopy = 0
"""
conditional_column = "Platelet concentrate (PC)_endoscopy"
implied_columns = ["Sum of PC units before endoscopy", "Sum of PC units after endoscopy", "Total sum of PC before/after endoscopy"]

conditional_column_value = "No"
implied_columns_value = "0.0"

apply_implicit_data_rule(data_lab_endo, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Was an abdominal operation performed after endoscopy [start_endo]? = No
=>
Open, Laparoscopic, Other_surgery_approach, Exploration, Enterectomy, Colectomy, Other_surgery_type = No
"""
conditional_column = "Was an abdominal operation performed after endoscopy [start_endo]?"
implied_columns = ["Open", "Laparoscopic", "Other_surgery_approach", "Exploration", "Enterectomy", "Colectomy", "Other_surgery_type"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_surgical, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Was an abdominal operation performed after endoscopy [start_endo]? = No
=>
Duration of operation = 0
"""
conditional_column = "Was an abdominal operation performed after endoscopy [start_endo]?"
implied_columns = ["Duration of operation"]

conditional_column_value = "No"
implied_columns_value = "0.0"

apply_implicit_data_rule(data_surgical, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Surgical findings? = No
=>
Non-necrotic ischemic lesions, Peritonitis, Perforation_surgical_findings, Necrosis_surgical_findings, Peritoneal effusion/Ascites, Other_surgical_findings = No
"""
conditional_column = "Surgical findings?"
implied_columns = ["Non-necrotic ischemic lesions", "Peritonitis", "Perforation_surgical_findings", "Necrosis_surgical_findings", "Peritoneal effusion/Ascites", "Other_surgical_findings"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_surgical, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
LAB SURGICAL
"""
"""
Dialysis/hemofiltration_surgery = No
=>
Dialysis/hemofiltration x days before/after surgery = No
"""
conditional_column = "Dialysis/hemofiltration_surgery"
implied_columns = ['Dialysis/hemofiltration 3 days before surgery',
 'Dialysis/hemofiltration 2 days before surgery',
 'Dialysis/hemofiltration 1 day before surgery',
 'Dialysis/hemofiltration day of surgery']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_lab_surgical, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Ventilation_surgery = No
=>
ECMO_surgery, Other_Ventilation, Ventilation x days before/after surgery, ECMO x days before/after surgery = No
"""
conditional_column = "Ventilation_surgery"
implied_columns = ["ECMO_surgery", "Other_Ventilation_surgery",
 'Ventilation 3 days before surgery',
 'Ventilation 2 days before surgery',
 'Ventilation 1 day before surgery',
 'Ventilation day of surgery',
 'ECMO 3 days before surgery',
 'ECMO 2 days before surgery',
 'ECMO 1 day before surgery',
 'ECMO day of surgery']

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_lab_surgical, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Blood transfusions before surgery? [start_op] = No
=>
PRBC_surgery, FFP_surgery, Platelet concentrate (PC)_surgery = No
"""
conditional_column = "Blood transfusions before surgery? [start_op]"
implied_columns = ["PRBC_surgery", "FFP_surgery", "Platelet concentrate (PC)_surgery"]

conditional_column_value = "No"
implied_columns_value = "No"

apply_implicit_data_rule(data_lab_surgical, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Blood transfusions before surgery? [start_op] = No
=>
Amount of PRBC units x days before/after surgery, Amount of FFP units x days before/after surgery, Amount of PC units x days before/after surgery = 0
"""
conditional_column = "Blood transfusions before surgery? [start_op]"
implied_columns = ['Amount of PRBC units 3 days before surgery',
 'Amount of PRBC units 2 days before surgery',
 'Amount of PRBC units 1 day before surgery',
 'Amount of PRBC units day of surgery',
 'Amount of FFP units 3 days before surgery',
 'Amount of FFP units 1 day before surgery',
 'Amount of FFP units day of surgery',
 'Amount of PC units day of surgery']

conditional_column_value = "No"
implied_columns_value = "0.0"

apply_implicit_data_rule(data_lab_surgical, conditional_column, implied_columns, conditional_column_value, implied_columns_value)

"""
Select Blood Value with most datapoints
"""
lab_suffixes = [
    "Albumin",
    "Bilirubin",
    "Urea",
    "Creatinine",
    "GFR",
    "Potassium",
    "Lactate",
    "Bicarbonate",
    "pH",
    "CRP",
    "PCT",
    "LDH",
    "CPK",
    "ALP",
    "GGT",
    "ALT",
    "AST",
    "White blood cell count",
    "Hemoglobin",
    "Platelet count",
    "Quick",
    "INR",
    "Oxygen saturation",
    "pO2",
    "pCO2",
    "Body temperature",
    "Heart rate",
    "MAP",
    "Cardiac index",
    "Cardiac output",
    "ITBVI",
    "IAP",
    "Volume balance (24hrs)",
    "Ventilation",
    "PEEP",
    "ECMO",
    "Dialysis/hemofiltration",
    "Amount of PRBC units",
    "Sum of PRBC units",
    "Amount of FFP units",
    "Sum of FFP units",
    "Amount of PC units",
    "Sum of PC"
]

lab_labels_endo = {}
lab_labels_surgical = {}

# Set lab_labels_endo's and lab_labels_surgical's keys to lab_suffixes and an empty list as values
for item in lab_suffixes:
    lab_labels_endo[item] = []
    lab_labels_surgical[item] = []


"""
For data_lab_endo
"""
# Sort all blood_markers to their corresponding lab_label
# e.g. Bilirubin -> Bilirubin 3 days before endo, Bilirubin 2 days before endo, ...
for item, proposal in product(data_lab_endo.columns, lab_suffixes):
    if proposal in item and "day" in item:
        lab_labels_endo[proposal].append(item)

for key in lab_labels_endo.keys():
    # If only one Record for a blood marker, skip this blood marker
    if len(lab_labels_endo[key]) < 1:
        continue

    # Else use Record of blood marker with most datapoints
    blood_marker_records = lab_labels_endo[key]
    column_name_with_max_values = data_lab_endo[blood_marker_records].count().sort_values(ascending=False).index[0]

    # Drop all other Records of blood marker, except for the chosen one (with most datapoints)
    to_drop = copy.copy(lab_labels_endo[key])
    to_drop.remove(column_name_with_max_values)

    data_lab_endo = data_lab_endo.drop(to_drop, axis=1)


"""
For data_lab_surgical
"""
# Do the same for surgical lab data
for item, proposal in product(data_lab_surgical.columns, lab_suffixes):
    if proposal in item and "day" in item:
        lab_labels_surgical[proposal].append(item)

for key in lab_labels_surgical.keys():
    # If only one Record for a blood marker, skip this blood marker
    if len(lab_labels_surgical[key]) < 1:
        continue

    # Else use Record of blood marker with most datapoints
    blood_marker_records = lab_labels_surgical[key]
    column_name_with_max_values = data_lab_surgical[blood_marker_records].count().sort_values(ascending=False).index[0]

    # Drop all other Records of blood marker, except for the chosen one (with most datapoints)
    to_drop = copy.copy(lab_labels_surgical[key])
    to_drop.remove(column_name_with_max_values)

    data_lab_surgical = data_lab_surgical.drop(to_drop, axis=1)

"""
Export data
"""
data_common.to_csv("data/sub_data/data_common.csv")
data_personal.to_csv("data/sub_data/data_personal.csv")
data_imaging.to_csv("data/sub_data/data_imaging.csv")
data_endo.to_csv("data/sub_data/data_endo.csv")
data_lab_endo.to_csv("data/sub_data/data_lab_endo.csv")
data_surgical.to_csv("data/sub_data/data_surgical.csv")
data_lab_surgical.to_csv("data/sub_data/data_lab_surgical.csv")

