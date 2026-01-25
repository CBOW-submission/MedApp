import pandas as pd

dts = pd.read_feather("./disease_to_symptom.feather")
std = pd.read_feather("./symptom_to_disease.feather")
score = pd.read_feather("./scores.feather")
kg = pd.read_feather("./kg.feather")
print(dts.head())
print(dts.info())
print(std.head())
print(std.info())

print(std.query('disease_name=="posterior corneal dystrophy"')["symptom_id"].astype(int).tolist())

print(score.head())

print(kg.query('x_type=="effect/phenotype"&y_type=="disease"&y_name=="posterior corneal dystrophy"').head())
