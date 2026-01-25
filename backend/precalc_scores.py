import pandas as pd
import numpy as np
from collections import defaultdict

# 1. Load data
print("Loading data...")
odf = pd.read_feather("official.feather")
kg = pd.read_feather("kg.feather")

# Ensure ID is consistent
odf['symptom_id'] = odf['symptom_id'].astype(int)

# 2. Filter knowledge graph for phenotype/effect ↔ disease relationships
print("Filtering knowledge graph for symptom-disease relationships...")
# Note: In PrimeKG, symptoms are represented as 'phenotype' or 'effect' nodes
mask = (
    ((kg['x_type'] == "effect/phenotype") & (kg['y_type'] == "disease")) |
    ((kg['x_type'] == 'disease') & (kg['y_type'] == "effect/phenotype"))
)

filtered_kg = kg[mask].copy()
print(f"Found {len(filtered_kg)} symptom-disease relationships")

# 3. Create symptom-to-disease and disease-to-symptom mappings
print("Creating symptom-to-disease mapping...")
symptom_to_disease = []

for _, row in filtered_kg.iterrows():
    if row['x_type'] == "effect/phenotype":
        symptom_id = str(row['x_id'])
        symptom_name = row['x_name']
        disease_name = row['y_name']
    else:  # x_type is 'disease'
        symptom_id = str(row['y_id'])
        symptom_name = row['y_name']
        disease_name = row['x_name']

    symptom_to_disease.append({
        'symptom_id': symptom_id,
        'symptom_name': symptom_name,
        'disease_name': disease_name
    })

symptom_to_disease_df = pd.DataFrame(symptom_to_disease).drop_duplicates()
print(f"Created symptom-to-disease DataFrame with {len(symptom_to_disease_df)} unique relationships")

# Save symptom-to-disease mapping
symptom_to_disease_df.to_feather("symptom_to_disease.feather")
print("Saved symptom_to_disease.feather")

# 4. Create disease-to-symptom mapping
print("Creating disease-to-symptom mapping...")
disease_to_symptom = []

for _, row in filtered_kg.iterrows():
    if row['x_type'] == "effect/phenotype":
        disease_name = row['y_name']
        symptom_id = str(row['x_id'])
        symptom_name = row['x_name']
    else:  # x_type is 'disease'
        disease_name = row['x_name']
        symptom_id = str(row['y_id'])
        symptom_name = row['y_name']

    disease_to_symptom.append({
        'disease_name': disease_name,
        'symptom_id': symptom_id,
        'symptom_name': symptom_name
    })

disease_to_symptom_df = pd.DataFrame(disease_to_symptom).drop_duplicates()
print(f"Created disease-to-symptom DataFrame with {len(disease_to_symptom_df)} unique relationships")

# Save disease-to-symptom mapping
disease_to_symptom_df.to_feather("disease_to_symptom.feather")
print("Saved disease_to_symptom.feather")

# 5. Compute scores for each symptom in the official dataset
print("Computing scores for each symptom...")

# Create a dictionary to store disease counts for each symptom
symptom_disease_counts = defaultdict(set)

# Populate the dictionary using the symptom_to_disease DataFrame
for _, row in symptom_to_disease_df.iterrows():
    symptom_id = row['symptom_id']
    disease_name = row['disease_name']
    symptom_disease_counts[symptom_id].add(disease_name)

# Compute scores for all symptoms in the official dataset
scores = {}
all_symptom_ids = set(odf['symptom_id'].astype(str))
found_symptoms = set()

for s_id in all_symptom_ids:
    if s_id in symptom_disease_counts:
        num_diseases = len(symptom_disease_counts[s_id])
        if num_diseases > 0:
            scores[int(s_id)] = 1.0 / num_diseases
            found_symptoms.add(s_id)
        else:
            scores[int(s_id)] = 0.0
    else:
        scores[int(s_id)] = 0.0

print(f"Computed scores for {len(found_symptoms)}/{len(all_symptom_ids)} symptoms with disease associations")

# 6. Create and save the scores DataFrame
print("Creating scores DataFrame...")
df_scores = pd.DataFrame(list(scores.items()), columns=['symptom_id', 'score'])

# Merge with symptom names for reference
df_scores = df_scores.merge(
    odf[['symptom_id', 'symptom_name']].drop_duplicates(),
    on='symptom_id',
    how='left'
)

# Add disease count for each symptom (for reference)
def get_disease_count(s_id):
    s_id_str = str(s_id)
    if s_id_str in symptom_disease_counts:
        return len(symptom_disease_counts[s_id_str])
    return 0

df_scores['disease_count'] = df_scores['symptom_id'].apply(get_disease_count)

# Save scores
df_scores.to_feather("scores.feather")
print("Saved scores.feather")

# 7. Summary statistics
print("\n=== SUMMARY ===")
print(f"Total symptoms in official dataset: {len(odf['symptom_id'].unique())}")
print(f"Symptoms with disease associations: {len(found_symptoms)}")
print(f"Symptoms without disease associations: {len(all_symptom_ids) - len(found_symptoms)}")
print(f"Average score: {df_scores['score'].mean():.4f}")
print(f"Median score: {df_scores['score'].median():.4f}")
print(f"Min score: {df_scores['score'].min():.4f}")
print(f"Max score: {df_scores['score'].max():.4f}")
print(df_scores)
# Display top 10 symptoms by score
print("\nTop 10 symptoms by score (rare symptoms):")
top_scores = df_scores.nlargest(10, 'score')[['symptom_name', 'disease_count', 'score']]
print(top_scores.to_string(index=False))

print("\nBottom 10 symptoms by score (common symptoms):")
bottom_scores = df_scores.nsmallest(10, 'score')[['symptom_name', 'disease_count', 'score']]
print(bottom_scores[bottom_scores['score'] > 0].to_string(index=False))
