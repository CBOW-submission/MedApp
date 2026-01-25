import medspacy
from medspacy.target_matcher import TargetRule
from spacy.tokens import Span
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel



from diag_model import run_inference_on_sample,load_final_model
model,tokenizer,device = load_final_model()




# 1. Setup Data
odf = pd.read_feather("official.feather")

kg = pd.read_feather("kg.feather")
# Ensure ID is consistent (using int for matching logic)
odf['symptom_id'] = odf['symptom_id'].astype(int)
id_to_name = dict(zip(odf['symptom_id'], odf['symptom_name']))

# 2. Initialize MedSpaCy with Context and Sectionizer
# Adding 'medspacy_context' and 'medspacy_sectionizer' to the pipeline
nlp = medspacy.load(enable=["medspacy_target_matcher", "medspacy_context", "medspacy_sectionizer"])

if not Span.has_extension("hpo_id"):
    Span.set_extension("hpo_id", default=None)

target_matcher = nlp.get_pipe("medspacy_target_matcher")

# 3. Build Rules from Official DataFrame
target_rules = [
    TargetRule(literal=str(row['symptom_name']), category="SYMPTOM", attributes={"hpo_id": str(row['symptom_id'])})
    for _, row in odf.iterrows()
]
target_matcher.add(target_rules)





app = FastAPI()

@app.get("/")
def root():
    return "Hello World"

class MedicalNote(BaseModel):
    text : str

class RawText(BaseModel):
    text : str

class Symptom(BaseModel):
    HPO_ID : int
    Phenotype : str
    start: int
    end : int
    status: str


@app.post("/extract", response_model = list[Symptom])
def extract_symptoms(rawnote: RawText)-> list[Symptom]:

    doc = nlp(rawnote.text)

    header = f"{'Entity':<25} | {'HPO ID':<8} | {'Status':<12} | {'Section':<15} | {'Official Name'}"
    print(header)
    print("-" * len(header))

    found_ids = set()
    ans = []

    for ent in doc.ents:
        hpo_id_raw = ent._.hpo_id
        if hpo_id_raw is None:
            continue

        hpo_id = int(hpo_id_raw)

        # Deduplicate based on ID in this specific output run
        if hpo_id in found_ids:
            continue
        found_ids.add(hpo_id)

        # Get Context Attributes
        # is_negated: True if the text says "no [symptom]"
        # is_historical: True if it's in the past
        # is_hypothetical: True if it's a "risk of"
        # is_family: True if it belongs to a relative
        status = "Positive"
        if ent._.is_negated:
            status = "Negated"
        elif ent._.is_family:
            status = "Family Hist"
        elif ent._.is_historical:
            status = "Historical"

        # Get Section Info
        section = f"{ent.start_char}, {ent.end_char}"
        official_name = id_to_name.get(hpo_id, "Unknown ID")

        new = Symptom(HPO_ID = hpo_id, Phenotype=official_name, start=ent.start_char, end = ent.end_char, status = status)
        ans.append(new)

        print(f"{ent.text[:25]:<25} | {hpo_id:<8} | {status:<12} | {section:<15} | {official_name}")

    return ans



@app.post("/aianalysis", response_model = RawText)
def ai_analysis(rawnote: RawText)->RawText:
    return RawText(text=run_inference_on_sample(model,tokenizer,rawnote.text,device))
