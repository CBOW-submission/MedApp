from typing import Optional
import medspacy
from medspacy.target_matcher import TargetRule
from spacy.tokens import Span
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from diag_model import run_inference_on_sample, load_final_model

model, tokenizer, device = load_final_model()


# 1. Setup Data
odf = pd.read_feather("official.feather")
print(odf.info())
kg = pd.read_feather("kg.feather")
std = pd.read_feather("./symptom_to_disease.feather")
scores = pd.read_feather("./norm_scores.feather")
# Ensure ID is consistent (using int for matching logic)
odf["symptom_id"] = odf["symptom_id"].astype(int)
id_to_name = dict(zip(odf["symptom_id"], odf["symptom_name"]))

# 2. Initialize MedSpaCy with Context and Sectionizer
# Adding 'medspacy_context' and 'medspacy_sectionizer' to the pipeline
nlp = medspacy.load(
    enable=["medspacy_target_matcher", "medspacy_context", "medspacy_sectionizer"]
)

if not Span.has_extension("hpo_id"):
    Span.set_extension("hpo_id", default=None)

target_matcher = nlp.get_pipe("medspacy_target_matcher")

# 3. Build Rules from Official DataFrame
target_rules = [
    TargetRule(
        literal=str(row["symptom_name"]),
        category="SYMPTOM",
        attributes={"hpo_id": str(row["symptom_id"])},
    )
    for _, row in odf.iterrows()
]
target_matcher.add(target_rules)


app = FastAPI()

<<<<<<< Updated upstream
=======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],  # React dev server ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
>>>>>>> Stashed changes

@app.get("/")
def root():
    return "Hello World"


class MedicalNote(BaseModel):
    text: str


class RawText(BaseModel):
    text: str


class Symptom(BaseModel):
    HPO_ID: int
    Phenotype: str
    start: Optional[int]
    end: Optional[int]
    status: str = "Positive"


class Disease(BaseModel):
    # MONDO_ID : int
    name: str
    f1: float
    confidence_score: float


@app.post("/extract", response_model=list[Symptom])
def extract_symptoms(rawnote: RawText) -> list[Symptom]:
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

        new = Symptom(
            HPO_ID=hpo_id,
            Phenotype=official_name,
            start=ent.start_char,
            end=ent.end_char,
            status=status,
        )
        ans.append(new)

        print(
            f"{ent.text[:25]:<25} | {hpo_id:<8} | {status:<12} | {section:<15} | {official_name}"
        )

    return ans


@app.get("/search", response_model=list[Symptom])
def search(search: str) -> list[Symptom]:
    mask = (
        odf['symptom_name'].astype(str).str.contains(search, case=False, na=False) | 
        odf['symptom_id'].astype(str).str.contains(search, case=False, na=False)
    )
    
    # Convert filtered rows to list of dicts
    records = odf[mask].to_dict(orient="records")
    
    return [
        Symptom(
            HPO_ID=r["symptom_id"],
            Phenotype=r["symptom_name"],
            start=None,
            end=None,
            status="Positive"
            ) for r in records[:5]
    ]


@app.post("/aianalysis", response_model=RawText)
def ai_analysis(rawnote: RawText) -> RawText:
    return RawText(text=run_inference_on_sample(model, tokenizer, rawnote.text, device))


@app.post("/ddx", response_model=list[Disease])
def ddx(symptoms: list[Symptom]) -> list[Disease]:
    diseases = []
    dname = set()
    stoms = set(x.HPO_ID for x in symptoms)

    # Get all diseases that have any of patient's symptoms
    for symptom in symptoms:
        rds = set(std.query(f'symptom_id=="{symptom.HPO_ID}"')["disease_name"].tolist())
        dname = dname.union(rds)

    for name in dname:
        symptoms_in_disease = set(
            std.query(f'disease_name=="{name}"')["symptom_id"].astype(int).tolist()
        )
        matches = symptoms_in_disease.intersection(stoms)

        # Skip if no matches at all
        if len(matches) == 0:
            continue

        # F1 calculation with safety
        p = len(matches) / len(stoms)
        r = len(matches) / (
            len(symptoms_in_disease) if len(symptoms_in_disease) <= 10 else 10
        )

        if p + r == 0:
            f1 = 0
        else:
            f1 = (2 * p * r) / (p + r)

        # Confidence: use weights of matching symptoms
        conf_weights = []
        for match in matches:
            # Get the weight for this symptom
            weight = scores.loc[scores["symptom_id"] == match, "score"]
            if not weight.empty:
                conf_weights.append(weight.iloc[0])

        if conf_weights:
            confidence_score = 0.8 * max(conf_weights) + 0.2 * (
                sum(conf_weights) / len(conf_weights)
            )
        else:
            confidence_score = 0

        diseases.append(Disease(name=name, f1=f1, confidence_score=confidence_score))

    # Sort by combined score
    diseases.sort(key=lambda x: (0.5 * x.f1 + 0.5 * x.confidence_score), reverse=True)
    return diseases[:5]  # Return top 5
