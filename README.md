# MediGraph | Backend & Mathematical Diagnostic Engine 🏥

[![Hackathon](https://img.shields.io/badge/Hackathon-UDBHAV_Inter--IIIT-blue)](https://github.com/CBOW-submission/MedApp)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow)](https://www.python.org/)

**MediGraph** is a hybrid medical diagnostic tool developed for the **UDBHAV Inter-IIIT Hackathon**. Our team represented **IIIT Naya Raipur** as finalists, presenting a system that bridges the gap between unstructured clinical notes and structured medical knowledge.

## 🚀 The Hybrid Approach

MediGraph utilizes two distinct methods to generate a differential diagnosis:
1.  **Deterministic Math Model:** A knowledge-graph-based scoring engine using **PrimeKG**.
2.  **Neural Reasoning:** A fine-tuned **Qwen3-1.7b Instruct** model for clinical interpretation.

> **Note:** This repository focuses on the **Backend, Math Model, and Data Filtering** pipelines. For the LLM fine-tuning details, visit [MediQwen-fine-tuned](https://github.com/joetheguide2/MediQwen-fine-tuned).

---

## 🛠️ How it Works

### 1. Clinical Extraction (MedSpaCy)
We process raw clinical notes using `medspacy`. To ensure high precision, we built **custom entity recognition rules** derived directly from symptom names in the **PrimeKG** (Precision Medicine Knowledge Graph). This allows the system to:
* Identify symptoms mentioned in text.
* Handle negation (e.g., distinguishing "patient has cough" from "no cough").
* Map colloquial mentions to standardized PrimeKG nodes.

### 2. The Mathematical Ranking Engine
Once symptoms are extracted, we query the Knowledge Graph to find all associated diseases. We then rank these diseases using a multi-factor scoring algorithm:

#### A. Reverse Frequency Score ($S_{rf}$)
To account for the "diagnostic weight" of a symptom, we use the reverse frequency. Rarer symptoms provide a higher score than common ones (like fever).
$$S_{rf} = \frac{1}{n}$$
*Where $n$ is the number of diseases associated with that specific symptom in PrimeKG.*

#### B. Coverage Ratios
We calculate two specific ratios to measure the "fit":
* **Patient Coverage ($a$):** Percentage of patient-reported symptoms that match the disease.
* **Disease Coverage ($b$):** Percentage of the disease's total known symptoms that the patient possesses.

#### C. Final Scoring Formula
We combine these using the **Harmonic Mean** of the coverage ratios and add the weighted Reverse Frequency scores to produce the final ranking:

$$\text{Score} = w_1 \cdot \left( \frac{2 \cdot a \cdot b}{a + b} \right) + w_2 \cdot \sum S_{rf}$$

---

## 🏗️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **NLP Pipeline** | MedSpaCy, SpaCy |
| **Knowledge Graph** | PrimeKG |
| **Data Processing** | Pandas, NumPy |
| **LLM** | Qwen3-1.7b (Fine-tuned) |

---

## 📂 Repository Structure

* `model/` : Implementation of the mathematical scoring and ranking algorithm.
* `backend/` : FastAPI based backend.



Collaborators: [fine2006](github.com/fine2006) , [abhay1074](github.com/abhay1074), [joetheguide2](github.com/joetheguide2) 
Github of PrimeKG: [PrimeKG](https://github.com/mims-harvard/PrimeKG)
