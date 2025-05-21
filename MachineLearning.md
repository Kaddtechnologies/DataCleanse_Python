# Machine Learning Implementation Strategy

## Problem Analysis
Provide a detailed analysis of the specific problem we're addressing, including data characteristics, constraints, and success metrics.

## Recommended ML Approach
Outline the most suitable machine learning paradigm (supervised/unsupervised/reinforcement learning) with justification based on problem requirements. Include specific algorithms and models recommended, with comparative analysis of alternatives.

## Technical Architecture
Detail the complete tech stack:
- Data processing framework (Spark/Pandas/Dask)
- ML libraries and frameworks (TensorFlow/PyTorch/scikit-learn)
- Model serving infrastructure (TensorFlow Serving/Seldon/KFServing)
- Cloud platform recommendations (AWS/GCP/Azure) with specific services
- Data storage solutions (SQL/NoSQL/data lakes)

## Implementation Roadmap
1. Data acquisition and preparation strategy
2. Feature engineering approach
3. Model development process
4. Evaluation methodology
5. Deployment architecture
6. Monitoring and maintenance plan

## Resource Requirements
Estimate computational resources, team expertise, and timeline for implementation phases.

## Future Scalability
Address how the solution can evolve for Phase 2, including potential enhancements and scaling considerations.

# Phase 2 Deduplication – Three Upgrade Paths
(ordered from least to most expensive in overall cost / resources / time)

#	Option	Up-front cost & effort	Ongoing cost	Typical time-to-value	What it adds
A	Interactive Human-in-the-Loop + Active Learning	Low – small Streamlit/Flask UI plus a review table	Reviewers’ time	Days → 1-2 weeks	Human judgment on low-confidence cases, labelled dataset for future ML
B	External Reference Data Standardisation	Medium – integrate address & company registries; extra ETL step	API fees (Geocoding, postal, company db)	2-4 weeks	Deterministic, explainable canonical forms (e.g., “IBM Corp.” ↔ “International Business Machines Corporation”)
C	Probabilistic “Learning-to-Match” Classifier	High – build feature store, train LightGBM model, deploy scorer	Occasional re-training compute	4-8 weeks (after enough labels)	Data-driven scores, optimal thresholds, future-proof for new fields

Option A – Human-in-the-Loop & Active Learning (cheapest / quickest)
Idea
Flag any pair/group with overall_score below your auto-merge threshold (e.g., < 0.90) as Needs Review. A lightweight UI lets reviewers confirm or reject. Decisions are logged to a labelled_pairs table that you can later feed into Option C.

<details> <summary>🔎 Minimal Streamlit review UI (add to <code>app_review.py</code>)</summary>
python
Copy
Edit
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://…")

st.title("🕵️‍♂️ Deduplication Review Queue")

df = pd.read_sql("SELECT * FROM potential_duplicates WHERE reviewed IS FALSE", engine)

def mark(label):
    sel = st.session_state.selected
    if sel:
        ids = tuple(sel)
        engine.execute(
            "UPDATE potential_duplicates SET reviewed=TRUE, label=%s WHERE id IN %s",
            (label, ids)
        )
        st.experimental_rerun()

st.dataframe(df, selection_mode="multi", key="selected")
col1, col2, col3 = st.columns(3)
col1.button("✅ Confirm Duplicate", on_click=mark, args=("duplicate",))
col2.button("❌ Not Duplicate", on_click=mark, args=("unique",))
col3.button("🤷 Needs More Info", on_click=mark, args=("unknown",))
</details>
Pros

Uses existing logic; no major refactor.

Labels you gather become training data for Option C.

Review UI is transparent and auditable.

Cons

Throughput limited by reviewer bandwidth.

Accuracy only increases as fast as you review.

Option B – External Reference Data Canonicalisation (moderate cost)
Idea
Normalise names & addresses against authoritative registries before blocking, so string variation disappears.

python
Copy
Edit
# 🔖 new dependency block
pip install usaddress pypostalcode googlemaps python-company-name

# ➕ add to app.py
import usaddress, googlemaps, company_name

gmaps = googlemaps.Client(key=os.getenv("GOOGLE_API_KEY"))

def standardise_address(raw: str) -> str:
    try:
        parsed = usaddress.tag(raw)[0]
        full = f"{parsed.get('AddressNumber','')} {parsed.get('StreetName','')} {parsed.get('StreetNamePostType','')}, " \
               f"{parsed.get('PlaceName','')}, {parsed.get('StateName','')} {parsed.get('ZipCode','')}"
        # hit geocoder for canonical form
        geo = gmaps.geocode(full)
        return geo[0]["formatted_address"] if geo else full
    except Exception:
        return raw.lower().strip()

def standardise_company(raw: str) -> str:
    cleaned = company_name.clean(raw)  # strips Inc., Corp., GmbH …
    return cleaned.lower()

def apply_reference_standardisation(df, cm):
    if cm.address in df:
        df["address_std"] = df[cm.address].fillna("").map(standardise_address)
    if cm.customer_name in df:
        df["name_std"] = df[cm.customer_name].fillna("").map(standardise_company)
    return df
Pros

Big precision boost on spelling / abbreviation edge-cases.

Deterministic = easy to explain to auditors.

Integrates cleanly with existing blocking (“use address_std instead of address”).

Cons

Requires API keys; volume-based fees.

Adds latency per record (batch nightly ETL recommended).

Global coverage varies by provider.

Option C – Probabilistic Learning-to-Match Classifier (highest investment, best long-term)
Idea
Replace hand-tuned overall_score with a model that learns from many per-field similarity features.

python
Copy
Edit
# 1️⃣  Build training data --------------------------------------
pairs = pd.read_sql("SELECT * FROM labelled_pairs", engine)

feature_cols = [c for c in pairs.columns if c.startswith("sim_")] + ["block_type_flag"]
X, y = pairs[feature_cols], pairs["label"].map({"duplicate":1, "unique":0})

# 2️⃣  Train model ---------------------------------------------
from lightgbm import LGBMClassifier
model = LGBMClassifier(
    n_estimators=600, learning_rate=0.05, max_depth=-1,
    class_weight="balanced", random_state=42
)
model.fit(X, y)
joblib.dump(model, "dedupe_model.pkl")

# 3️⃣  Use in pipeline -----------------------------------------
model = joblib.load("dedupe_model.pkl")
proba = model.predict_proba(X_pair)[0,1]          # 0-1 duplicate probability
Tech-stack notes

Layer	Recommended libs	GPU needed?
Feature engineering	pandas / numpy / thefuzz, jellyfish (already in place)	No
Model	LightGBM (lightgbm), or fallback to sklearn.ensemble.GradientBoostingClassifier	No (CPU-optimised)
Serving	joblib-loaded model inside FastAPI route	No

TensorFlow / PyTorch are overkill unless you later move to deep Siamese encoders; start with LightGBM.

Pros

Learns optimal weights & non-linear interactions → highest F-score uplift (5-15 pp).

Two threshold knobs give auto-merge vs. human-review bands.

Feature importance keeps model explainable.

Cons

Needs a few hundred labelled pairs (Option A supplies them).

ML lifecycle: retraining, versioning, monitoring.

Longer initial build time.

Recommendation Snapshot
Scenario	Best pick
Need rapid boost, minimal budget	Option A – ship review UI next sprint and start labelling
Have budget for APIs and want deterministic output	Option B – pipe data through canonical registries
Long-term accuracy & scalability drive value most	Option C – invest in LightGBM classifier (my top strategic choice)
