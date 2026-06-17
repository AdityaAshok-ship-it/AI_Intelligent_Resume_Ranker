"""
sandbox/app.py — Phase 5: Streamlit demo

Accepts ≤100 candidates (JSON/JSONL), embeds live with bge-base-en-v1.5
(bundled weights, no network), scores and displays the ranked output.

Satisfies Stage 1 sandbox requirement (submission_spec.md §10.5).
Deployed to HuggingFace Spaces as primary; `docker run` recipe in README as backup.

Usage:
    streamlit run sandbox/app.py

Note: This path loads the model; the main rank.py does NOT load the model.
Both paths must be tested before submission (edgecases.md EC-67).
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd

# Add parent to path so we can import from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob Intelligent Candidate Ranker")
st.caption("Demo: upload ≤100 candidates in JSON or JSONL format.")

MODEL_PATH = Path(__file__).parent.parent / "models" / "bge-base-en-v1.5"
HF_MODEL_ID = "BAAI/bge-base-en-v1.5"

@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    if MODEL_PATH.exists():
        # Local run: use bundled weights (offline — no network needed)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        return SentenceTransformer(str(MODEL_PATH))
    # HuggingFace Spaces or first Docker run: download from Hub
    return SentenceTransformer(HF_MODEL_ID)


@st.cache_data
def get_jd_vector(_model):
    from precompute import JD_TEXT
    return _model.encode([JD_TEXT], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")


uploaded = st.file_uploader(
    "Upload candidates.json or candidates.jsonl (≤100 records)",
    type=["json", "jsonl"],
)

if uploaded:
    raw = uploaded.read().decode("utf-8")
    try:
        if uploaded.name.endswith(".json"):
            data = json.loads(raw)
            candidates = data if isinstance(data, list) else [data]
        else:
            candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}")
        st.stop()

    if len(candidates) > 100:
        st.warning(f"Only the first 100 of {len(candidates)} candidates will be scored.")
        candidates = candidates[:100]

    st.write(f"Loaded {len(candidates)} candidates.")

    with st.spinner("Loading model and scoring ..."):
        from precompute import parse_features, build_embedding_text
        from rubric import compute_base, apply_disqualifier_caps, compute_availability_multiplier
        from detectors import get_honeypot_mask

        try:
            model = load_model()
        except Exception as e:
            st.error(f"Failed to load model from {MODEL_PATH}: {e}")
            st.stop()

        jd_vec = get_jd_vector(model)

        rows, texts, ids = [], [], []
        for cand in candidates:
            feat = parse_features(cand)
            rows.append(feat)
            texts.append(build_embedding_text(cand))
            ids.append(feat["candidate_id"])

        features = pd.DataFrame(rows)
        matrix = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        sim = matrix @ jd_vec

        base = compute_base(features, sim)
        gated = apply_disqualifier_caps(base, features)
        honeypot_mask = get_honeypot_mask(features)
        multiplier = compute_availability_multiplier(features)
        final = np.where(~honeypot_mask, gated * multiplier, -np.inf)

        features["score"] = final
        features["honeypot"] = honeypot_mask
        ranked = features.sort_values("score", ascending=False).reset_index(drop=True)
        ranked["rank"] = range(1, len(ranked) + 1)

    display_cols = [
        "rank", "candidate_id", "current_title", "score",
        "is_eng_title", "built_real_system", "product_vs_services",
        "years_of_experience", "staleness_days", "recruiter_response_rate",
        "honeypot", "notice_period_days",
    ]
    st.dataframe(ranked[display_cols].style.format({"score": "{:.4f}"}), use_container_width=True)

    n_hp = honeypot_mask.sum()
    st.info(f"{n_hp} honeypot(s) excluded from ranking.")
