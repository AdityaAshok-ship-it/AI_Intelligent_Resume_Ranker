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


upload_col, dl_col = st.columns([3, 1])
with upload_col:
    uploaded = st.file_uploader(
        "Upload candidates.json or candidates.jsonl (≤100 records)",
        type=["json", "jsonl"],
    )
with dl_col:
    dl_placeholder = st.empty()

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
        from detectors import get_honeypot_mask, get_anachronism_mask, get_education_mask

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

        base             = compute_base(features, sim)
        gated            = apply_disqualifier_caps(base, features)
        honeypot_mask    = get_honeypot_mask(features)
        anachronism_mask = get_anachronism_mask(features)
        education_mask   = get_education_mask(features)
        surviving_mask   = ~honeypot_mask & ~anachronism_mask & ~education_mask
        multiplier       = compute_availability_multiplier(features)
        final            = np.where(surviving_mask, gated * multiplier, -np.inf)

        features["score"]    = final
        features["honeypot"] = honeypot_mask
        features["excluded"] = ~surviving_mask
        # Exclude all gate failures before ranking — mirrors rank.py pipeline.
        ranked = (
            features[surviving_mask]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        ranked["rank"] = range(1, len(ranked) + 1)

    n_hp    = int(honeypot_mask.sum())
    n_anach = int(anachronism_mask.sum())
    n_edu   = int(education_mask.sum())
    n_excl  = int((~surviving_mask).sum())

    # ── YOE: clean display (6.0 → 6, 6.5 → 6.5) ─────────────────────────────
    def fmt_yoe(v):
        f = float(v)
        return str(int(f)) if f == int(f) else str(round(f, 1))

    ranked["yoe"] = ranked["years_of_experience"].apply(fmt_yoe)

    # ── Concerns: explicit flag column ────────────────────────────────────────
    def build_concerns(row):
        notes = []
        if not bool(row.get("hands_on_code_18mo", True)):
            notes.append("no IC/hands-on 18mo")
        if bool(row.get("d4_all_consulting", False)):
            notes.append("all consulting")
        if bool(row.get("d2_recent_llm_only", False)):
            notes.append("recent LLM-only")
        if bool(row.get("d1_research_only", False)):
            notes.append("research-only")
        if bool(row.get("d5_cv_speech_robotics", False)):
            notes.append("CV/speech primary")
        if bool(row.get("title_chaser_flag", False)):
            notes.append("job-hopper")
        if not bool(row.get("is_india_based", True)):
            notes.append("international")
        if int(row.get("notice_period_days", 30)) >= 120:
            notes.append("notice 120d+")
        elif int(row.get("notice_period_days", 30)) >= 90:
            notes.append("notice 90d")
        if str(row.get("preferred_work_mode", "")) == "remote":
            notes.append("remote-only")
        if int(row.get("staleness_days", 0)) > 120 and float(row.get("recruiter_response_rate", 1)) < 0.15:
            notes.append("ghost risk")
        if not bool(row.get("built_real_system", True)):
            notes.append("no ship evidence")
        return "; ".join(notes) if notes else "—"

    ranked["concerns"] = ranked.apply(build_concerns, axis=1)

    display_cols = [
        "rank", "candidate_id", "current_title", "score",
        "yoe", "is_eng_title", "built_real_system",
        "staleness_days", "recruiter_response_rate",
        "notice_period_days", "concerns",
    ]
    csv_cols  = ["rank", "candidate_id", "current_title", "yoe", "score",
                 "is_eng_title", "built_real_system", "staleness_days",
                 "recruiter_response_rate", "notice_period_days", "concerns"]
    csv_bytes = ranked[[c for c in csv_cols if c in ranked.columns]].to_csv(index=False).encode("utf-8")

    # Download button — top-right, aligned with the file uploader
    dl_placeholder.download_button(
        label="⬇ Download CSV",
        data=csv_bytes,
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )

    # Pre-format columns as strings so column_config can control widths freely
    display_df = ranked[display_cols].copy()
    display_df["score"] = display_df["score"].apply(lambda v: f"{float(v):.4f}")
    display_df["recruiter_response_rate"] = display_df["recruiter_response_rate"].apply(
        lambda v: f"{float(v):.0%}"
    )

    st.dataframe(
        display_df,
        column_config={
            "concerns": st.column_config.TextColumn("Concerns", width="large"),
            "current_title": st.column_config.TextColumn("Title", width="medium"),
            "candidate_id": st.column_config.TextColumn("Candidate ID", width="medium"),
            "recruiter_response_rate": st.column_config.TextColumn("RRR"),
        },
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        f"Excluded {n_excl} candidates — "
        f"honeypots: {n_hp} | skill anachronism: {n_anach} | education integrity: {n_edu}. "
        f"Showing {len(ranked)} ranked candidates."
    )
