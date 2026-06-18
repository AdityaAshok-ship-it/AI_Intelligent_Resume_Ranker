#!/usr/bin/env python3
"""
precompute.py — Phase 0: offline, uncapped precompute

Streams candidates.jsonl, parses deterministic features, embeds candidate texts
and the fixed JD with bge-base-en-v1.5, writes four artifacts consumed by rank.py.

Run ONCE (uncapped time). The in-budget ranking step loads no model.

Usage:
    # Full run on the 100K corpus:
    python precompute.py --candidates /path/to/candidates.jsonl

    # Phase 0 exit gate — parse only (no embedding), inspect flags on the 50-sample:
    python precompute.py --candidates /path/to/sample_candidates.json --dry-run --inspect

    # Download bge-base-en-v1.5 weights to local path first (run once with network):
    python precompute.py --download-model

Artifacts written to ./artifacts/:
    candidate_matrix.npy   float32 (N, 768) — L2-normalised embeddings
    candidate_ids.npy      str array,  row i → candidate_id
    features.parquet       one row per candidate, parsed features
    jd_vector.npy          float32 (768,) — JD embedding, same normalisation

Alignment contract (EC-11):
    candidate_matrix[i] ↔ candidate_ids[i] ↔ features row where candidate_id matches
    Asserted at write time here; asserted again at load time in rank.py.

Edge cases handled here — see edgecases.md for the full catalog:
    EC-1  github_activity_score == -1  (sentinel, not a real score)
    EC-2  offer_acceptance_rate == -1  (sentinel, not a real rate)
    EC-3  skill_assessment_scores absent for 76% of pool
    EC-4  career_history[].end_date is null for the current role
    EC-5  18.5% of pool has exactly one career role (H3 slack guards them)
    EC-6  description/industry text is noisy — eyeball flags on 50-sample
    EC-7  skills[] excluded from embedding text
    EC-8  education[] may be empty
    EC-9  notice_period_days quantized; bonus on <=30, not <30
    EC-10 487 MB file — stream line-by-line, never materialise 100K list of dicts
    EC-11 row-alignment asserted at write time
    EC-12 built_real_system uses sentence-level verb+noun guard (SEO false-match filter)
    EC-29–32 H1–H4 slack values exactly as specified
"""

import argparse
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_DATE = date(2026, 5, 27)  # confirmed max last_active_date [DATA, 2026-06-15]
ARTIFACTS_DIR = Path("artifacts")
DEFAULT_MODEL_PATH = Path("models") / "bge-base-en-v1.5"

# ── JD text (fixed; embed once) ───────────────────────────────────────────────
# Distilled from the key technical requirements and ideal-candidate description
# in job_description.docx. Focused on signals the embedding should amplify.

JD_TEXT = (
    "Senior AI Engineer — Recommendation, Ranking, and Retrieval Systems at Redrob AI. "
    "5 to 9 years of experience in applied ML and AI roles at product companies, not consulting. "
    "Production experience with embeddings-based retrieval systems deployed to real users. "
    "Handled embedding drift, index refresh cycles, and retrieval-quality regression in production. "
    "Production experience with vector databases and hybrid search infrastructure: "
    "Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, Annoy, HNSW. "
    "Strong Python. Code quality matters. "
    "Hands-on experience designing evaluation frameworks for ranking systems: "
    "NDCG, MRR, MAP, offline-to-online correlation, A/B test design and interpretation. "
    "Shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale. "
    "Strong opinions about retrieval — hybrid versus dense, sparse versus learned — "
    "evaluation — offline versus online — and LLM integration — when to fine-tune versus prompt. "
    "Built ranking models, recommendation engines, search infrastructure, retrieval pipelines. "
    "Sentence transformers, BGE, E5, OpenAI embeddings, bi-encoders, cross-encoders. "
    "Experience with learning-to-rank: XGBoost-based, neural, LambdaRank. "
    "LLM fine-tuning: LoRA, QLoRA, PEFT. "
    "Located in India preferably Noida, Pune, Hyderabad, Mumbai, Delhi NCR, Bangalore. "
    "Actively available, open to recruiters, low notice period. "
    "Not purely academic or research-only. Not entire career at IT services firms. "
    "Not only recent LangChain or ChatGPT wrapper work without pre-LLM ML production experience. "
    "Not senior who has not written production code in the last 18 months due to management or architecture roles. "
    "Not primarily computer vision, speech recognition, or robotics without NLP or information retrieval experience."
)

# ── Company & title classification sets ───────────────────────────────────────

SERVICES_COMPANIES: frozenset = frozenset({
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree", "hcl", "hcl technologies", "tech mahindra",
    "mphasis", "hexaware", "niit technologies",
})

# Known real Indian tech companies and their founding years.
# Used for founding-date plausibility check in top_150_audit():
# a role at Swiggy starting before 2014 is physically impossible.
# Fictional employers (Hooli, Stark, Dunder Mifflin) cannot be detected this way
# — that residual is bounded by the >10-in-top-100 DQ threshold. (EC-33)
FOUNDING_YEAR_MAP: dict = {
    "swiggy": 2014,
    "razorpay": 2014,
    "paytm": 2010,
    "flipkart": 2007,
    "zomato": 2008,
    "ola": 2010,
    "meesho": 2015,
    "cred": 2018,
    "zepto": 2021,
    "sharechat": 2015,
    "dream11": 2008,
    "groww": 2016,
    "upstox": 2011,
    "smallcase": 2015,
    "dunzo": 2015,
    "udaan": 2016,
    "nykaa": 2012,
}

# Technology inception dates — for the skill-duration anachronism gate (Phase 2b).
# A candidate cannot have used a technology for more months than it has existed.
# Each value is the EARLIEST credible public availability (paper preprint or first
# library/product release), web-verified 2026-06 (sources in commit message).
# Only NAMED artifacts with a hard release date are gated. Generic concepts
# (Embeddings, NLP, Vector Search, Semantic Search, Prompt Engineering,
# "Fine-tuning LLMs", LLMs) are intentionally EXCLUDED — undatable, false-positive
# risk. Tools older than ~2018-05 (FAISS 2017, scikit-learn, Elasticsearch, PyTorch,
# TensorFlow, YOLO, GANs) can NEVER be anachronistic at this corpus's 96-month max
# duration, so they are omitted (their inclusion would be inert).
# ── H5 founding gate: real-company inception dates ────────────────────────────
# The corpus uses a FIXED pool of 63 company names (verified by enumeration):
# 8 fictional + 55 real. H5 flags a resume if ANY role at a REAL company starts
# before that company was founded (role.start_date.year < founding_year). Per the
# user decision, FICTIONAL-company roles are SKIPPED (unknowable founding), but
# real-company roles in the same resume are STILL validated.
#
# Only 5 companies can actually bind (founded AFTER their earliest role-start in the
# data) — these were verified to the YEAR against authentic/first-party sources
# (2026-06-18); the rest post-date every role that references them, so their exact
# year is non-binding and taken from established public record.
FICTIONAL_COMPANIES: frozenset = frozenset({
    "wayne enterprises", "initech", "pied piper", "acme corp",
    "globex inc", "hooli", "dunder mifflin", "stark industries",
})
COMPANY_FOUNDING: dict = {                 # lowercased company name -> founding year
    # — Binding companies (founding > earliest data role-start) — WEB-VERIFIED —
    "cred": 2018,            # Kunal Shah, 2018
    "glance": 2019,          # InMobi Glance, launched 2019 (yourstory.com/2019/04)
    "rephrase.ai": 2019,     # Bengaluru, 2019 (later acq. Adobe Nov 2023)
    "sarvam ai": 2023,       # sarvam.ai/about-us, Aug 2023
    "krutrim": 2023,         # Krutrim Si Designs, 2023 (Ola; launched 15 Dec 2023)
    # — Non-binding (every data role post-dates founding) — established record —
    "infosys": 1981, "wipro": 1945, "tcs": 1968, "capgemini": 1967,
    "accenture": 1989, "hcl": 1976, "mindtree": 1999, "cognizant": 1994,
    "tech mahindra": 1986, "mphasis": 1998,
    "swiggy": 2014, "razorpay": 2014, "zomato": 2008, "flipkart": 2007,
    "meesho": 2015, "nykaa": 2012, "inmobi": 2007, "zoho": 1996, "ola": 2010,
    "vedantu": 2014, "byju's": 2011, "policybazaar": 2008, "paytm": 2010,
    "freshworks": 2010, "upgrad": 2015, "pharmeasy": 2015, "phonepe": 2015,
    "dream11": 2008, "unacademy": 2015,
    "genpact ai": 1997, "aganitha": 2016, "niramai": 2016, "saarthi.ai": 2017,
    "wysa": 2015, "mad street den": 2013, "haptik": 2013, "verloop.io": 2016,
    "observe.ai": 2017, "yellow.ai": 2016, "locobuzz": 2015,
    "google": 1998, "netflix": 1997, "amazon": 1994, "meta": 2004,
    "salesforce": 1999, "microsoft": 1975, "uber": 2009, "adobe": 1982,
    "linkedin": 2003, "apple": 1976,
}

# Every date below is the EARLIEST authentic first-party public artifact, verified
# 2026-06-18 against PRIMARY sources only (arXiv submission pages, the PyPI registry's
# immutable upload timestamps, official repo CHANGELOGs, first-party vendor blogs).
# NO secondary/editable sources (no Wikipedia/Medium/explainer sites). For databases
# the earliest PyPI client release is used as an authentic "existed-by" anchor.
TECH_INCEPTION: dict = {                        # skill name (lowercased) -> (year, month)
    "lora": (2021, 6),                          # arXiv:2106.09685 v1, 17 Jun 2021
    "qlora": (2023, 5),                         # arXiv:2305.14314 v1, 23 May 2023
    "peft": (2023, 1),                          # PyPI peft 0.0.1, 19 Jan 2023
    "rag": (2020, 5),                           # arXiv:2005.11401 v1, 22 May 2020
    "langchain": (2022, 10),                    # PyPI langchain 0.0.1, 25 Oct 2022
    "llamaindex": (2022, 11),                   # PyPI gpt-index 0.0.4, 22 Nov 2022
    "pinecone": (2020, 12),                     # PyPI pinecone-client 0.7.5, 31 Dec 2020
    "milvus": (2019, 6),                        # PyPI pymilvus 0.1.0, 16 Jun 2019
    "qdrant": (2021, 2),                        # PyPI qdrant-client 0.1.0, 9 Feb 2021
    "weaviate": (2019, 11),                     # PyPI weaviate-client 0.0.0, 4 Nov 2019
    "pgvector": (2021, 4),                      # pgvector CHANGELOG 0.1.0 "First release", 20 Apr 2021
    "sentence transformers": (2019, 7),         # PyPI sentence-transformers 0.1.0, 25 Jul 2019
    "haystack": (2019, 11),                     # PyPI farm-haystack 0.1.0.post2, 28 Nov 2019
    "opensearch": (2021, 4),                    # AWS first-party announcement blog, 12 Apr 2021
    "hugging face transformers": (2018, 11),    # PyPI pytorch-pretrained-bert 0.1.1, 17 Nov 2018
}
# Slack: tolerate preprint/beta usage a few months pre-release + duration rounding.
SKILL_ANACHRONISM_SLACK_MONTHS = 6

# Title-chaser / tenure signal — JD "Things we explicitly do NOT want" (listed FIRST):
# "switching companies every 1.5 years ... We need someone who plans to be here for 3+ years."
# Measurable proxy: average tenure per DISTINCT employer. Require >=3 employers so a single
# short stint or a 1-2 role career is never flagged. 18 mo == the JD's literal "1.5 years".
TITLE_CHASE_MIN_COMPANIES = 3
TITLE_CHASE_MAX_AVG_TENURE_MONTHS = 18

ENG_TITLE_KEYWORDS: frozenset = frozenset({
    "engineer", "developer", "scientist", "sde", "swe", "programmer",
    "devops", "mlops", "backend", "frontend", "fullstack", "full-stack",
    "machine learning", "deep learning", "data scientist", "applied scientist",
    "inference", "site reliability", "sre", "platform engineer",
    "recommendation", "search engineer", "ranking", "nlp engineer", "retrieval",
    "research engineer",  # prod-focused research engineering; ≠ pure researcher
})

# Unambiguous non-IC management roles (no code by definition)
NON_IC_TITLE_RE = re.compile(
    r"\b(engineering manager|product manager|program manager|project manager"
    r"|vp\s+of|vice president|chief\s+\w+\s+officer|cto\b|ceo\b"
    r"|director\s+of|head\s+of|scrum master|agile coach)\b",
    re.IGNORECASE,
)

TIER1_CITIES: frozenset = frozenset({
    "noida", "pune", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru",
    "gurgaon", "gurugram", "ncr",
})

# ── Regex patterns ────────────────────────────────────────────────────────────

# Verbs indicating the candidate *shipped* something (for built_real_system, EC-12)
SHIPPED_VERB_RE = re.compile(
    r"\b(built|developed|designed|implemented|shipped|deployed|created"
    r"|architected|launched|scaled|owned|wrote|engineered|led\s+development"
    r"|delivered|released)\b",
    re.IGNORECASE,
)

# System *nouns* in the JD domain — more specific than bare "ranking" or "search"
# Prevents "ranked on the first page of search" SEO text from firing (EC-12)
SYSTEM_NOUN_RE = re.compile(
    # Plurals handled with s? on every count noun (fix: "ranking models" has \b after 'l' in 'models')
    # Hyphens allowed in learning-to-rank (fix: CAND_0000031 uses hyphenated form)
    r"\b(ranking\s+systems?|ranking\s+models?|ranking\s+engines?|ranking\s+pipelines?"
    r"|ranking\s+layer"  # e.g. "Owned the ranking layer" (CAND_0000031 Zomato role)
    r"|search\s+systems?|search\s+engines?|search\s+infrastructure|search\s+platform"
    r"|retrieval\s+systems?|retrieval\s+pipelines?|retrieval\s+engines?"
    r"|recommendation\s+systems?|recommendation\s+engines?|recommender\s+systems?"
    r"|recommender\s+engines?|discovery\s+systems?|matching\s+systems?"
    r"|semantic\s+search|vector\s+search|hybrid\s+search|dense\s+retrieval"
    r"|neural\s+ranking|learn(ing)?[\s-]to[\s-]rank|candidate\s+retrieval"
    r"|feed\s+rank|feed\s+retriev|relevance\s+(systems?|models?|engines?))\b",
    re.IGNORECASE,
)

# Pre-LLM ML production evidence (for D2 exception check)
PRE_LLM_ML_RE = re.compile(
    r"\b(machine\s+learning|deep\s+learning|neural\s+network|xgboost|lightgbm"
    r"|random\s+forest|gradient\s+boost|sklearn|scikit[\s-]learn|pytorch"
    r"|tensorflow|keras|bert|transformer|nlp|natural\s+language"
    r"|recommend(ation|er)|rank(ing)?|retriev(al)?|collaborat(ive)?\s+filter"
    r"|matrix\s+factori|feature\s+engineer|model\s+train|model\s+deploy"
    r"|a[/\s]?b\s+test|experiment|personali(s|z)|(word|doc)2vec|fasttext"
    r"|embedding|vector\s+(search|index|store|db))\b",
    re.IGNORECASE,
)

# Recent LLM-era-only indicators (D2 trigger check)
LANGCHAIN_RE = re.compile(
    r"\b(langchain|langsmith|langgraph|openai\s+api|gpt-[34]|chatgpt|llamaindex"
    r"|llama[\s-]index|autogen|crewai|retrieval[\s-]augmented\s+generation"
    r"|rag\s+pipeline|prompt\s+engineer)\b",
    re.IGNORECASE,
)

# Production deployment evidence (D1 — must have at least one of these)
PRODUCTION_RE = re.compile(
    r"\b(production|prod\b|deployed|shipped|launched|live\s+system|live\s+service"
    r"|product\s+eng|platform\s+eng|serving\s+(infrastructure|layer|system)"
    r"|api\s+(endpoint|service)|microservice|ml\s+platform|inference\s+server)\b",
    re.IGNORECASE,
)

# CV / speech / robotics primary-expertise markers (D5 trigger)
CV_SR_RE = re.compile(
    r"\b(computer\s+vision|image\s+recogni|object\s+detect|image\s+segment"
    r"|depth\s+estim|speech\s+recogni|\basr\b|text[\s-]to[\s-]speech|\btts\b"
    r"|speech\s+synth|robotics|\bros\b|autonomous\s+driv|\bslam\b|\blidar\b"
    r"|point\s+cloud|3d\s+vision|pose\s+estim)\b",
    re.IGNORECASE,
)

# NLP / IR evidence (D5 exception — if present, D5 does NOT fire)
NLP_IR_RE = re.compile(
    r"\b(nlp|natural\s+language|information\s+retrieval|text\s+classif"
    r"|named\s+entity|sentiment|question\s+answer|summari(s|z)|translat"
    r"|language\s+model|bert|transformer|search\s+(system|engine|infra)"
    r"|ranking\s+(system|model)|recommendation|retriev(al)?|embedding"
    r"|vector\s+(search|store|index)|semantic\s+search|lucene|elasticsearch"
    r"|solr|faiss|pinecone|weaviate|qdrant|milvus|annoy|hnsw)\b",
    re.IGNORECASE,
)

# AI-skill keywords for keyword-stuffer count (≥3 on non-eng title → stuffer flag)
AI_SKILL_RE = re.compile(
    r"\b(langchain|llm|large\s+language\s+model|gpt|chatgpt|openai|rag\b"
    r"|retrieval[\s-]augmented|prompt\s+engineer|fine[\s-]tun|lora\b|qlora"
    r"|peft|hugging[\s-]?face|diffusion\s+model|stable\s+diffusion"
    r"|vector\s+database|pinecone|llamaindex|autogen|ai\s+agent"
    r"|agents\s+framework|function\s+calling|tool\s+use)\b",
    re.IGNORECASE,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def months_between(start: date, end: date) -> int:
    """Non-negative integer months (floor)."""
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))


# Degree seniority for the education-integrity gate: Bachelor < Master < Doctorate.
_DEGREE_RANK: dict = {
    "b.tech": 1, "b.e": 1, "b.sc": 1, "bachelor": 1,
    "m.tech": 2, "m.e": 2, "m.s": 2, "m.sc": 2, "mba": 2, "master": 2,
    "ph.d": 3, "phd": 3, "doctor": 3,
}


def _degree_rank(degree: str) -> int:
    """Map a degree string to its seniority (1=Bachelor, 2=Master, 3=PhD; 0=unknown)."""
    dl = (degree or "").lower()
    for key, rank in _DEGREE_RANK.items():
        if key in dl:
            return rank
    return 0


def is_services_role(company: str) -> bool:
    c = company.lower().strip()
    return any(s in c for s in SERVICES_COMPANIES)


def split_sentences(text: str) -> list:
    return re.split(r"(?<=[.!?;])\s+", text)


# ── Embedding text builder ────────────────────────────────────────────────────

def build_embedding_text(cand: dict) -> str:
    """
    Concatenate high-signal fields only.
    skills[] is deliberately excluded — it is the keyword-stuffer attack surface (EC-7).
    Fields: current_title + headline + summary + career_history[].description
    """
    p = cand["profile"]
    parts = [
        p.get("current_title", ""),
        p.get("headline", ""),
        p.get("summary", ""),
    ]
    for role in cand.get("career_history", []):
        d = role.get("description", "")
        if d:
            parts.append(d)
    return " ".join(x for x in parts if x)


# ── Feature parser ────────────────────────────────────────────────────────────

def parse_features(cand: dict) -> dict:
    """
    Deterministic, model-free feature extraction from a single candidate dict.

    All field paths use candidate_schema.json naming.
    Reference now = 2026-05-27 (confirmed max last_active_date).
    See edgecases.md for the complete edge-case catalog.
    """
    cid = cand["candidate_id"]
    p = cand["profile"]
    career: list = cand.get("career_history", [])
    skills: list = cand.get("skills", [])
    edu: list = cand.get("education", [])        # may be empty (EC-8)
    signals: dict = cand.get("redrob_signals", {})

    yoe = float(p.get("years_of_experience", 0))
    yoe_months = yoe * 12

    # ── Career text aggregates ────────────────────────────────────────────────

    current_title = p.get("current_title", "")
    current_title_lo = current_title.lower()

    all_desc = " ".join(r.get("description", "") for r in career)
    sentences = split_sentences(all_desc)

    # ── is_eng_title ─────────────────────────────────────────────────────────

    is_eng_title = any(kw in current_title_lo for kw in ENG_TITLE_KEYWORDS)

    # ── built_real_system (EC-12: sentence-level verb+noun guard) ─────────────
    # A SHIPPED_VERB and a SYSTEM_NOUN must appear in the SAME sentence.
    # This filters "ranked on the first page of search" (SEO) while keeping
    # "built a recommendation system for our discovery feed".
    built_real_system = any(
        SHIPPED_VERB_RE.search(s) and SYSTEM_NOUN_RE.search(s)
        for s in sentences
    )

    # system_type: specific type of production system shipped (for reasoning.py fact-grounding).
    # Extracted from the same sentences that triggered built_real_system.
    # Priority: ranking > recommendation > retrieval > search (matches JD emphasis order).
    system_type = ""
    if built_real_system:
        for s in sentences:
            if SHIPPED_VERB_RE.search(s) and SYSTEM_NOUN_RE.search(s):
                s_lo = s.lower()
                if re.search(r"\branking\b", s_lo):
                    system_type = "ranking"
                elif re.search(r"\b(recommend|discover|feed[\s-]rank)", s_lo):
                    system_type = "recommendation"
                elif re.search(r"\b(retriev|dense|semantic[\s-]search|vector[\s-]search)", s_lo):
                    system_type = "retrieval"
                elif re.search(r"\bsearch\b", s_lo):
                    system_type = "search"
                if system_type:
                    break
        if not system_type:
            system_type = "ranking/retrieval"  # generic fallback when noun is multi-domain

    # ── product_vs_services (EC-21, D4 exception) ────────────────────────────
    # Classify per role. D4 fires only on ENTIRE-career consulting.
    has_product_role = False
    has_services_role = False
    for role in career:
        if is_services_role(role.get("company", "")):
            has_services_role = True
        else:
            has_product_role = True
    all_consulting = has_services_role and not has_product_role
    product_vs_services = has_product_role  # True = has at least one product role

    # ── hands_on_code_18mo (D3 input) ────────────────────────────────────────
    # True if current role title suggests IC engineering (not pure management).
    # Titles don't reliably reveal who writes code — see edgecases.md EC-24.
    # "engineering manager" has both "engineer" and "manager" → non-IC.
    hands_on_code_18mo = False
    for role in career:
        if role.get("is_current", False):
            rt = role.get("title", "")
            rt_lo = rt.lower()
            is_eng = any(kw in rt_lo for kw in ENG_TITLE_KEYWORDS)
            is_pure_mgmt = bool(NON_IC_TITLE_RE.search(rt))
            is_eng_mgr = ("manager" in rt_lo and "engineer" in rt_lo)
            if is_eng and not is_pure_mgmt and not is_eng_mgr:
                hands_on_code_18mo = True
            break

    # ── D1: research-only career, zero production (EC-22) ────────────────────
    all_titles_lo = [r.get("title", "").lower() for r in career]
    has_researcher_title = any(
        # "researcher" or "research X" (but NOT "research engineer" — has prod focus)
        ("researcher" in t or ("research" in t and "engineer" not in t))
        for t in all_titles_lo
    )
    has_production_evidence = bool(PRODUCTION_RE.search(all_desc))
    d1_research_only = has_researcher_title and not has_production_evidence

    # ── D2: AI experience only in recent (<12 mo) LangChain/OpenAI role (EC-23)
    current_role_start = None
    current_role_desc = ""
    prior_descs: list = []
    for role in career:
        if role.get("is_current", False):
            try:
                current_role_start = parse_date(role["start_date"])
            except (KeyError, ValueError):
                pass
            current_role_desc = role.get("description", "")
        else:
            prior_descs.append(role.get("description", ""))

    current_role_months = (
        months_between(current_role_start, REFERENCE_DATE)
        if current_role_start else 999
    )
    current_is_llm_only = (
        bool(LANGCHAIN_RE.search(current_role_desc))
        and not bool(PRE_LLM_ML_RE.search(current_role_desc))
    )
    prior_desc_combined = " ".join(prior_descs)
    has_pre_llm_ml = bool(PRE_LLM_ML_RE.search(prior_desc_combined))

    # D2 fires only if: current role is recent LLM-only AND sub-12 months AND no prior ML
    d2_recent_llm_only = (
        current_is_llm_only and current_role_months < 12 and not has_pre_llm_ml
    )

    # ── D4: entire-career consulting ──────────────────────────────────────────
    d4_all_consulting = all_consulting  # product-company exception applied in rubric.py

    # ── D5: primary CV/speech/robotics, no NLP/IR evidence (EC-26) ───────────
    has_cv_sr = bool(CV_SR_RE.search(all_desc))
    has_nlp_ir = bool(NLP_IR_RE.search(all_desc))
    d5_cv_speech_robotics = has_cv_sr and not has_nlp_ir

    # ── Availability signals ──────────────────────────────────────────────────

    last_active_str = signals.get("last_active_date", "")
    try:
        last_active = parse_date(last_active_str)
        staleness_days = (REFERENCE_DATE - last_active).days
    except (ValueError, TypeError):
        staleness_days = 0

    recruiter_response_rate = float(signals.get("recruiter_response_rate", 0.5))
    open_to_work_flag = bool(signals.get("open_to_work_flag", False))

    # Sentinels — must NOT be treated as real values in any arithmetic (EC-1, EC-2)
    github_activity_score = float(signals.get("github_activity_score", -1))  # -1 = absent
    offer_acceptance_rate = float(signals.get("offer_acceptance_rate", -1))  # -1 = absent

    # skill_assessment_scores: absent (empty dict) for 76% of pool (EC-3)
    # Values are 0-100; normalize to 0-1 for use in rubric.
    skill_assessments: dict = signals.get("skill_assessment_scores") or {}
    has_skill_assessments = len(skill_assessments) > 0
    avg_skill_assessment = (
        sum(skill_assessments.values()) / len(skill_assessments) / 100.0
        if skill_assessments else 0.0
    )

    # ── Logistics ─────────────────────────────────────────────────────────────

    country = p.get("country", "")
    location = p.get("location", "")
    location_lo = location.lower()

    is_india_based = (
        country.lower() in {"india", "in"} or "india" in location_lo
    )
    is_tier1_city = any(city in location_lo for city in TIER1_CITIES)
    willing_to_relocate = bool(signals.get("willing_to_relocate", False))

    # notice_period_days: quantized to {0,15,30,45,60,90,120,150}
    # Bonus keyed on <=30 (not <30 — strict <30 has only 22 records) (EC-9)
    notice_period_days = int(signals.get("notice_period_days", 90))

    preferred_work_mode = signals.get("preferred_work_mode", "flexible")

    # ── Honeypot flags H1–H4 ─────────────────────────────────────────────────

    # H1: a role's duration_months > months since its start_date (+3 mo slack) (EC-29)
    h1_flag = False
    for role in career:
        try:
            start = parse_date(role["start_date"])
        except (KeyError, ValueError):
            continue
        end_str = role.get("end_date")
        if end_str:
            try:
                end = parse_date(end_str)
            except ValueError:
                end = REFERENCE_DATE
        else:
            end = REFERENCE_DATE  # null end_date for current role (EC-4)
        actual_months = months_between(start, end)
        claimed = int(role.get("duration_months", 0))
        if claimed > actual_months + 3:
            h1_flag = True
            break

    # H2: Σ role tenure > YOE×12 + 30 mo (EC-30: +30 slack for concurrent roles)
    total_tenure = sum(int(r.get("duration_months", 0)) for r in career)
    h2_flag = total_tenure > yoe_months + 30

    # H3: YOE×12 > career span + 18 mo (EC-31: +18 slack for early internships)
    # Protects the 18.5% single-role population (EC-5).
    start_dates = []
    for role in career:
        try:
            start_dates.append(parse_date(role["start_date"]))
        except (KeyError, ValueError):
            pass
    career_span_months = (
        months_between(min(start_dates), REFERENCE_DATE) if start_dates else 0
    )
    h3_flag = yoe_months > career_span_months + 18

    # H4: expert/advanced skill with duration_months == 0 (EC-32)
    # Absent duration_months key ≠ zero; only explicit 0 fires.
    h4_flag = False
    for skill in skills:
        dur = skill.get("duration_months")          # may be absent (key missing)
        if dur is None:
            continue
        if dur == 0 and skill.get("proficiency") in {"advanced", "expert"}:
            h4_flag = True
            break

    # H5: a role at a REAL company that starts before that company was founded
    # (role.start_date.year < founding_year). Skip FICTIONAL-company roles (their
    # founding is unknowable) but still validate every real-company role in the same
    # resume — a scammer can't hide an impossible real-company claim behind Hooli.
    # Year-granularity + strict '<' so a company founded in 2018 may hire in 2018.
    # UNKNOWN companies (in neither map) are SKIPPED at ROLE granularity, never
    # flagged — we only flag what we can PROVE impossible — and recorded in
    # h5_unmapped so a future corpus with new companies surfaces them for
    # verification instead of silently eroding the gate (skipping the whole resume
    # would instead let a scammer neutralise H5 by adding one unmapped company).
    h5_flag = False
    _h5_unmapped: list = []
    for role in career:
        comp = (role.get("company", "") or "").strip().lower()
        if not comp or comp in FICTIONAL_COMPANIES:
            continue                                   # skip empty / fictional roles
        founded = COMPANY_FOUNDING.get(comp)
        if founded is None:
            _h5_unmapped.append(role.get("company", "").strip())  # unknown → skip, log
            continue
        try:
            role_start_h5 = parse_date(role["start_date"])
        except (KeyError, ValueError):
            continue
        if role_start_h5.year < founded:
            h5_flag = True
    h5_unmapped = "; ".join(sorted(set(_h5_unmapped)))

    honeypot_flag = h1_flag or h2_flag or h3_flag or h4_flag or h5_flag

    # top_skills_text: top 3 non-AI-buzzword skills by duration_months (for reasoning.py).
    # AI-buzzword skills excluded (keyword-stuffer attack surface; EC-7).
    # Absent duration_months treated as 0 (same handling as H4 check).
    _domain_skills: list = sorted(
        [
            (s.get("name", ""), int(s.get("duration_months") or 0))
            for s in skills
            if s.get("name") and not AI_SKILL_RE.search(s.get("name", ""))
        ],
        key=lambda x: -x[1],
    )
    top_skills_text = ", ".join(n for n, _ in _domain_skills[:3])

    # ── Founding-date plausibility (for top_150_audit; EC-33) ─────────────────
    # Checks known real companies only. Fictional employers (Hooli, Stark, Dunder
    # Mifflin) are undetectable — their residual is bounded by the >10-in-top-100
    # DQ threshold, not this check. Do NOT claim this catches them.
    # Short names (<=5 chars: "ola", "cred") use word-token matching to avoid
    # false-positives on "Motorola", "credentials", etc.
    founding_date_anomaly = False
    for role in career:
        company = role.get("company", "").lower()
        if not company:
            continue
        try:
            role_start = parse_date(role["start_date"])
        except (KeyError, ValueError):
            continue
        company_tokens = set(re.split(r"[^a-z0-9]", company))
        for known_co, founding_year in FOUNDING_YEAR_MAP.items():
            if len(known_co) <= 5:
                matches = known_co in company_tokens   # "ola" ∈ {"ola","cabs"} but not "motorola"
            else:
                matches = known_co in company          # substring fine for long names
            if matches and role_start.year < founding_year:
                founding_date_anomaly = True
                break
        if founding_date_anomaly:
            break

    # ── Skill-duration anachronism (Phase 2b gate) ────────────────────────────
    # Flag if a candidate claims a NAMED technology for more months than it has
    # existed (+slack). Physical impossibility → flawed/fabricated resume. Applied
    # as a HARD pre-sort exclusion in rank.py, after the H1–H4 honeypot gate.
    skill_anachronism_flag = False
    _anachronisms: list = []
    for s in skills:
        nm = (s.get("name", "") or "").strip().lower()
        inc = TECH_INCEPTION.get(nm)
        if inc is None:
            continue
        dur = s.get("duration_months")
        if not isinstance(dur, int):
            continue
        available = months_between(date(inc[0], inc[1], 1), REFERENCE_DATE)
        if dur > available + SKILL_ANACHRONISM_SLACK_MONTHS:
            skill_anachronism_flag = True
            _anachronisms.append(f"{s.get('name')}:{dur}>{available}mo")
    skill_anachronism_detail = "; ".join(_anachronisms)

    # ── Education-timeline integrity (Phase 2b gate) ──────────────────────────
    # Flawed-resume conditions the H1–H4 honeypot rules never inspect:
    #   (a) degree-order reversal — a higher degree ENDS before a lower one STARTS
    #   (b) overlapping full-time degrees at DIFFERENT institutions (>=2y concurrent)
    #   (c) negative span — a degree ends before it starts (degenerate impossibility)
    # Degree NAMES are NOT judged: "B.Tech Artificial Intelligence" is a valid degree.
    education_anomaly_flag = False
    _edu_reasons: list = []
    _edu_spans: list = []
    for e in edu:
        sy, ey = e.get("start_year"), e.get("end_year")
        inst = (e.get("institution", "") or "").strip().lower()
        deg = e.get("degree", "")
        if isinstance(sy, int) and isinstance(ey, int):
            if ey < sy:
                education_anomaly_flag = True
                _edu_reasons.append(f"neg-span {deg} {sy}-{ey}")
            _edu_spans.append((sy, ey, inst, deg))
    # (a) reversal: higher degree finishes before a lower degree begins
    for i, a in enumerate(edu):
        for j, b in enumerate(edu):
            if i == j:
                continue
            ra, rb = _degree_rank(a.get("degree")), _degree_rank(b.get("degree"))
            ay, by = a.get("end_year"), b.get("start_year")
            if ra > rb > 0 and isinstance(ay, int) and isinstance(by, int) and ay < by:
                education_anomaly_flag = True
                _edu_reasons.append(f"reversal {a.get('degree')}<{b.get('degree')}")
    # (b) impossible CONCURRENCY: two degrees enrolled at overlapping times whose
    #     level gap is too large to be legitimate. A PhD requires a COMPLETED
    #     bachelor's, so a Bachelor's and a PhD (degree-rank gap >= 2) cannot
    #     overlap in time. Same-level (gap 0) and adjacent-level (gap 1) overlaps
    #     are ALLOWED — a candidate may legitimately pursue two correspondence/
    #     remote or integrated/dual degrees at once (e.g. distance MBA + M.S.,
    #     5-yr B.Tech+M.Tech, integrated MS-PhD). Institution is irrelevant: remote
    #     degrees are by definition at a different institution. All pairs checked
    #     (<=5 education entries), not just adjacent, so a non-adjacent B.Sc/Ph.D
    #     overlap is not missed.
    _edu_spans.sort()
    for k in range(len(_edu_spans)):
        for m in range(k + 1, len(_edu_spans)):
            s1, e1, _i1, d1 = _edu_spans[k]
            s2, e2, _i2, d2 = _edu_spans[m]
            r1, r2 = _degree_rank(d1), _degree_rank(d2)
            overlap_years = min(e1, e2) - max(s1, s2)
            if r1 and r2 and abs(r1 - r2) >= 2 and overlap_years >= 1:
                education_anomaly_flag = True
                _edu_reasons.append(f"overlap {d1}+{d2} {overlap_years}y(gap{abs(r1 - r2)})")
    education_anomaly_detail = "; ".join(_edu_reasons)

    # ── Title-chaser / tenure (JD "do NOT want") ──────────────────────────────
    # Average tenure per DISTINCT employer. >=3 employers required so a single short
    # stint (or a 1-2 role career) is never flagged. avg==0 (all durations missing) is
    # NOT treated as hopping — that is a data gap, handled by the honeypot rules.
    company_names = {
        (r.get("company", "") or "").strip().lower()
        for r in career if (r.get("company", "") or "").strip()
    }
    num_companies = len(company_names)
    avg_company_tenure_months = (
        total_tenure / num_companies if num_companies else 0.0
    )
    title_chaser_flag = (
        num_companies >= TITLE_CHASE_MIN_COMPANIES
        and 0 < avg_company_tenure_months < TITLE_CHASE_MAX_AVG_TENURE_MONTHS
    )

    # ── Keyword-stuffer inputs ────────────────────────────────────────────────

    ai_skill_count = sum(
        1 for s in skills if AI_SKILL_RE.search(s.get("name", ""))
    )

    # ── Platform signals ──────────────────────────────────────────────────────

    profile_completeness = float(signals.get("profile_completeness_score", 0))
    connection_count = int(signals.get("connection_count", 0))
    interview_completion_rate = float(signals.get("interview_completion_rate", 0))
    applications_30d = int(signals.get("applications_submitted_30d", 0))
    profile_views_30d = int(signals.get("profile_views_received_30d", 0))
    saved_by_recruiters_30d = int(signals.get("saved_by_recruiters_30d", 0))
    avg_response_time_h = float(signals.get("avg_response_time_hours", 0))

    # ── Education ─────────────────────────────────────────────────────────────

    edu_count = len(edu)
    top_edu_tier = "unknown"
    TIER_ORDER = {"tier_1": 0, "tier_2": 1, "tier_3": 2, "tier_4": 3, "unknown": 4}
    for e in edu:
        t = e.get("tier", "unknown")
        if TIER_ORDER.get(t, 4) < TIER_ORDER.get(top_edu_tier, 4):
            top_edu_tier = t

    return {
        # Identity
        "candidate_id": cid,
        # Title / career evidence (65% rubric component)
        "current_title": current_title,
        "is_eng_title": is_eng_title,
        "built_real_system": built_real_system,
        "system_type": system_type,          # "ranking"/"recommendation"/"retrieval"/"search"/""
        "top_skills_text": top_skills_text,  # top 3 non-AI-buzzword skills, comma-sep
        "product_vs_services": product_vs_services,
        "has_product_role": has_product_role,
        "all_consulting": all_consulting,
        "years_of_experience": yoe,
        # Recency / D3 input
        "hands_on_code_18mo": hands_on_code_18mo,
        # Disqualifier gate inputs (D1–D5)
        "d1_research_only": d1_research_only,
        "d2_recent_llm_only": d2_recent_llm_only,
        "d4_all_consulting": d4_all_consulting,
        "d5_cv_speech_robotics": d5_cv_speech_robotics,
        # Availability (ghost conjunction inputs)
        "staleness_days": staleness_days,
        "recruiter_response_rate": recruiter_response_rate,
        "open_to_work_flag": open_to_work_flag,
        "github_activity_score": github_activity_score,    # -1 = no GitHub (EC-1)
        "offer_acceptance_rate": offer_acceptance_rate,     # -1 = no history (EC-2)
        "has_skill_assessments": has_skill_assessments,
        "avg_skill_assessment": avg_skill_assessment,     # 0-1 (0 if absent)
        # Logistics (small modifier)
        "country": country,
        "location": location,
        "is_india_based": is_india_based,
        "is_tier1_city": is_tier1_city,
        "willing_to_relocate": willing_to_relocate,
        "notice_period_days": notice_period_days,
        "preferred_work_mode": preferred_work_mode,
        # Honeypot flags H1–H4 (hard-exclude gate inputs)
        "h1_flag": h1_flag,
        "h2_flag": h2_flag,
        "h3_flag": h3_flag,
        "h4_flag": h4_flag,
        "h5_flag": h5_flag,
        "honeypot_flag": honeypot_flag,
        "founding_date_anomaly": founding_date_anomaly,
        "h5_unmapped": h5_unmapped,           # companies in neither map (future-proofing)
        # Phase 2b discrepancy gates (hard-exclude, applied after honeypots)
        "skill_anachronism_flag": skill_anachronism_flag,
        "skill_anachronism_detail": skill_anachronism_detail,
        "education_anomaly_flag": education_anomaly_flag,
        "education_anomaly_detail": education_anomaly_detail,
        # Keyword-stuffer inputs
        "ai_skill_count": ai_skill_count,
        # Sub-signals for calibration / Phase 0 exit-gate eyeballing
        "career_history_length": len(career),
        "career_span_months": career_span_months,
        "total_tenure_months": total_tenure,
        # Title-chaser / tenure signal (JD "do NOT want")
        "num_companies": num_companies,
        "avg_company_tenure_months": avg_company_tenure_months,
        "title_chaser_flag": title_chaser_flag,
        "yoe_months_computed": yoe_months,
        "current_role_months": current_role_months,
        "has_pre_llm_ml": has_pre_llm_ml,
        "has_nlp_ir": has_nlp_ir,
        "has_cv_sr": has_cv_sr,
        # Education (5% component)
        "edu_count": edu_count,
        "top_edu_tier": top_edu_tier,
        # Platform signals
        "profile_completeness": profile_completeness,
        "connection_count": connection_count,
        "interview_completion_rate": interview_completion_rate,
        "applications_30d": applications_30d,
        "profile_views_30d": profile_views_30d,
        "saved_by_recruiters_30d": saved_by_recruiters_30d,
        "avg_response_time_h": avg_response_time_h,
    }


# ── Loader (handles both .jsonl and .json array) ──────────────────────────────

def stream_candidates(path: Path):
    """
    Yield parsed candidate dicts one at a time.
    Handles candidates.jsonl (one JSON object per line) and
    sample_candidates.json (JSON array). Never builds a full list. (EC-10)
    """
    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix == ".json":
            # Sample file is a JSON array — load it (small; 50 records)
            data = json.load(f)
            yield from (data if isinstance(data, list) else [data])
        else:
            # JSONL — stream line by line (487 MB; never materialise) (EC-10)
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  WARNING line {line_num + 1}: {e}", file=sys.stderr)


# ── Phase 0 exit-gate: eyeball parsed flags on the 50-sample ─────────────────

INSPECT_COLS = [
    "candidate_id", "current_title",
    "is_eng_title", "built_real_system", "product_vs_services",
    "hands_on_code_18mo", "d1_research_only", "d2_recent_llm_only",
    "d4_all_consulting", "d5_cv_speech_robotics",
    "honeypot_flag", "h1_flag", "h2_flag", "h3_flag", "h4_flag", "h5_flag",
    "founding_date_anomaly",
    "ai_skill_count", "staleness_days", "recruiter_response_rate",
    "open_to_work_flag", "notice_period_days",
]

def inspect_sample(path: Path) -> None:
    """
    Dump key flags to stdout as a readable table for the Phase 0 exit gate.
    Eyeball every row: if a flag disagrees with what a human reads from the
    raw title / description, the extraction logic is wrong — fix it here. (EC-6)
    """
    rows = []
    raw_titles = {}
    raw_descs = {}

    for cand in stream_candidates(path):
        cid = cand["candidate_id"]
        feat = parse_features(cand)
        rows.append({k: feat[k] for k in INSPECT_COLS})
        raw_titles[cid] = cand["profile"].get("current_title", "")
        raw_descs[cid] = " | ".join(
            r.get("description", "")[:80] for r in cand.get("career_history", [])
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 200)
    print("\n=== Phase 0 exit gate: flag inspection on sample ===")
    print(df.to_string(index=False))
    print(f"\nHoneypot flags: {df['honeypot_flag'].sum()} / {len(df)}")
    print(f"Eng title: {df['is_eng_title'].sum()}")
    print(f"Built real system: {df['built_real_system'].sum()}")
    print(f"All consulting: {df['d4_all_consulting'].sum()}")
    print(f"D1 research only: {df['d1_research_only'].sum()}")
    print(f"D2 recent LLM only: {df['d2_recent_llm_only'].sum()}")
    print(f"D5 CV/SR: {df['d5_cv_speech_robotics'].sum()}")
    print("\n=== Raw title <-> description cross-check (first 300 chars) ===")
    for cid in df["candidate_id"]:
        title = raw_titles[cid]
        descs = raw_descs[cid]
        row = df[df["candidate_id"] == cid].iloc[0]
        flags = (
            f"eng={row['is_eng_title']} "
            f"built={row['built_real_system']} "
            f"prod={row['product_vs_services']} "
            f"honey={row['honeypot_flag']}"
        )
        print(f"\n{cid} | {title}")
        print(f"  flags : {flags}")
        print(f"  desc  : {descs[:200]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0 precompute: parse features and embed all candidates."
    )
    parser.add_argument(
        "--candidates", default=None,
        help="Path to candidates.jsonl (100K) or sample_candidates.json (50 records)",
    )
    parser.add_argument(
        "--model-path", default=str(DEFAULT_MODEL_PATH),
        help=f"Local path to bge-base-en-v1.5 weights (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Embedding batch size (default: 256)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse features only, skip embedding (fast; for parser testing)",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Print flag-inspection table (Phase 0 exit gate; implies --dry-run)",
    )
    parser.add_argument(
        "--download-model", action="store_true",
        help="Download bge-base-en-v1.5 from HuggingFace Hub to --model-path, then exit",
    )
    args = parser.parse_args()

    # ── Optional: download model weights once ─────────────────────────────────

    if args.download_model:
        from sentence_transformers import SentenceTransformer
        model_path = Path(args.model_path)
        print(f"Downloading bge-base-en-v1.5 -> {model_path} ...")
        m = SentenceTransformer("BAAI/bge-base-en-v1.5")
        m.save(str(model_path))
        print(f"Saved to {model_path}. Set HF_HUB_OFFLINE=1 for all future runs.")
        return

    if args.candidates is None:
        parser.error("--candidates is required unless --download-model is set")

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: {candidates_path} not found", file=sys.stderr)
        sys.exit(1)

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # ── inspect mode (Phase 0 exit gate) ─────────────────────────────────────

    if args.inspect:
        inspect_sample(candidates_path)
        return

    # ── Pass 1: stream + parse + collect ─────────────────────────────────────

    print(f"Pass 1: streaming {candidates_path} ...")
    feature_rows: list = []
    texts: list = []
    ids: list = []

    for i, cand in enumerate(stream_candidates(candidates_path)):
        feat = parse_features(cand)
        feature_rows.append(feat)
        texts.append(build_embedding_text(cand))
        ids.append(feat["candidate_id"])
        if (i + 1) % 10_000 == 0:
            print(f"  {i + 1:,} records parsed ...")

    n = len(ids)
    print(f"Pass 1 done: {n:,} candidates.")

    # ── Write features.parquet ────────────────────────────────────────────────

    features_path = ARTIFACTS_DIR / "features.parquet"
    print(f"Writing {features_path} ...")
    df = pd.DataFrame(feature_rows)
    df.to_parquet(features_path, index=False, engine="pyarrow")
    size_mb = features_path.stat().st_size / 1e6
    print(f"  {features_path}: {size_mb:.1f} MB, {len(df)} rows, {len(df.columns)} cols")

    # ── H5 coverage diagnostic: surface any company in neither map ─────────────
    if "h5_unmapped" in df.columns:
        unmapped = set()
        for s in df["h5_unmapped"]:
            if s:
                unmapped.update(s.split("; "))
        if unmapped:
            print(
                f"  H5 WARNING: {len(unmapped)} company name(s) in NEITHER "
                f"COMPANY_FOUNDING nor FICTIONAL_COMPANIES — H5 skipped them "
                f"(no false flags, but verify their founding and add): {sorted(unmapped)}"
            )
        else:
            print("  H5 company coverage: all companies mapped (0 unmapped). [OK]")

    # ── Write candidate_ids.npy ───────────────────────────────────────────────

    ids_path = ARTIFACTS_DIR / "candidate_ids.npy"
    ids_arr = np.array(ids, dtype=object)   # object dtype preserves CAND_XXXXXXX strings
    np.save(ids_path, ids_arr)
    print(f"Written {ids_path}: {len(ids_arr)} IDs")

    # Early alignment check (EC-11)
    assert list(df["candidate_id"]) == list(ids_arr), (
        "ALIGNMENT FAILURE: features.parquet rows do not match candidate_ids.npy order"
    )
    print("Alignment assertion PASSED (features <-> ids).")

    if args.dry_run:
        print("\nDry run complete. Embedding skipped.")
        print("Next: python precompute.py --candidates sample_candidates.json --inspect")
        return

    # Free Pass-1 allocations before model load to keep peak RSS under ~2 GB.
    # texts is kept — model.encode() needs it. ids_arr length is captured in n.
    import gc
    del feature_rows, df, ids, ids_arr
    gc.collect()
    print(f"Pass-1 structures freed. Beginning model load ...")
    sys.stdout.flush()

    # ── Load model (offline; no Hub call) ────────────────────────────────────

    # Force all available CPU cores for PyTorch and BLAS before loading model.
    n_cores = os.cpu_count() or 4
    os.environ["OMP_NUM_THREADS"] = str(n_cores)
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
    import torch
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(n_cores)
    print(f"  CPU threads: {n_cores} (torch intra+interop, OMP, MKL)")
    sys.stdout.flush()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(
            f"ERROR: model weights not found at {model_path}\n"
            f"Run once with network: python precompute.py --download-model",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading model from {model_path} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(model_path))
    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
    model.max_seq_length = 128
    print(f"  Model loaded. Embedding dim: {dim}, max_seq_length=128")
    assert dim == 768, f"Expected 768-d (bge-base-en-v1.5), got {dim}"

    # ── Embed candidates ──────────────────────────────────────────────────────

    matrix_path = ARTIFACTS_DIR / "candidate_matrix.npy"

    if matrix_path.exists():
        # Checkpoint: skip re-embedding if the matrix was already written.
        print(f"Found existing {matrix_path} — skipping candidate embedding.")
        sys.stdout.flush()
        matrix = None  # not needed for JD embed; freed immediately
    else:
        print(f"Embedding {n:,} candidates (batch_size={args.batch_size}) ...")
        sys.stdout.flush()
        # Encode in one call; sentence-transformers handles batching internally.
        # 100K × 768 × float32 = 307 MB — well within 16 GB cap.
        matrix = model.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,   # L2-norm: cosine = dot product at rank time
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        np.save(matrix_path, matrix)
        size_mb = matrix_path.stat().st_size / 1e6
        print(f"Written {matrix_path}: float32 {matrix.shape}, {size_mb:.0f} MB")
        sys.stdout.flush()

    # Final alignment assertion (EC-11) — only when matrix was freshly computed
    if matrix is not None:
        assert matrix.shape[0] == n, (
            f"ALIGNMENT FAILURE: matrix rows ({matrix.shape[0]}) != ids ({n})"
        )
        print("Alignment assertion PASSED (matrix <-> ids).")
        del matrix   # free RAM before embedding the JD

    # ── Embed JD ──────────────────────────────────────────────────────────────

    jd_path = ARTIFACTS_DIR / "jd_vector.npy"
    print("Embedding JD ...")
    jd_vec = model.encode(
        [JD_TEXT],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype("float32")
    np.save(jd_path, jd_vec)
    norm = float(np.linalg.norm(jd_vec))
    print(f"Written {jd_path}: shape {jd_vec.shape}, norm {norm:.4f} (should be ~1.0)")

    # ── Summary ───────────────────────────────────────────────────────────────

    print("\nPhase 0 artifacts:")
    for artifact in sorted(ARTIFACTS_DIR.iterdir()):
        mb = artifact.stat().st_size / 1e6
        print(f"  {artifact.name}: {mb:.1f} MB")
    print("\nPhase 0 complete.")
    print("Exit gate check:")
    print("  python precompute.py --candidates sample_candidates.json --inspect")


if __name__ == "__main__":
    main()
