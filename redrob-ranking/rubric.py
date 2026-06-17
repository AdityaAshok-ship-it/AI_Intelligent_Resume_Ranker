"""
rubric.py — Phase 1: relevance rubric (base scorer + disqualifier caps + availability multiplier)

Phase 1 deliverable. Composition order (immutable — EC-45, EC-46, decision.md [P1.3]):
    base  = 0.65·career_title + 0.20·embedding + 0.10·skills + 0.05·education + logistics_modifier
    gated = apply_disqualifier_caps(base)          # caps on base BEFORE multiplier
    # honeypot rows dropped before this call (Phase 2 / rank.py)
    final = gated × availability_multiplier        # over honeypot-excluded rows

Caps before multiplier (EC-45):
    multiplier ≤ 1, so a disqualified candidate cannot be rescued by high availability.

Honeypot exclusion position (EC-46):
    Rows are DROPPED (not score-adjusted) before the multiplier step in rank.py.
    "A honeypot never enters scoring" — cleaner to defend than post-sort removal.

Weights (calibrated against 50-sample, Phase 1 exit gate):
    Career: 0.65 — dominates; keyword-stuffer defence; most separable score
    Embed:  0.20 — plain-language tier-5 rescue (precomputed cosine similarity)
    Skills: 0.10 — verified assessments + github; deliberately low (keyword stuffing risk)
    Edu:    0.05 — institution tier; weak signal only
    Logistics: additive modifier (must never dominate career evidence)

See decision.md for every weight decision and calibration log.
"""

import numpy as np
import pandas as pd

# ── Component weights ─────────────────────────────────────────────────────────
W_CAREER = 0.65
W_EMBED  = 0.20
W_SKILLS = 0.10
W_EDU    = 0.05

# ── Disqualifier cap severities (most-restrictive wins; EC-27) ────────────────
D1_CAP = 0.02   # hard floor — "we will not move forward" (JD verbatim)
D2_CAP = 0.40   # heavy — recent LLM-only with no pre-LLM ML history
D3_CAP = 0.45   # heavy — no hands-on IC code in 18 mo (title-inferred; EC-24)
D4_CAP = 0.35   # cap   — entire-career consulting, no product-co exception
D5_CAP = 0.50   # strong — CV/speech/robotics primary with no NLP/IR evidence

# ── Logistics modifiers (each small; must not dominate career signal) ─────────
NOTICE_BONUS        =  0.03   # notice_period_days <= 30 (EC-9: <=30 has 13.8%; <30 has 22 people)
NOTICE_LONG_PENALTY = -0.03   # notice_period_days >= 120 — honest concern, not disqualifier
INDIA_BONUS         =  0.02   # India-based candidate (75.1% of pool [DATA])
TIER1_BONUS         =  0.01   # Tier-1 city: Noida/Pune/Hyderabad/Mumbai/Delhi/Bangalore
RELOCATE_BONUS      =  0.01   # willing_to_relocate AND not already India-based
REMOTE_ONLY_PENALTY = -0.01   # remote-only preference vs Pune/Noida hybrid offices

# ── Availability multiplier parameters (EC-35, EC-38, EC-39) ─────────────────
# Ghost floor fires on CONJUNCTION of all three (not any single signal)
GHOST_FLOOR       = 0.15
HEALTHY_BAND_LOW  = 0.70
# Thresholds calibrated to target the 3.4% ghost population [DATA]:
#   staleness > 120 AND rrr < 0.15 AND not open → 3,372/100K = 3.37%
STALE_THRESHOLD   = 120     # days (not 180 — too strict, yielded 0.8% not 3.4%)
RRR_FLOOR         = 0.15    # recruiter_response_rate below this = near-zero


def compute_career_score(features: pd.DataFrame) -> np.ndarray:
    """
    65% component — career evidence + title fit.

    Sub-component weights (sum to ~1.0 for a perfect-fit candidate):
        is_eng_title         0.30  — engineering role exists
        built_real_system    0.40  — shipped a ranking/search/recommendation system
        product_vs_services  0.15  — at least one product-company role
        yoe_bonus            0.10  — YOE in 5-9 sweet spot
        hands_on_code_18mo   0.05  — current role is IC engineering

    A candidate with all five scores 1.0 on career; only CAND_0000031-class
    candidates should reach that level on this dataset.
    """
    f = features

    score = np.where(f["is_eng_title"].values.astype(bool), 0.30, 0.0)
    score = score + np.where(f["built_real_system"].values.astype(bool), 0.40, 0.0)
    score = score + np.where(f["product_vs_services"].values.astype(bool), 0.15, 0.0)

    # YOE: tapers outside the sweet spot (not a hard gate — EC notes it is gradual)
    yoe = f["years_of_experience"].values.astype(float)
    yoe_bonus = np.where(
        (yoe >= 5) & (yoe <= 9), 0.10,          # sweet spot (JD: "5 to 9 years")
        np.where((yoe >= 4) & (yoe <= 11), 0.06, 0.02)  # adjacent / outside band
    )
    score = score + yoe_bonus

    # Recency: IC-engineering current role (D3 input — title-inferred; EC-24)
    score = score + np.where(f["hands_on_code_18mo"].values.astype(bool), 0.05, 0.0)

    return np.clip(score, 0.0, 1.0)


def compute_skills_score(features: pd.DataFrame) -> np.ndarray:
    """
    10% component — verified skills signal.

    Deliberately low weight to resist keyword stuffing. Two sub-signals:
      0.50: skill_assessment_scores (independently verified; present 24% of pool)
      0.50: github_activity_score (active coding proxy; -1 = absent, EC-1)

    ai_skill_count is NOT used positively here (it is the stuffer flag input).
    avg_skill_assessment is 0-1 (mean of assessment dict / 100); 0 if absent.
    """
    n = len(features)

    # Sub-signal 1: verified assessment score (0-1 normalized)
    if "avg_skill_assessment" in features.columns:
        assess = features["avg_skill_assessment"].values.astype(float)
    else:
        # Fallback: binary present/absent (for parquet without avg_skill_assessment col)
        assess = np.where(features["has_skill_assessments"].values.astype(bool), 0.65, 0.0)

    # Sub-signal 2: github activity (sentinel -1 = absent; real values 0-100)
    github_raw = features["github_activity_score"].values.astype(float)
    github_norm = np.where(
        github_raw < 0,          # EC-1: -1 sentinel means absent, not zero
        0.0,
        np.clip(github_raw / 100.0, 0.0, 1.0),
    )

    skill_score = 0.50 * assess + 0.50 * github_norm
    return np.clip(skill_score, 0.0, 1.0)


def compute_edu_score(features: pd.DataFrame) -> np.ndarray:
    """5% component — education institution tier."""
    tier_map = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.50, "tier_4": 0.25, "unknown": 0.10}
    scores = features["top_edu_tier"].map(tier_map).fillna(0.10).values.astype(float)
    return np.clip(scores, 0.0, 1.0)


def compute_logistics_modifier(features: pd.DataFrame) -> np.ndarray:
    """
    Small logistics modifier — additive, bounded, must never dominate career evidence.
    All logistics factors also surface as honest concerns in reasoning (Phase 4).
    """
    mod = np.zeros(len(features), dtype=float)

    # Notice: bonus at <=30 (not <30 — strict <30 has only 22 records in 100K; EC-9)
    mod += np.where(features["notice_period_days"].values <= 30, NOTICE_BONUS, 0.0)
    mod += np.where(features["notice_period_days"].values >= 120, NOTICE_LONG_PENALTY, 0.0)

    is_india = features["is_india_based"].values.astype(bool)
    mod += np.where(is_india, INDIA_BONUS, 0.0)
    mod += np.where(features["is_tier1_city"].values.astype(bool), TIER1_BONUS, 0.0)
    mod += np.where(
        features["willing_to_relocate"].values.astype(bool) & ~is_india,
        RELOCATE_BONUS,
        0.0,
    )
    mod += np.where(
        features["preferred_work_mode"].values == "remote",
        REMOTE_ONLY_PENALTY,
        0.0,
    )
    return mod


def compute_base(features: pd.DataFrame, cosine_sim: np.ndarray) -> np.ndarray:
    """
    Composite base score before disqualifier caps or availability multiplier.
    base = 0.65·career + 0.20·embedding + 0.10·skills + 0.05·education + logistics
    """
    career  = compute_career_score(features)
    skills  = compute_skills_score(features)
    edu     = compute_edu_score(features)
    logist  = compute_logistics_modifier(features)

    base = (
        W_CAREER * career
        + W_EMBED  * np.clip(cosine_sim, 0.0, 1.0)
        + W_SKILLS * skills
        + W_EDU    * edu
        + logist
    )
    # Allow small headroom above 1.0 from logistics bonuses; clip at 1.1
    return np.clip(base, 0.0, 1.1)


def apply_disqualifier_caps(base: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    """
    Apply D1–D5 caps to base. Most-restrictive cap wins (EC-27).
    Applied BEFORE availability multiplier (EC-45, decision.md [P1.3]).

    Severity matched to JD wording:
        D1 alone = "we will not move forward" → hard floor 0.02
        D2–D5 = strong penalties (heavy, cap, strong) → not eliminators

    Gate activation rates on 100K (measured Phase 1 [DATA]):
        D1: 0 (no pure researchers in pool)
        D2: 0 (everyone with LangChain also has pre-LLM ML)
        D3: 58.2% — all genuinely non-IC-engineering titles; no collateral on eng titles
        D4: 9.7%  — matches confirmed [DATA] figure exactly
        D5: 0     — CV/robotics candidates absent or all have NLP/IR co-evidence
    """
    gated = base.copy()

    # D1: pure research, zero production — hard floor (EC-22)
    gated = np.where(
        features["d1_research_only"].values.astype(bool),
        np.minimum(gated, D1_CAP),
        gated,
    )

    # D2: AI experience only in recent sub-12-mo LangChain role, no prior ML (EC-23)
    gated = np.where(
        features["d2_recent_llm_only"].values.astype(bool),
        np.minimum(gated, D2_CAP),
        gated,
    )

    # D3: no hands-on IC code in 18 mo (title-inferred; fires on 58.2% but all
    #     are non-engineers already below D3_CAP baseline — zero collateral on eng titles)
    gated = np.where(
        ~features["hands_on_code_18mo"].values.astype(bool),
        np.minimum(gated, D3_CAP),
        gated,
    )

    # D4: entire-career consulting, no product-co exception (EC-25)
    gated = np.where(
        features["d4_all_consulting"].values.astype(bool),
        np.minimum(gated, D4_CAP),
        gated,
    )

    # D5: CV/speech/robotics primary, no NLP/IR evidence (EC-26)
    gated = np.where(
        features["d5_cv_speech_robotics"].values.astype(bool),
        np.minimum(gated, D5_CAP),
        gated,
    )

    return gated


def compute_availability_multiplier(features: pd.DataFrame) -> np.ndarray:
    """
    Continuous availability multiplier ∈ [GHOST_FLOOR, 1.0].
    Applied over honeypot-excluded rows, AFTER disqualifier caps (EC-45).

    Ghost conjunction: staleness > 120 AND rrr < 0.15 AND not open → floor ~0.15
    This targets the 3.4% ghost population [DATA]; stricter thresholds gave 0.8%.

    Healthy engaged candidates: ~0.70–1.0.

    NOT a multiplicative stack — that would push the median to ~0.37× and demote
    the 96.6% healthy majority (decision.md [P3.1], EC-39).

    Band/floor calibrated against synthetic twins in calibrate.py (Phase 3 task):
    clone archetype into engaged/ghost pair; confirm ghost < engaged < unrelated-weaker-engaged.
    """
    staleness = features["staleness_days"].values.astype(float)
    rrr       = features["recruiter_response_rate"].values.astype(float)
    open_flag = features["open_to_work_flag"].values.astype(bool)

    # Ghost: ALL THREE required (EC-35 — any single alone is not a ghost signal)
    ghost_flag = (staleness > STALE_THRESHOLD) & (rrr < RRR_FLOOR) & ~open_flag

    # Continuous engagement score: weighted average of three signals
    staleness_norm = np.clip(staleness / 240.0, 0.0, 1.0)   # 240 = max staleness
    rrr_norm       = np.clip(rrr, 0.0, 1.0)
    open_norm      = np.where(open_flag, 1.0, 0.0)

    engagement = 0.40 * rrr_norm + 0.40 * (1.0 - staleness_norm) + 0.20 * open_norm
    engagement = np.clip(engagement, 0.0, 1.0)

    # Map engagement → multiplier ∈ [GHOST_FLOOR, 1.0]
    multiplier = GHOST_FLOOR + (1.0 - GHOST_FLOOR) * engagement

    # Hard floor override for confirmed ghost conjunction
    multiplier = np.where(ghost_flag, GHOST_FLOOR, multiplier)

    # Enforce healthy band minimum for clearly engaged candidates
    genuinely_engaged = ~ghost_flag & (rrr >= 0.50) & (staleness < 90)
    multiplier = np.where(
        genuinely_engaged,
        np.maximum(multiplier, HEALTHY_BAND_LOW),
        multiplier,
    )

    return np.clip(multiplier, GHOST_FLOOR, 1.0)
