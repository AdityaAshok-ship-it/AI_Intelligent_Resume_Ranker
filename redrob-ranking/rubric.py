"""
rubric.py — Phase 1: relevance rubric (base scorer + disqualifier caps + availability multiplier)

Phase 1 deliverable. Composition order (immutable — EC-45, EC-46, decision.md [P1.3]):
    base  = 0.70·career_title + 0.20·embedding + 0.10·skills + logistics_modifier  (no education — no JD basis)
    gated = apply_disqualifier_caps(base)          # caps on base BEFORE multiplier
    # honeypot rows dropped before this call (Phase 2 / rank.py)
    final = gated × availability_multiplier        # over honeypot-excluded rows

Caps before multiplier (EC-45):
    multiplier ≤ 1, so a disqualified candidate cannot be rescued by high availability.

Honeypot exclusion position (EC-46):
    Rows are DROPPED (not score-adjusted) before the multiplier step in rank.py.
    "A honeypot never enters scoring" — cleaner to defend than post-sort removal.

Weights (calibrated against 50-sample, Phase 1 exit gate):
    Career: 0.70 — dominates; keyword-stuffer defence; most separable score
    Embed:  0.20 — plain-language tier-5 rescue (precomputed cosine similarity)
    Skills: 0.10 — verified assessments + github; deliberately low (keyword stuffing risk)
    Logistics: additive modifier (must never dominate career evidence)
    (Education tier removed 2026-06-18 — no JD basis; see weights block below.)

See decision.md for every weight decision and calibration log.
"""

import numpy as np
import pandas as pd

# ── Component weights ─────────────────────────────────────────────────────────
W_CAREER = 0.70   # was 0.65; absorbed the removed education weight
W_EMBED  = 0.20
W_SKILLS = 0.10
# W_EDU removed 2026-06-18: the job_description.docx (read line-by-line) makes ZERO
# mention of degree/college/university/institution tier; the only "tier" refs are to
# Indian CITIES and the hackathon's candidate tier, and the only "academic" ref is a
# DISQUALIFIER. At 5-9 YOE the production track record dominates pedigree (and class
# rank is unknown anyway), so education tier is not a ranking signal.

# ── Disqualifier cap severities (most-restrictive wins; EC-27) ────────────────
D1_CAP = 0.02   # hard floor — "we will not move forward" (JD verbatim)
D2_CAP = 0.40   # heavy — recent LLM-only with no pre-LLM ML history
D3_CAP = 0.45   # heavy — no hands-on IC code in 18 mo (title-inferred; EC-24)
D4_CAP = 0.35   # cap   — entire-career consulting, no product-co exception
D5_CAP = 0.50   # strong — CV/speech/robotics primary with no NLP/IR evidence
# JD "Things we explicitly do NOT want" + logistics — the two asks added 2026-06-18:
D6_CAP = 0.50   # title-chaser/job-hopper (>=3 employers, <18-mo avg tenure) — JD wants 3+ yr commitment
VISA_CAP = 0.55 # non-India AND not willing to relocate — JD: "we don't sponsor work visas" (case-by-case → cap, not floor)
NO_SHIP_CAP = 0.62  # no career evidence of shipping a ranking/search/ML system — JD primary signal; ensures
                    # built=False candidates never outscore built=True candidates at the scoring boundary.
                    # 0.62 × best_multiplier(0.977) = 0.606, below the weakest built=True in top-100 (0.627).

# ── Logistics modifiers (each small; must not dominate career signal) ─────────
NOTICE_BONUS        =  0.03   # notice_period_days <= 30 (EC-9: <=30 has 13.8%; <30 has 22 people)
NOTICE_MID1_PENALTY = -0.01   # notice_period_days 31–60  — JD: "bar gets higher" for 30+
NOTICE_MID2_PENALTY = -0.02   # notice_period_days 61–119 — moderate friction
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
    70% component — career evidence + title fit.

    Sub-component weights (sum to 1.0 for a perfect sweet-spot candidate):
        is_eng_title         0.25  — engineering role exists
        built_real_system    0.35  — shipped a ranking/search/recommendation system
        product_vs_services  0.15  — at least one product-company role
        yoe_bonus       0.20-0.24  — YOE, peaked & gently RISING across the 5-9 band
        hands_on_code_18mo   0.05  — current role is IC engineering

    Rebalanced 2026-06-18: title/system weights lowered (0.30/0.40 -> 0.25/0.35) and
    YOE raised (0.10 -> 0.20). The old weights saturated the base near 0.90 for every
    strong candidate, leaving the availability multiplier to dominate the final order;
    giving YOE real weight lets experience separate candidates. See change.md.
    """
    f = features

    score = np.where(f["is_eng_title"].values.astype(bool), 0.25, 0.0)
    score = score + np.where(f["built_real_system"].values.astype(bool), 0.35, 0.0)
    score = score + np.where(f["product_vs_services"].values.astype(bool), 0.15, 0.0)

    # YOE: sweet-spot peaked (JD: "5 to 9 years"). WITHIN the 5-9 band the bonus
    # rises gently 0.20 -> 0.24 so that, between two otherwise-equal sweet-spot
    # candidates, the more experienced one edges ahead (a 6.9y beats a 5.4y);
    # smooth taper outside the band. The +0.04 within-band ramp needs headroom
    # above 1.0 (a full-profile 9y reaches 1.04), so the clip is raised to 1.05.
    yoe = f["years_of_experience"].values.astype(float)
    yoe_bonus = np.select(
        [(yoe >= 5) & (yoe <= 9),                               # sweet spot (monotonic)
         ((yoe >= 4) & (yoe < 5)) | ((yoe > 9) & (yoe <= 11)),  # adjacent
         ((yoe >= 3) & (yoe < 4)) | ((yoe > 11) & (yoe <= 13))],# near
        [0.20 + 0.01 * (yoe - 5), 0.12, 0.06], default=0.02)    # far
    score = score + yoe_bonus

    # Recency: IC-engineering current role (D3 input — title-inferred; EC-24)
    score = score + np.where(f["hands_on_code_18mo"].values.astype(bool), 0.05, 0.0)

    return np.clip(score, 0.0, 1.05)


def compute_skills_score(features: pd.DataFrame) -> np.ndarray:
    """
    10% component — verified skills signal.

    Deliberately low weight to resist keyword stuffing. Two sub-signals:
      0.75: skill_assessment_scores (independently verified; present 24% of pool)
      0.25: github_activity_score (weak proxy per JD; -1 = absent, EC-1)

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

    skill_score = 0.75 * assess + 0.25 * github_norm
    return np.clip(skill_score, 0.0, 1.0)


def compute_logistics_modifier(features: pd.DataFrame) -> np.ndarray:
    """
    Small logistics modifier — additive, bounded, must never dominate career evidence.
    All logistics factors also surface as honest concerns in reasoning (Phase 4).
    """
    mod = np.zeros(len(features), dtype=float)

    # Notice: 4-tier graduated modifier. JD: "love sub-30, can buy out 30, 30+ bar gets higher."
    # (not <30 — strict <30 has only 22 records in 100K; EC-9)
    notice = features["notice_period_days"].values
    mod += np.select(
        [notice <= 30,
         (notice >= 31) & (notice <= 60),
         (notice >= 61) & (notice <= 119),
         notice >= 120],
        [NOTICE_BONUS, NOTICE_MID1_PENALTY, NOTICE_MID2_PENALTY, NOTICE_LONG_PENALTY],
        default=0.0,
    )

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
    base = 0.70·career + 0.20·embedding + 0.10·skills + logistics
    (Education tier removed 2026-06-18 — no JD basis; its 0.05 was folded into career.)
    """
    career  = compute_career_score(features)
    skills  = compute_skills_score(features)
    logist  = compute_logistics_modifier(features)

    # Education tier is intentionally NOT in the base — no JD basis (see weights note).
    base = (
        W_CAREER * career
        + W_EMBED  * np.clip(cosine_sim, 0.0, 1.0)
        + W_SKILLS * skills
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

    # D6: title-chaser / job-hopper — JD "do NOT want" (listed first): switching
    # companies every ~1.5 years. Cap, not floor — a strong builder who happens to
    # have moved a lot is demoted below committed peers, not deleted. Guarded so a
    # stale parquet (column absent) is a no-op rather than a crash.
    if "title_chaser_flag" in features.columns:
        gated = np.where(
            features["title_chaser_flag"].values.astype(bool),
            np.minimum(gated, D6_CAP),
            gated,
        )

    # Location/visa: non-India AND not willing to relocate. JD: "Outside India:
    # case-by-case, but we don't sponsor work visas." Such a candidate is
    # realistically un-hireable, so cap below the genuine hireable band; "case-by-case"
    # is why this is a cap (re-rankable on exception) rather than a hard floor.
    visa_blocked = (
        ~features["is_india_based"].values.astype(bool)
        & ~features["willing_to_relocate"].values.astype(bool)
    )
    gated = np.where(visa_blocked, np.minimum(gated, VISA_CAP), gated)

    # No-ship cap: JD primary signal is "shipped a ranking/search/recommendation system
    # at a product company." Without career evidence of this, the candidate is a weaker
    # fit regardless of availability. Cap ensures built=False never outscores built=True
    # at the scoring boundary (audit finding: rank-100 no-ship was displacing rank-101
    # built=True by 0.001 on availability alone). Cap set so max final for no-ship =
    # 0.62 × best_multiplier(0.977) = 0.606, below any boundary built=True candidate.
    gated = np.where(
        ~features["built_real_system"].values.astype(bool),
        np.minimum(gated, NO_SHIP_CAP),
        gated,
    )

    return gated


def compute_availability_multiplier(features: pd.DataFrame) -> np.ndarray:
    """
    Continuous availability multiplier ∈ [GHOST_FLOOR, 1.0]. Applied over
    honeypot-excluded rows, AFTER disqualifier caps (EC-45).

    Redesigned 2026-06-18 (see change.md): ONE continuous engagement dimension
    combining the three signals the JD names — "actively available, open to
    recruiters" — linearly mapped to [0.15, 1.0]:

        multiplier = 0.15 + 0.85 · engagement
        engagement = 0.20·ActiveScore + 0.45·ResponseRate + 0.35·OpenFlag

    ActiveScore is an EVEN, continuous 15-day-slot ramp: 1.0 for ≤15 days, then
    steps down by 1/7 each slot to 0.0 at the 120-day ghost line. So a 1-day and a
    15-day candidate are equal; each later slot costs the same.

    Weighting (tuned 2026-06-18): recency (ActiveScore) is the LIGHTEST signal at
    0.20 — a strong passive candidate who simply hasn't logged in for ~40 days is
    still highly hireable (LinkedIn's passive-candidate principle), so recency must
    not override experience. Response rate (0.45, the best predictor of whether you
    can actually engage them) and open-to-work (0.35) carry the weight. Net effect:
    a 41-day-stale 8-YOE candidate keeps a ~0.89 multiplier and stays above a fresh
    4-YOE one — experience wins, as required.

    The 0.15 floor is reached only at the GHOST CORNER (inactive AND unresponsive
    AND not open). A very-stale but responsive-and-open candidate still floors
    near 0.83 (rrr+open hold the score up) — a reachable passive candidate, which
    the JD values. Availability stays SECONDARY to the experience base by design
    (the JD lists it as one line among ~12 experience/technical requirements).
    """
    staleness = features["staleness_days"].values.astype(float)
    rrr       = features["recruiter_response_rate"].values.astype(float)
    open_flag = features["open_to_work_flag"].values.astype(bool)

    # True-ghost conjunction (EC-35): inactive AND unresponsive AND not open.
    ghost_flag = (staleness > STALE_THRESHOLD) & (rrr < RRR_FLOOR) & ~open_flag

    # ActiveScore: even 15-day-slot ramp, 1.0 (≤15d) → 0.0 (≥120d). Slots 0..7.
    slot = np.minimum(np.floor(staleness / 15.0), 8.0)
    active_score = np.clip(1.0 - slot / 7.0, 0.0, 1.0)

    # One continuous engagement dimension → [GHOST_FLOOR, 1.0].
    engagement = (
        0.20 * active_score                     # recency — lightest signal (shrunk from 0.45)
        + 0.45 * np.clip(rrr, 0.0, 1.0)         # response rate — strongest engagement signal
        + 0.35 * open_flag.astype(float)        # open-to-work — direct availability statement
    )
    multiplier = GHOST_FLOOR + (1.0 - GHOST_FLOOR) * np.clip(engagement, 0.0, 1.0)

    # Hard floor guarantees the confirmed ghost conjunction lands exactly at 0.15.
    multiplier = np.where(ghost_flag, GHOST_FLOOR, multiplier)

    return np.clip(multiplier, GHOST_FLOOR, 1.0)
