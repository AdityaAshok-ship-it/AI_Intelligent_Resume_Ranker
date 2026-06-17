"""
detectors.py — Phase 2: trap and honeypot detection

Phase 2 deliverable. All detectors consume pre-computed columns from features.parquet —
they never re-parse the raw JSON corpus.

Build order (by risk, per implementation-plan.md Phase 2):
    1. Honeypot detector  — binary DQ insurance (H1–H4, hard-exclude)
    2. Keyword-stuffer detector — flag, do not double-penalise
    3. Top-150 audit guard — post-sort sanity check

The H1–H4 flags are pre-computed by precompute.py and stored in features.parquet.
This module reads them; it does NOT recompute them (that would require the JSON corpus).

Verified [DATA, 2026-06-15]:
    H1–H4 union catches exactly 68 records; near-zero collateral.
    Honeypots wear attractive titles (CAND_0010770 = "Recommendation Systems Engineer")
    — caught ONLY by H3, not by title or keywords.
    → Exclusion is a PRE-SORT HARD GATE, not a score penalty.
"""

import numpy as np
import pandas as pd

# Keyword-stuffer threshold: non-eng title + ≥3 AI skills + no career evidence
# Threshold held at ≥3 (lowering to ≥2 adds 166 genuine 2-skill transitioners
# for zero ranking gain — see edgecases.md EC-17, EC-18).
STUFFER_AI_SKILL_THRESHOLD = 3


def get_honeypot_mask(features: pd.DataFrame) -> np.ndarray:
    """
    Return boolean mask (True = honeypot, exclude from ranking).
    Reads pre-computed h1–h4 flags from features.parquet.

    Slack values (verified against full 100K [DATA]):
        H1: +3 mo  (date rounding)
        H2: +30 mo (concurrent roles)
        H3: +18 mo (early internships, protects 18.5% single-role pool)
        H4: exact 0 at {advanced, expert}

    DO NOT apply the naive "skill duration > career length" rule —
    it fires on 13–19% of the pool and deletes CAND_0000031 (confirmed).
    """
    return features["honeypot_flag"].values.astype(bool)


def get_stuffer_flag(features: pd.DataFrame) -> np.ndarray:
    """
    Keyword-stuffer flag: non-eng title AND ≥3 AI skills AND no built_real_system.

    FLAG only — do not double-penalise. Career-first 65% weight already demotes
    stuffers below genuine fits. Aggressive penalty false-positives genuine
    career-changers (the modal real changer has 2 AI skills — EC-18).

    Quote the count from YOUR shipped detector, not the doc's 3.6% (EC-63).
    """
    non_eng = ~features["is_eng_title"].values.astype(bool)
    high_ai = features["ai_skill_count"].values >= STUFFER_AI_SKILL_THRESHOLD
    no_system = ~features["built_real_system"].values.astype(bool)
    return non_eng & high_ai & no_system


def top_150_audit(
    top_indices: np.ndarray,
    features: pd.DataFrame,
    ids_arr: np.ndarray,
) -> np.ndarray:
    """
    Post-sort audit over the top ~150 candidates (cheap vs. 100K-wide rules).

    Steps (deterministic + logged; the human eyeball is a CHECK, never a CSV edit):
        1. Re-run H1–H4 over the top 150 (catches any precompute omission).
        2. Founding-date plausibility for KNOWN REAL companies only
           (Swiggy, Razorpay, Paytm — companies in FOUNDING_YEAR_MAP in precompute.py).
        3. Print top-10 diagnostic for human "too good to be true" review.

    Residual-honeypot note (EC-33, decision.md [P2.3]):
        ~12 uncaught honeypots are at FICTIONAL companies (Hooli, Stark, Dunder Mifflin).
        The founding-date check CANNOT see those; their risk is bounded by the
        >10-honeypots-in-top-100 DQ threshold — improbable for ~11 of ~12 to both
        surface and out-score real fits. Do NOT claim the audit catches them.

    The 'founding_date_anomaly' column is precomputed by precompute.py.
    If absent (old features.parquet), the check is skipped with a warning —
    re-run `python precompute.py --candidates candidates.jsonl --dry-run` to enable.

    Returns: filtered top_indices (score order; length shrinks only if late detections).
    """
    top_features = features.iloc[top_indices].reset_index(drop=True)

    # Combined exclusion mask: True = remove from surviving set
    exclude = np.zeros(len(top_indices), dtype=bool)

    # ── 1. Re-run H1–H4 as a safety net ───────────────────────────────────────
    late_honeypots = get_honeypot_mask(top_features)
    n_late = int(late_honeypots.sum())
    if n_late > 0:
        late_ids = top_features[late_honeypots]["candidate_id"].tolist()
        print(f"  AUDIT H1-H4: {n_late} late honeypot(s) — removing: {late_ids}")
    exclude |= late_honeypots

    # ── 2. Founding-date plausibility (known real companies only) ──────────────
    if "founding_date_anomaly" in top_features.columns:
        founding_anomaly = top_features["founding_date_anomaly"].values.astype(bool)
        n_founding = int(founding_anomaly.sum())
        if n_founding > 0:
            founding_ids = top_features[founding_anomaly]["candidate_id"].tolist()
            print(
                f"  AUDIT founding-date: {n_founding} anomaly(s) (role before company "
                f"founding) — removing: {founding_ids}"
            )
        exclude |= founding_anomaly
    else:
        print(
            "  AUDIT: 'founding_date_anomaly' column absent — re-run "
            "`python precompute.py --candidates candidates.jsonl --dry-run` to enable."
        )

    n_total = int(exclude.sum())
    if n_total > 0:
        print(f"  AUDIT: {n_total} total removed from top {len(top_indices)}.")
    else:
        print(f"  AUDIT: top {len(top_indices)} clean — no late detections.")

    # ── 3. Top-10 diagnostic for human review ("too good to be true" check) ────
    surviving_features = top_features[~exclude]
    diag_cols = [c for c in [
        "candidate_id", "current_title", "is_eng_title",
        "built_real_system", "honeypot_flag", "staleness_days",
    ] if c in surviving_features.columns]
    pd.set_option("display.max_colwidth", 45)
    print("\n  AUDIT top-10 after filtering (human eyeball — look for implausible profiles):")
    print(surviving_features.head(10)[diag_cols].to_string(index=False))
    print()

    return top_indices[~exclude]
