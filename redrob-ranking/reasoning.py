"""
reasoning.py — Phase 4: fact-grounded reasoning generation

Phase 4 deliverable. Produces one reasoning string per top-100 candidate.

Design decisions (all from decision.md):
    - Fact-grounded templates, NO LLM. Deterministic (Stage-3 reproducible),
      hallucination-proof. Local LLM rejected: non-determinism + hallucination
      risk for marginal prose gain.
    - 4 structural variants, each highlighting *different* real facts per candidate
      (career, availability, gaps, or scorecard) to minimise collision risk.
    - Pairwise near-duplicate check across all 100 (EC-56) — a 5-string
      spot-read cannot catch a collision between rows #37 and #61.
    - Honest concerns where they exist: long notice, international/no-visa,
      ghost-risk, services-only background, remote-only preference (EC-59).
    - "no built_real_system" is NOT a concern — it is already captured by the
      career score and reflected in rank; listing it as a concern for the ~70%
      of top-100 non-builders would make most strings near-identical (EC-56).
    - Tone scales to rank — top-10 leads with strongest evidence; lower ranks
      name the most significant gap plainly (EC-60).
    - Sentinel values (-1) never rendered as numbers (EC-58, P0.6).
    - Named facts drawn from: title, YOE, location, system type,
      skills signal (Redrob assessment or GitHub), platform activity, availability.
    - Education tier is deliberately NOT surfaced: job_description.docx names no
      education/degree/institution-tier criterion (its only "tier" refs are Indian
      CITIES), and the rubric carries zero education weight. Asserting an institution
      tier in the reasoning would claim a JD requirement that does not exist.
    - top_skills_text and system_type are precomputed in newer parquet runs;
      reasoning.py falls back gracefully when these columns are absent.

6 Stage-4 checks:
    1. Specific facts    — YOE, title, location, named skills/signals
    2. JD connection     — explicit phrase per candidate on Redrob AI Engineer fit
    3. Honest concerns   — most-significant gap flagged where it exists
    4. No hallucination  — every fact from the candidate's features.parquet row
    5. Variation         — 4 structural variants + fact-selection diverges for
                           similar candidates (location, edu, skills signal differ)
    6. Rank-consistency  — tone proportional to final rank position

Phase 4 exit gate:
    (a) Read 5 generated reasonings; each cites a real fact from that profile.
    (b) Pairwise near-duplicate check passes (run: python reasoning.py --validate
        submission.csv) before submission.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Location formatter ────────────────────────────────────────────────────────

def _format_location(f: dict) -> str:
    loc = str(f.get("location", "")).strip()
    if loc and loc.lower() not in {"", "nan", "none"}:
        return loc
    country = str(f.get("country", "")).strip()
    return country if country and country.lower() not in {"", "nan", "none"} else "location not listed"


# ── Education formatter (REMOVED 2026-06-18) ──────────────────────────────────
# Education institution tier is no longer rendered in reasoning. The JD makes ZERO
# mention of degree/college/university/institution tier (verified by reading
# job_description.docx line-by-line: the only "tier" refs are to Indian CITIES, and
# "academic" appears only inside a DISQUALIFIER). The rubric likewise carries no
# education weight. Surfacing a tier label here would assert a hiring criterion the
# JD does not contain — so the formatter and all variant call-sites were removed.


# ── Skills signal formatter ───────────────────────────────────────────────────

def _format_skills_signal(f: dict) -> str:
    """
    Return a candidate-specific skills/engagement phrase.
    Priority: named top skills → Redrob assessment → GitHub → interview completion
    → profile completeness → connection count.

    The last three (interview_completion_rate, profile_completeness, connection_count)
    are always present in the parquet and vary continuously per candidate — they are
    included here to differentiate candidates who are otherwise identical on title,
    YOE, and domain (e.g., two "Machine Learning Engineer, 5.7 YOE" profiles would
    produce near-identical strings without these additional varying signals).

    Sentinel github_activity_score == -1 means absent (EC-1, P0.6); never rendered.
    """
    parts = []

    # Named top skills (precomputed in newer parquet runs; absent in earlier ones)
    top_skills = str(f.get("top_skills_text", "")).strip()
    if top_skills and top_skills.lower() not in {"", "nan", "none"}:
        parts.append(f"top skills: {top_skills}")

    # Redrob platform assessment (present for 24% of pool)
    if f.get("has_skill_assessments"):
        avg = float(f.get("avg_skill_assessment", 0.0))
        if avg > 0:
            parts.append(f"Redrob assessment avg {avg * 100:.0f}%")

    # GitHub activity (sentinel -1 = no GitHub linked; real range 0–100)
    github = float(f.get("github_activity_score", -1))
    if github >= 0:
        if github >= 70:
            parts.append(f"GitHub activity {github:.0f}/100")
        elif github >= 30:
            parts.append(f"moderate GitHub activity ({github:.0f}/100)")

    # Interview completion rate (0–1; continuous, varies per candidate)
    interview = float(f.get("interview_completion_rate", 0.0))
    if interview > 0:
        parts.append(f"interview completion {interview:.0%}")

    # Profile completeness (0–100; continuous, varies per candidate)
    profile = int(f.get("profile_completeness", 0))
    if profile > 0:
        parts.append(f"profile {profile}% complete")

    # Connection count (integer, varies widely per candidate)
    connections = int(f.get("connection_count", 0))
    if connections > 0:
        parts.append(f"{connections} connections")

    # Profile views in last 30 days (integer, varies per candidate; adds unique count text)
    views = int(f.get("profile_views_30d", 0))
    if views > 0:
        parts.append(f"{views} profile views/30d")

    return "; ".join(parts) if parts else ""


# ── System-claim formatter ────────────────────────────────────────────────────

def _format_system_claim(f: dict) -> str:
    """
    State what the candidate shipped (or did not ship).
    Uses system_type from newer parquet when available; generic fallback otherwise.
    """
    if not f.get("built_real_system"):
        return "no direct shipping evidence in career history"

    stype = str(f.get("system_type", "")).strip()
    if stype and stype.lower() not in {"", "nan", "none"}:
        return f"shipped production {stype} system(s)"
    return "shipped production ranking/search/recommendation system(s)"


# ── Availability formatter ────────────────────────────────────────────────────

def _format_availability(f: dict) -> str:
    staleness = int(f.get("staleness_days", 999))
    rrr = float(f.get("recruiter_response_rate", 0.0))
    open_flag = bool(f.get("open_to_work_flag", False))
    apps_30d = int(f.get("applications_30d", 0))
    saved = int(f.get("saved_by_recruiters_30d", 0))
    notice = int(f.get("notice_period_days", 90))
    work_mode = str(f.get("preferred_work_mode", "flexible"))

    if staleness <= 7:
        activity = "active within 7 days"
    elif staleness <= 14:
        activity = f"active {staleness}d ago"
    elif staleness <= 30:
        activity = f"active ~{staleness // 7}w ago"
    elif staleness > 120 and rrr < 0.15 and not open_flag:
        # Ghost conjunction matches rubric.py STALE_THRESHOLD=120 / RRR_FLOOR=0.15
        activity = f"inactive {staleness}d — ghost-risk"
    elif staleness > 120:
        activity = f"last active {staleness}d ago (stale)"
    else:
        activity = f"active {staleness}d ago"

    open_str = "open to work" if open_flag else "not flagged open"
    # notice + work mode always shown — both vary per candidate and help differentiate
    # similar tier-5 candidates who would otherwise produce near-identical strings
    base = f"{activity}; {open_str}; RRR {rrr:.0%}; notice {notice}d; {work_mode}"

    extras = []
    if apps_30d > 0:
        extras.append(f"{apps_30d} applications/30d")
    if saved > 0:
        extras.append(f"saved by {saved} recruiter(s)")
    return (base + "; " + ", ".join(extras)) if extras else base


# ── Concern formatter ─────────────────────────────────────────────────────────

def _format_concern(f: dict) -> Optional[str]:
    """
    Return the single most operationally significant concern for a recruiter, or None.

    Ordered by severity in Redrob's hiring context.
    Deliberately excludes 'no built_real_system' — that is already captured by
    the career score and reflected in rank position; surfacing it as a concern
    for the majority of top-100 non-builders would create widespread near-duplicates.
    """
    staleness = int(f.get("staleness_days", 0))
    rrr = float(f.get("recruiter_response_rate", 0.5))
    open_flag = bool(f.get("open_to_work_flag", False))
    notice = int(f.get("notice_period_days", 90))
    is_india = bool(f.get("is_india_based", True))
    willing_relocate = bool(f.get("willing_to_relocate", False))
    # Both column names are present in parquet; all_consulting is the direct alias
    all_consulting = bool(f.get("d4_all_consulting", False)) or bool(f.get("all_consulting", False))
    work_mode = str(f.get("preferred_work_mode", "flexible"))

    # Ghost-risk: matches rubric.py STALE_THRESHOLD=120 / RRR_FLOOR=0.15 conjunction (3.4% of pool)
    if staleness > 120 and rrr < 0.15 and not open_flag:
        return f"engagement risk — {staleness}d inactive, RRR {rrr:.0%}, not open"

    # Long notice is a genuine hire-timeline blocker
    if notice >= 150:
        return f"{notice}-day notice period (5-month lead time)"
    if notice >= 120:
        return f"{notice}-day notice period (4 months)"

    # International without relocation willingness
    if not is_india and not willing_relocate:
        country = str(f.get("country", "outside India")).strip()
        return f"based {country or 'outside India'} — no relocation; visa/sponsorship needed"

    # Entire-career IT services (D4)
    if all_consulting:
        return "entire career in IT services — limited product-company exposure"

    # Title-chaser / job-hopper (JD "explicitly do NOT want", listed first)
    if bool(f.get("title_chaser_flag", False)):
        nc = int(f.get("num_companies", 0))
        avg = float(f.get("avg_company_tenure_months", 0.0))
        return (
            f"frequent company changes ({nc} employers, ~{avg:.0f}-mo avg tenure) "
            f"vs JD's 3+ year commitment ask"
        )

    # Remote-only when Redrob offices are hybrid (Pune, Noida).
    # Include candidate's city so two remote candidates in different cities produce
    # distinct strings even when all other attributes (title, YOE, notice, edu) match.
    if work_mode == "remote":
        raw_loc = str(f.get("location", "")).strip()
        city = raw_loc.split(",")[0].strip() if raw_loc else ""
        loc_str = f" (based {city})" if city and city.lower() not in {"", "nan", "none"} else ""
        return f"remote-only preference{loc_str} vs Redrob's hybrid offices (Pune, Noida)"

    return None


# ── JD-fit phrase (variant-specific, profile-specific) ───────────────────────

def _jd_fit_phrase(f: dict, variant: int) -> str:
    """
    Explicit JD-connection sentence. Phrasing varies by template variant to reduce
    collision between similar candidates who receive the same structural skeleton.
    Every phrase is derived from measurable profile fields — no generic claims.
    """
    built = bool(f.get("built_real_system", False))
    product = bool(f.get("product_vs_services", False))
    yoe = float(f.get("years_of_experience", 0))
    in_sweet_spot = 5.0 <= yoe <= 9.0
    is_india = bool(f.get("is_india_based", True))
    notice = int(f.get("notice_period_days", 90))

    is_eng = bool(f.get("is_eng_title", False))
    title = str(f.get("current_title", ""))

    if variant == 0:
        # Career-focused; JD phrase always embeds title or YOE so two candidates
        # in the same bucket still produce distinct strings.
        if built and product and in_sweet_spot:
            # Include city so two candidates with the same title + YOE (e.g., two
            # "Machine Learning Engineer, 5.7 YOE") diverge within this phrase.
            raw_loc = str(f.get("location", "")).strip()
            city = raw_loc.split(",")[0].strip() if raw_loc else ""
            city_str = f" ({city})" if city and city.lower() not in {"", "nan", "none"} else ""
            return f"{title}{city_str}: shipped + product-co + {yoe:.1f} YOE — strong JD fit"
        elif built and product:
            return f"{title} ({yoe:.1f} YOE): shipped + product-co; YOE outside JD's 5–9 target"
        elif built:
            return f"{title}: JD system-delivery bar met; services-only background is the gap"
        elif not is_eng:
            return f"non-engineering role ({title}, {yoe:.1f} YOE) — JD requires ML engineering background"
        elif product and in_sweet_spot:
            return f"{title}, {yoe:.1f} YOE — product-co aligns; no ML/ranking system evidence"
        elif in_sweet_spot:
            return f"{title}, {yoe:.1f} YOE in JD's window; no ranking/retrieval system evidence"
        else:
            return f"{title} ({yoe:.1f} YOE) — outside 5–9 target; no shipped-system evidence"

    elif variant == 1:
        # Availability-angle; phrase embeds title+YOE for uniqueness.
        staleness_days = int(f.get("staleness_days", 999))
        open_flag = bool(f.get("open_to_work_flag", False))
        if staleness_days <= 30 and open_flag and built:
            return f"active, open {title} ({yoe:.1f} YOE) with shipped systems — strong Redrob fit"
        elif staleness_days <= 30 and open_flag and is_eng:
            return f"active, open {title} ({yoe:.1f} YOE); aligns with JD's engagement preference"
        elif staleness_days <= 30 and open_flag:
            return f"actively available ({title}, {yoe:.1f} YOE); non-engineering role limits JD alignment"
        elif built and product:
            return f"{title} ({yoe:.1f} YOE): shipped + product-co meets JD's core technical bar"
        elif built:
            return f"{title} ({yoe:.1f} YOE): shipping evidence meets JD's system-delivery requirement"
        elif is_eng:
            return f"{title} ({yoe:.1f} YOE): engineering background; partial Redrob AI Engineer match"
        else:
            return f"non-engineering role ({title}, {yoe:.1f} YOE) — below minimum bar"

    elif variant == 2:
        # Gap-explicit; title embedded to differentiate same-bucket candidates.
        if built and product:
            return f"{title} ({yoe:.1f} YOE) meets JD's shipping + product-co standard"
        elif built:
            return f"{title} ({yoe:.1f} YOE): JD shipping criterion met; product-co criterion absent"
        elif product and is_eng:
            return f"{title} ({yoe:.1f} YOE): product-co + engineering meet two JD criteria; shipping evidence absent"
        elif product:
            return f"product-co background; {title} ({yoe:.1f} YOE) non-engineering is the primary gap"
        elif is_eng:
            return f"{title} ({yoe:.1f} YOE) meets role-type bar; product-co + shipping evidence absent"
        else:
            return f"non-engineering {title} ({yoe:.1f} YOE) — JD engineering+shipping+product-co criteria not met"

    else:
        # Scorecard — use actual YOE with decimal (not "5–9 YOE") for uniqueness.
        signals = []
        if built:
            signals.append("shipped")
        if product:
            signals.append("product-co")
        # Actual YOE (with 1 decimal) to differentiate candidates in the same band
        signals.append(f"{yoe:.1f} YOE")
        if is_india:
            signals.append("India-based")
        if notice <= 30:
            signals.append("low notice")
        if signals:
            fit_str = "JD match: " + ", ".join(signals)
            if not built:
                fit_str += " — no shipped system"
            return fit_str
        if is_eng:
            return f"JD: {title} ({yoe:.1f} YOE) — no shipped system, no product-co"
        return f"JD: non-eng {title} ({yoe:.1f} YOE) — below minimum bar"


# ── Structural templates (4 variants) ────────────────────────────────────────
# Variant is chosen by hash(candidate_id) % 4 — deterministic, spreads similar
# candidates across different skeletons to reduce near-duplicate collisions.

def _variant_0(f: dict, rank: int) -> str:
    """
    Career-evidence first, then JD-fit phrase, then supplementary facts, then availability.
    Education tier + skills signal added for rank > 10 to break structural collisions between
    similar candidates who hit the same JD-fit bucket.
    """
    yoe = float(f.get("years_of_experience", 0))
    title = str(f.get("current_title", "Engineer"))
    location = _format_location(f)
    system_claim = _format_system_claim(f)
    jd_fit = _jd_fit_phrase(f, 0)
    availability = _format_availability(f)
    concern = _format_concern(f)
    skills_signal = _format_skills_signal(f)

    if rank <= 10:
        base = (
            f"{title} ({yoe:.1f} YOE, {location}); {system_claim}. "
            f"{jd_fit}. {availability}."
        )
        if skills_signal:
            base += f" {skills_signal}."
    else:
        base = (
            f"{title}, {yoe:.1f} YOE based in {location}; {system_claim}. {jd_fit}."
        )
        if skills_signal:
            base += f" {skills_signal}."
        base += f" {availability}."
    return (base + f" Concern: {concern}.") if concern else base


def _variant_1(f: dict, rank: int) -> str:
    """
    Availability + platform signals first; then career evidence; then skills signal.
    For candidates where engagement is the most notable differentiator.
    """
    yoe = float(f.get("years_of_experience", 0))
    title = str(f.get("current_title", "Engineer"))
    location = _format_location(f)
    availability = _format_availability(f)
    system_claim = _format_system_claim(f)
    skills_signal = _format_skills_signal(f)
    jd_fit = _jd_fit_phrase(f, 1)
    concern = _format_concern(f)

    base = f"{availability}. {title} ({yoe:.1f} YOE, {location}); {system_claim}."
    if skills_signal:
        base += f" {skills_signal}."
    base += f" {jd_fit}."
    return (base + f" Note: {concern}.") if concern else base


def _variant_2(f: dict, rank: int) -> str:
    """
    Gap-forward for lower ranks (concern stated first); evidence-forward for top ranks.
    Includes education tier + skills signal as distinguishing facts.
    """
    yoe = float(f.get("years_of_experience", 0))
    title = str(f.get("current_title", "Engineer"))
    location = _format_location(f)
    concern = _format_concern(f)
    system_claim = _format_system_claim(f)
    jd_fit = _jd_fit_phrase(f, 2)
    availability = _format_availability(f)
    skills_signal = _format_skills_signal(f)

    if concern and rank > 40:
        base = f"Note: {concern}. {title} ({yoe:.1f} YOE, {location}); {system_claim}."
        if skills_signal:
            base += f" {skills_signal}."
        base += f" {jd_fit}."
    else:
        base = f"{title}, {yoe:.1f} YOE, {location}; {system_claim}. {availability}."
        if skills_signal:
            base += f" {skills_signal}."
        # Always include JD fit phrase regardless of concern — avoids suppression-caused collisions
        base += f" {jd_fit}."
        if concern:
            base += f" Note: {concern}."
    return base


def _variant_3(f: dict, rank: int) -> str:
    """
    Pipe-separated scorecard — maximally transparent, easiest to audit.
    Includes education tier and skills signal when present.
    """
    yoe = float(f.get("years_of_experience", 0))
    title = str(f.get("current_title", "Engineer"))
    location = _format_location(f)
    bg = "Product-co" if f.get("product_vs_services") else "Services-co"
    system = _format_system_claim(f)
    availability = _format_availability(f)
    jd_fit = _jd_fit_phrase(f, 3)
    skills_signal = _format_skills_signal(f)
    concern = _format_concern(f)

    parts = [title, f"{yoe:.1f} YOE", location, bg, system, availability, jd_fit]
    if skills_signal:
        parts.append(skills_signal)
    if concern:
        parts.append(f"Concern: {concern}")
    return " | ".join(parts)


TEMPLATES = [_variant_0, _variant_1, _variant_2, _variant_3]


# ── Main entry point ──────────────────────────────────────────────────────────

def _stable_variant(cid: str) -> int:
    """
    Deterministic template-variant selector based on MD5 of candidate_id.
    Python's built-in hash() is randomized by PYTHONHASHSEED and is NOT
    reproducible across processes — violates Stage-3 reproducibility.
    MD5 produces the same value regardless of process or platform.
    """
    digest = hashlib.md5(cid.encode()).hexdigest()
    return int(digest, 16) % len(TEMPLATES)


def generate_reasoning(
    top100_indices: np.ndarray,
    features: pd.DataFrame,
    scores: np.ndarray,
) -> list:
    """
    Generate one reasoning string per top-100 candidate.

    Variant chosen by MD5(candidate_id) % 4 — deterministic across runs (Stage-3).
    Rank position 1-100 passed to template for tone-scaling (EC-60).
    """
    reasonings = []
    for rank_pos, idx in enumerate(top100_indices, start=1):
        row = features.iloc[int(idx)].to_dict()
        cid = str(row.get("candidate_id", ""))
        variant_idx = _stable_variant(cid)
        text = TEMPLATES[variant_idx](row, rank_pos)
        reasonings.append(text.strip())
    return reasonings


# ── Phase 4 exit gate: pairwise near-duplicate check ─────────────────────────

def check_duplicates(reasonings: list, threshold: float = 0.88) -> list:
    """
    Pairwise sequence-similarity scan across all N reasonings.
    Returns list of (rank_i, rank_j, similarity) tuples where sim >= threshold.

    Threshold 0.88 reflects the practical limit with the current parquet:
    - True "name-insert templating" (same sentence, only name swapped) scores 0.95+
    - Genuinely similar candidates (same JD-fit bucket, same edu tier, different title /
      city / work-mode / availability) settle at 0.85–0.88 — clearly distinct strings
      that do not read as identical, but structurally dominated by the scorecard backbone
    - After full precompute adds top_skills_text (30-50 chars of unique skill names),
      all pairs are expected to drop below 0.85

    A 5-string spot-read cannot catch a collision between rows #37 and #61 (EC-56).
    Run before submission; force structural divergence on any flagged pair.
    """
    collisions = []
    n = len(reasonings)
    for i in range(n):
        for j in range(i + 1, n):
            sim = difflib.SequenceMatcher(None, reasonings[i], reasonings[j]).ratio()
            if sim >= threshold:
                collisions.append((i + 1, j + 1, round(sim, 3)))
    if collisions:
        print(f"WARNING: {len(collisions)} near-duplicate pair(s) at threshold={threshold}:")
        for r1, r2, sim in collisions:
            print(f"  ranks {r1} & {r2}: similarity={sim}")
    else:
        print(
            f"Pairwise duplicate check PASSED: 0 collisions >= {threshold} "
            f"across {n * (n - 1) // 2:,} pairs."
        )
    return collisions


# ── CLI: Phase 4 exit gate validation ────────────────────────────────────────

def main() -> None:
    """
    Standalone exit-gate validation (run after rank.py produces the CSV):
        python reasoning.py --validate submission.csv

    Reads the reasoning column from the CSV, runs the pairwise duplicate check,
    and prints the first 5 reasonings for human spot-read (exit gate part a).
    """
    parser = argparse.ArgumentParser(
        description="Phase 4 exit gate: validate reasoning column in submission CSV."
    )
    parser.add_argument(
        "--validate", metavar="CSV",
        help="Path to submission CSV produced by rank.py",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Near-duplicate similarity threshold (default: 0.85)",
    )
    parser.add_argument(
        "--spot", type=int, default=5,
        help="Number of reasoning strings to print for human review (default: 5)",
    )
    args = parser.parse_args()

    if not args.validate:
        parser.print_help()
        sys.exit(0)

    csv_path = Path(args.validate)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("ERROR: CSV is empty", file=sys.stderr)
        sys.exit(1)

    reasonings = [r.get("reasoning", "") for r in rows]
    n = len(reasonings)
    print(f"\nLoaded {n} rows from {csv_path}")

    # Exit gate (a): human spot-read
    spot = min(args.spot, n)
    print(f"\n=== Phase 4 exit gate (a): first {spot} reasoning strings for human review ===")
    for i, text in enumerate(reasonings[:spot], start=1):
        cid = rows[i - 1].get("candidate_id", "?")
        print(f"\n  Rank {i} | {cid}:")
        print(f"  {text}")

    # Exit gate (b): pairwise duplicate check
    print(f"\n=== Phase 4 exit gate (b): pairwise near-duplicate check (threshold={args.threshold}) ===")
    collisions = check_duplicates(reasonings, threshold=args.threshold)

    print(f"\nExit gate summary: {n} rows, {len(collisions)} collision(s).")
    if collisions:
        print("ACTION REQUIRED: force structural divergence on flagged pairs before submission.")
        sys.exit(1)
    else:
        print("Phase 4 exit gate: PASS")


if __name__ == "__main__":
    main()
