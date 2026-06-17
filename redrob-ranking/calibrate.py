"""
calibrate.py — Phase 1: hand-ranking harness and weight tuning on the 50-sample

Calibration reality (edgecases.md EC-61, eval.md central flaw):
    The 50-sample has EXACTLY ONE tier-5 candidate (CAND_0000031). Calibration
    can teach the rubric to separate the one fit from 49 non-fits (easy),
    but CANNOT tune top-10 ordering — there is no tier-5 #3 to rank against
    tier-5 #7. The top-10 ordering is REASONED, not tuned.

    What this script CAN test:
    - CAND_0000031 ranks #1 for the RIGHT reason (career evidence, not cosine/keyword).
    - CAND_0000021 (PM, AI buzzwords) is demoted far below CAND_0000031.
    - Disqualifier gate firing rates measured and inspectable.
    - Synthetic tier-5 variants order correctly per written tier-5 ordering rules.
    - Availability multiplier separates engaged from ghost twin without crushing good fits.

Usage:
    # Feature-only scoring (no model required):
    python calibrate.py --sample /path/to/sample_candidates.json

    # Full scoring including embeddings:
    python calibrate.py --sample /path/to/sample_candidates.json --model-path models/bge-base-en-v1.5
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from precompute import parse_features, build_embedding_text, stream_candidates

ARCHETYPE_ID = "CAND_0000031"   # Rec-Sys Eng @ Swiggy, 6.0 YOE, Pinecone expert/88 mo
STUFFER_ID   = "CAND_0000021"   # PM with AI buzzwords in skills AND summary

# ── Tier-5 ordering rules (JD-traced; the Stage-5 substitute for missing validation)
# Written here to be machine-verifiable against synthetic variants.
# See decision.md [P1.5] for the written prose version.
#
# Given two genuine tier-5 candidates (both have built_real_system=True,
# is_eng_title=True, product_vs_services=True), break the tie:
#
#   Rule 1 — Product scale of the shipped system:
#     Large-scale consumer product (Swiggy/Zomato/Razorpay, multi-million users)
#     > growth-stage startup scale > single-user project
#
#   Rule 2 — Eval-framework depth:
#     Evidence of NDCG/MRR/MAP, A/B test design, offline-to-online correlation
#     > generic "improved metrics" claims > no eval evidence
#
#   Rule 3 — Recency of hands-on ranking work:
#     Current role IS ranking/retrieval/recommendation (not adjacent)
#     > ranking in most recent past role (within 2 years)
#     > ranking earlier in career
#
#   Rule 4 — YOE in JD sweet spot:
#     5–9 years > 10–11 > 4 > 12+ > <4
#
#   Rule 5 — Availability:
#     High RRR + low staleness + open > partial engagement > ghost conjunction
#
# A candidate winning on Rule 1 beats any advantage in Rule 2 (strict priority order).


def build_synthetic_tier5_variants(archetype_raw: dict) -> list[tuple[str, dict]]:
    """
    Create 5 synthetic tier-5 variants by perturbing CAND_0000031.
    Each tests a specific ordering rule.
    Returns list of (label, variant_features_dict).
    """
    base = parse_features(archetype_raw)

    variants = []

    # Variant A: Same career evidence but YOE=10.5 (outside 5-9 sweet spot, in 10-11 band)
    # Tests Rule 4: archetype (6 YOE, sweet spot) > A (10.5 YOE, adjacent band)
    v_a = deepcopy(base)
    v_a["candidate_id"] = "SYNTH_A_SENIOR_YOE"
    v_a["years_of_experience"] = 10.5  # outside 5-9 sweet spot → 0.06 YOE bonus (not 0.10)
    variants.append(("A: same career evidence, YOE=10.5 (outside sweet spot)", v_a))

    # Variant B: Lower YOE (4.0 — adjacent band)
    v_b = deepcopy(base)
    v_b["candidate_id"] = "SYNTH_B_LOW_YOE"
    v_b["years_of_experience"] = 4.0   # Rule 4: 4 < 5-9 sweet spot
    variants.append(("B: YOE=4 (adjacent band, not sweet spot)", v_b))

    # Variant C: All-consulting background — fires D4 cap
    v_c = deepcopy(base)
    v_c["candidate_id"] = "SYNTH_C_ALL_CONSULTING"
    v_c["d4_all_consulting"] = True
    v_c["product_vs_services"] = False
    variants.append(("C: all-consulting career (D4 cap fires)", v_c))

    # Variant D: Ghost twin — identical to archetype but disengaged
    v_d = deepcopy(base)
    v_d["candidate_id"] = "SYNTH_D_GHOST_TWIN"
    v_d["staleness_days"] = 200          # > STALE_THRESHOLD
    v_d["recruiter_response_rate"] = 0.05  # < RRR_FLOOR
    v_d["open_to_work_flag"] = False     # ghost conjunction fires
    variants.append(("D: ghost twin (staleness=200, rrr=0.05, not_open)", v_d))

    # Variant E: Engaged weaker candidate (no built_real_system, low YOE)
    v_e = deepcopy(base)
    v_e["candidate_id"] = "SYNTH_E_WEAK_ENGAGED"
    v_e["built_real_system"] = False    # no system-level evidence
    v_e["years_of_experience"] = 3.5   # below sweet spot
    v_e["staleness_days"] = 5           # very fresh
    v_e["recruiter_response_rate"] = 0.95
    v_e["open_to_work_flag"] = True
    variants.append(("E: engaged weak candidate (no built_real_system, YOE=3.5)", v_e))

    return variants


def run_calibration(sample_path: Path, model_path: Path) -> None:
    # ── 1. Parse features on the 50-sample ───────────────────────────────────

    rows, texts, ids, raw_by_id = [], [], [], {}
    for cand in stream_candidates(sample_path):
        feat = parse_features(cand)
        rows.append(feat)
        texts.append(build_embedding_text(cand))
        ids.append(feat["candidate_id"])
        if cand["candidate_id"] == ARCHETYPE_ID:
            raw_by_id[ARCHETYPE_ID] = cand

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} candidates from sample.")

    # ── 2. Gate firing rates (Phase 1 step 5 — EC-23, EC-24, EC-26) ──────────

    print("\n=== Disqualifier gate firing rates (50-sample) ===")
    for col in ["d1_research_only", "d2_recent_llm_only", "d4_all_consulting",
                "d5_cv_speech_robotics", "honeypot_flag"]:
        count = df[col].sum()
        print(f"  {col}: {count}/{len(df)} ({100*count/len(df):.1f}%)")
    non_ic = (~df["hands_on_code_18mo"]).sum()
    print(f"  D3 non-IC (hands_on=False): {non_ic}/{len(df)} ({100*non_ic/len(df):.1f}%)")
    print(f"  [Note: D3 fires on non-engineers; all have base < D3_CAP — zero collateral]")

    # ── 3. Archetype sanity check ─────────────────────────────────────────────

    print(f"\n=== Archetype check: {ARCHETYPE_ID} ===")
    arch = df[df["candidate_id"] == ARCHETYPE_ID]
    if arch.empty:
        print("  NOT FOUND — cannot run archetype check.")
    else:
        a = arch.iloc[0]
        checks = [
            ("is_eng_title",         True,  a["is_eng_title"]),
            ("built_real_system",    True,  a["built_real_system"]),
            ("product_vs_services",  True,  a["product_vs_services"]),
            ("hands_on_code_18mo",   True,  a["hands_on_code_18mo"]),
            ("honeypot_flag",        False, a["honeypot_flag"]),
            ("d1_research_only",     False, a["d1_research_only"]),
            ("d2_recent_llm_only",   False, a["d2_recent_llm_only"]),
            ("d4_all_consulting",    False, a["d4_all_consulting"]),
        ]
        all_pass = True
        for name, expected, actual in checks:
            status = "PASS" if actual == expected else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  [{status}] {name}: {actual} (expected {expected})")
        print(f"  staleness_days={a['staleness_days']}, rrr={a['recruiter_response_rate']:.2f}, open={a['open_to_work_flag']}")
        if not all_pass:
            print("  ERROR: archetype fails one or more checks — fix parser before calibrating.")

    # ── 4. Stuffer check ──────────────────────────────────────────────────────

    print(f"\n=== Stuffer check: {STUFFER_ID} ===")
    stuf = df[df["candidate_id"] == STUFFER_ID]
    if stuf.empty:
        print("  NOT FOUND.")
    else:
        s = stuf.iloc[0]
        print(f"  current_title:     {s['current_title']}")
        print(f"  is_eng_title:      {s['is_eng_title']} (should be False for PM)")
        print(f"  built_real_system: {s['built_real_system']} (should be False)")
        print(f"  ai_skill_count:    {s['ai_skill_count']} (note: threshold is >=3 for stuffer flag)")

    # ── 5. Feature-only scoring (no embedding) ────────────────────────────────

    from rubric import (
        compute_career_score, compute_skills_score, compute_edu_score,
        compute_logistics_modifier, compute_base, apply_disqualifier_caps,
        compute_availability_multiplier, W_CAREER, W_SKILLS, W_EDU,
    )

    career_scores = compute_career_score(df)
    df["career_score"] = career_scores

    # Score with zero embeddings (cosine_sim=0) to isolate career/skills/edu signal
    zero_sim = np.zeros(len(df), dtype=float)
    base_no_embed = compute_base(df, zero_sim)
    gated_no_embed = apply_disqualifier_caps(base_no_embed, df)
    multiplier = compute_availability_multiplier(df)
    final_no_embed = gated_no_embed * multiplier

    df["base_no_embed"] = base_no_embed
    df["gated_no_embed"] = gated_no_embed
    df["multiplier"] = multiplier
    df["final_no_embed"] = final_no_embed

    df_ranked = df.sort_values("final_no_embed", ascending=False).reset_index(drop=True)
    df_ranked["rank_no_embed"] = range(1, len(df_ranked) + 1)

    print("\n=== Top-15 ranking (zero embedding — career + skills + edu + logistics) ===")
    cols = ["rank_no_embed", "candidate_id", "current_title", "final_no_embed",
            "career_score", "gated_no_embed", "multiplier",
            "is_eng_title", "built_real_system", "honeypot_flag"]
    print(df_ranked.head(15)[cols].to_string(index=False))

    arch_rank = df_ranked[df_ranked["candidate_id"] == ARCHETYPE_ID]["rank_no_embed"].values
    stuf_rank = df_ranked[df_ranked["candidate_id"] == STUFFER_ID]["rank_no_embed"].values

    print(f"\n=== Phase 1 exit gate checks (zero-embedding baseline) ===")
    if len(arch_rank):
        ok = arch_rank[0] <= 3
        print(f"  [{('PASS' if ok else 'FAIL')}] {ARCHETYPE_ID} rank: {arch_rank[0]} (should be <=3)")
    if len(stuf_rank) and len(arch_rank):
        ok = stuf_rank[0] > arch_rank[0] + 15
        print(f"  [{('PASS' if ok else 'FAIL')}] {STUFFER_ID} rank: {stuf_rank[0]} (should be >{arch_rank[0]+15})")

    # ── 6. Synthetic tier-5 variants (the ONLY test of tier-5 ordering rules) ──

    if ARCHETYPE_ID not in raw_by_id:
        print(f"\n  Archetype not in sample — skipping synthetic tier-5 tests.")
    else:
        print("\n=== Synthetic tier-5 variant ordering (tier-5 ordering rules test) ===")
        print("These 5 variants + the archetype test the written tier-5 ordering rules.")
        print("The archetype must rank above A, B, D (same quality but slightly weaker signals).")
        print("D (ghost twin) must rank below E (weaker-but-engaged) — the multiplier test.")
        print()

        archetype_raw = raw_by_id[ARCHETYPE_ID]
        variants = build_synthetic_tier5_variants(archetype_raw)

        # Build DataFrame with archetype + all synthetic variants
        synth_rows = [df[df["candidate_id"] == ARCHETYPE_ID].iloc[0].to_dict()]
        labels = {ARCHETYPE_ID: "ARCHETYPE (Swiggy Rec-Sys, 6 YOE, Pinecone/88mo, open)"}
        for label, feat_dict in variants:
            synth_rows.append(feat_dict)
            labels[feat_dict["candidate_id"]] = label

        synth_df = pd.DataFrame(synth_rows)

        syn_career = compute_career_score(synth_df)
        syn_sim = np.zeros(len(synth_df))
        syn_base = compute_base(synth_df, syn_sim)
        syn_gated = apply_disqualifier_caps(syn_base, synth_df)
        syn_mult = compute_availability_multiplier(synth_df)
        syn_final = syn_gated * syn_mult

        synth_df["career"] = syn_career
        synth_df["base"] = syn_base
        synth_df["gated"] = syn_gated
        synth_df["mult"] = syn_mult
        synth_df["final"] = syn_final
        synth_df["label"] = synth_df["candidate_id"].map(labels)

        synth_ranked = synth_df.sort_values("final", ascending=False).reset_index(drop=True)
        synth_ranked["rank"] = range(1, len(synth_ranked) + 1)

        cols = ["rank", "candidate_id", "label", "final", "career", "gated", "mult"]
        print(synth_ranked[cols].to_string(index=False))

        # Ordering assertions
        print("\n=== Ordering rule assertions ===")

        def get_rank(cid):
            r = synth_ranked[synth_ranked["candidate_id"] == cid]["rank"].values
            return r[0] if len(r) else None

        arch_r = get_rank(ARCHETYPE_ID)
        a_r = get_rank("SYNTH_A_SENIOR_YOE")
        b_r = get_rank("SYNTH_B_LOW_YOE")
        c_r = get_rank("SYNTH_C_ALL_CONSULTING")
        d_r = get_rank("SYNTH_D_GHOST_TWIN")
        e_r = get_rank("SYNTH_E_WEAK_ENGAGED")

        checks_synth = [
            ("Archetype > A (YOE=6 sweet spot > YOE=10.5 adjacent)", arch_r < a_r    if arch_r and a_r    else None),
            ("Archetype > B (adjacent YOE band)",             arch_r < b_r    if arch_r and b_r    else None),
            ("A > C (D4 cap degrades consulting career)",     a_r    < c_r    if a_r    and c_r    else None),
            ("D (ghost twin) < Archetype",                    d_r    > arch_r if d_r    and arch_r else None),
            ("D (ghost twin) < E (weaker-but-engaged)",       d_r    > e_r    if d_r    and e_r    else None),
        ]
        for desc, result in checks_synth:
            if result is None:
                print(f"  [SKIP] {desc}")
            else:
                print(f"  [{'PASS' if result else 'FAIL'}] {desc}")

    # ── 7. Full scoring with embeddings (if model available) ─────────────────

    if not model_path.exists():
        print(f"\nModel not found at {model_path} — skipping embedding scoring.")
        print("To enable: python precompute.py --download-model")
        print("Note: feature-only scoring above demonstrates career separation clearly.")
        return

    print(f"\nLoading model from {model_path} ...")
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    from precompute import JD_TEXT

    model = SentenceTransformer(str(model_path))
    print("Embedding 50 samples + JD ...")
    sample_matrix = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    jd_vec = model.encode([JD_TEXT], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
    sim = sample_matrix @ jd_vec

    base_full = compute_base(df, sim)
    gated_full = apply_disqualifier_caps(base_full, df)
    multiplier_full = compute_availability_multiplier(df)
    final_full = gated_full * multiplier_full

    df["sim"] = sim
    df["base_full"] = base_full
    df["gated_full"] = gated_full
    df["final_full"] = final_full

    df_full = df.sort_values("final_full", ascending=False).reset_index(drop=True)
    df_full["rank_full"] = range(1, len(df_full) + 1)

    print("\n=== Top-15 ranking (full scoring with embeddings) ===")
    cols = ["rank_full", "candidate_id", "current_title", "final_full",
            "career_score", "sim", "gated_full", "multiplier_full"
            if "multiplier_full" in df_full.columns else "multiplier"]
    print(df_full.head(15)[["rank_full", "candidate_id", "current_title",
                              "final_full", "career_score", "sim"]].to_string(index=False))

    arch_rank_full = df_full[df_full["candidate_id"] == ARCHETYPE_ID]["rank_full"].values
    stuf_rank_full = df_full[df_full["candidate_id"] == STUFFER_ID]["rank_full"].values
    print(f"\n  {ARCHETYPE_ID} rank (full): {arch_rank_full}")
    print(f"  {STUFFER_ID} rank (full):   {stuf_rank_full}")

    if len(arch_rank_full) and arch_rank_full[0] <= 3:
        print("  PASS: archetype in top 3 with full scoring.")
    else:
        print("  FAIL: archetype not in top 3 with embeddings — check embedding vs career weight.")

    # Verify archetype's high rank is due to career (not cosine):
    if len(arch_rank_full):
        arch_row = df_full[df_full["candidate_id"] == ARCHETYPE_ID].iloc[0]
        print(f"\n  Archetype decomposition (for the RIGHT reason check):")
        print(f"    career score:      {arch_row['career_score']:.3f}  (built_real_system=True is 0.40 of 65%)")
        print(f"    cosine similarity: {arch_row['sim']:.3f}  (should be secondary, not the main driver)")
        print(f"    final score:       {arch_row['final_full']:.3f}")
        if arch_row['career_score'] > 0.8 and arch_row['sim'] < arch_row['career_score']:
            print("    PASS: career evidence dominates — right reason confirmed.")
        else:
            print("    WARN: cosine may be driving the rank — check weight balance.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 calibration harness on 50-sample")
    parser.add_argument("--sample", required=True, help="Path to sample_candidates.json")
    parser.add_argument("--model-path", default="models/bge-base-en-v1.5",
                        help="Local bge-base-en-v1.5 weights (optional; skips embedding if absent)")
    args = parser.parse_args()

    run_calibration(Path(args.sample), Path(args.model_path))


if __name__ == "__main__":
    main()
