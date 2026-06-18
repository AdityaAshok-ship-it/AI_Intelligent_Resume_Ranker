"""
Final critical audit — 6 checks:
1. Ideal-fit candidates not in top-100 (the "did we miss anyone?" check)
2. Reasoning template collision (near-duplicate check)
3. Score distribution sanity
4. Gate false-positive probe — sample excluded strong candidates
5. RRR-driven inversions — built=True candidates squeezed below rank 100 by availability
6. Scoring formula invariants
"""
import numpy as np
import pandas as pd
from pathlib import Path
from rubric import (compute_career_score, compute_skills_score,
                    compute_logistics_modifier, compute_base,
                    apply_disqualifier_caps, compute_availability_multiplier)
from detectors import get_honeypot_mask, get_anachronism_mask, get_education_mask

ARTIFACTS = Path("artifacts")
df   = pd.read_parquet(ARTIFACTS / "features.parquet")
mat  = np.load(ARTIFACTS / "candidate_matrix.npy")
jd   = np.load(ARTIFACTS / "jd_vector.npy")
ids  = np.load(ARTIFACTS / "candidate_ids.npy", allow_pickle=True)
sim  = mat @ jd

base = compute_base(df, sim)
gate = apply_disqualifier_caps(base, df)
mult = compute_availability_multiplier(df)

honeypot_mask    = get_honeypot_mask(df)
anachronism_mask = get_anachronism_mask(df)
education_mask   = get_education_mask(df)
surviving        = ~honeypot_mask & ~anachronism_mask & ~education_mask

final = np.where(surviving, gate * mult, -np.inf)
top150_idx = np.argsort(final)[::-1][:150]
for k, idx in enumerate(top150_idx):
    final[idx] += (150 - k) * 1e-9
sorted_all = np.argsort(final)[::-1]

sub         = pd.read_csv("submission.csv")
top100_ids  = set(sub["candidate_id"].tolist())
rank_map    = dict(zip(sub["candidate_id"], sub["rank"]))

# ── CHECK 1: Ideal-fit candidates not in top-100 ─────────────────────────────
print("=" * 70)
print("CHECK 1: Ideal-fit candidates NOT in top-100")
print("  Definition: built=T, eng=T, product=T, hands_on=T, YOE 5-9,")
print("  India, open=T, RRR>=0.80, notice<=30d, no gate exclusion")
print("=" * 70)

ideal_mask = (
    df["built_real_system"].astype(bool) &
    df["is_eng_title"].astype(bool) &
    df["product_vs_services"].astype(bool) &
    df["hands_on_code_18mo"].astype(bool) &
    (df["years_of_experience"] >= 5) &
    (df["years_of_experience"] <= 9) &
    df["is_india_based"].astype(bool) &
    df["open_to_work_flag"].astype(bool) &
    (df["recruiter_response_rate"] >= 0.80) &
    (df["notice_period_days"] <= 30) &
    surviving
)
ideal_df = df[ideal_mask].copy()
ideal_missing = ideal_df[~ideal_df["candidate_id"].isin(top100_ids)]

print(f"Total ideal-fit candidates in pool: {len(ideal_df)}")
print(f"  In top-100:    {len(ideal_df) - len(ideal_missing)}")
print(f"  NOT in top-100: {len(ideal_missing)}")

if len(ideal_missing) > 0:
    print(f"\n  {'CandID':>14}  {'Title':30}  {'YOE':>5}  {'Score':>7}  {'Why low?'}")
    for _, r in ideal_missing.iterrows():
        idx   = list(ids).index(r["candidate_id"])
        sc    = float(final[idx])
        rank_ = int(np.where(sorted_all == idx)[0][0]) + 1
        # find their score components
        row   = df.iloc[[idx]]
        g     = float(apply_disqualifier_caps(np.array([float(compute_base(row, np.array([float(sim[idx])]))[0])]), row)[0])
        m     = float(compute_availability_multiplier(row)[0])
        print(f"  {r['candidate_id']:>14}  {str(r['current_title'])[:30]:30}  "
              f"{float(r['years_of_experience']):>5.1f}  {sc:>7.4f} (rank {rank_:>4})  "
              f"gated={g:.3f} mult={m:.3f}")

# ── CHECK 2: Relaxed ideal (built=T, eng=T, YOE 5-9, surviving, RRR>0.70) ───
print()
print("CHECK 1b: Broader fit (built=T, eng=T, YOE 5-9, RRR>=0.70, surviving) NOT in top-100")
broad_mask = (
    df["built_real_system"].astype(bool) &
    df["is_eng_title"].astype(bool) &
    (df["years_of_experience"] >= 5) &
    (df["years_of_experience"] <= 9) &
    df["open_to_work_flag"].astype(bool) &
    (df["recruiter_response_rate"] >= 0.70) &
    surviving
)
broad_df      = df[broad_mask]
broad_missing = broad_df[~broad_df["candidate_id"].isin(top100_ids)]
print(f"Total: {len(broad_df)} | In top-100: {len(broad_df)-len(broad_missing)} | Missing: {len(broad_missing)}")
if len(broad_missing) > 0:
    print(f"\nTop 10 missing by score:")
    scores_miss = [(float(final[list(ids).index(r["candidate_id"])]), r) for _, r in broad_missing.iterrows()]
    scores_miss.sort(key=lambda x: -x[0])
    for sc, r in scores_miss[:10]:
        idx   = list(ids).index(r["candidate_id"])
        rank_ = int(np.where(sorted_all == idx)[0][0]) + 1
        g = float(apply_disqualifier_caps(np.array([float(compute_base(df.iloc[[idx]], np.array([float(sim[idx])]))[0])]), df.iloc[[idx]])[0])
        m = float(compute_availability_multiplier(df.iloc[[idx]])[0])
        print(f"  rank {rank_:>4}: {r['candidate_id']}  {str(r['current_title'])[:28]:28}  "
              f"YOE={r['years_of_experience']:.1f}  score={sc:.4f}  gated={g:.3f} mult={m:.3f}  "
              f"notice={int(r['notice_period_days'])}d  RRR={r['recruiter_response_rate']:.2f}")

# ── CHECK 3: Reasoning near-duplicate ────────────────────────────────────────
print()
print("=" * 70)
print("CHECK 2: Reasoning near-duplicate (template collision) check")
print("=" * 70)
from difflib import SequenceMatcher
reasons = sub["reasoning"].tolist()
collisions = []
for i in range(len(reasons)):
    for j in range(i+1, len(reasons)):
        ratio = SequenceMatcher(None, reasons[i], reasons[j]).ratio()
        if ratio > 0.85:
            collisions.append((i+1, j+1, ratio))
if collisions:
    print(f"WARNING: {len(collisions)} near-duplicate pairs (similarity > 0.85):")
    for r1, r2, ratio in sorted(collisions, key=lambda x: -x[2])[:10]:
        print(f"  Ranks {r1} & {r2}: similarity={ratio:.3f}")
else:
    print("  OK — no near-duplicate reasoning strings found.")

# ── CHECK 4: Score distribution ───────────────────────────────────────────────
print()
print("=" * 70)
print("CHECK 3: Score distribution")
print("=" * 70)
scores = sub["score"].astype(float).values
percentiles = [10, 25, 50, 75, 90]
for p in percentiles:
    print(f"  p{p:2d}: {np.percentile(scores, p):.4f}")
gaps = np.diff(scores)  # should all be negative (non-increasing)
largest_gap_idx = np.argmin(gaps)
print(f"  Largest single-step drop: rank {largest_gap_idx+1}->{largest_gap_idx+2}, "
      f"gap={abs(gaps[largest_gap_idx]):.4f}")
print(f"  Score is strictly monotone: {all(gaps <= 0)}")
print(f"  All scores distinct: {len(set(scores)) == len(scores)}")

# ── CHECK 5: Gate false-positive probe ───────────────────────────────────────
print()
print("=" * 70)
print("CHECK 4: Anachronism-excluded strong candidates — spot-check top 5 by base score")
print("=" * 70)
anach_strong = df[
    anachronism_mask &
    df["built_real_system"].astype(bool) &
    df["is_eng_title"].astype(bool) &
    (df["years_of_experience"] >= 5)
].copy()
anach_strong["base_score"] = [float(compute_base(df.iloc[[list(ids).index(r["candidate_id"])]],
                               np.array([float(sim[list(ids).index(r["candidate_id"])])]))[0])
                               for _, r in anach_strong.iterrows()]
anach_strong = anach_strong.sort_values("base_score", ascending=False)
print(f"Total anachronism-excluded strong candidates: {len(anach_strong)}")
print(f"\nTop 5 by base score (highest potential impact if wrongly excluded):")
for _, r in anach_strong.head(5).iterrows():
    print(f"  {r['candidate_id']}  {str(r['current_title'])[:30]:30}  "
          f"YOE={r['years_of_experience']:.1f}  base={r['base_score']:.4f}")
    # Find which skill triggered it
    if "anachronism_skill" in r.index:
        print(f"    triggered by: {r['anachronism_skill']}")

# ── CHECK 6: Top-100 built=True count and YOE health ─────────────────────────
print()
print("=" * 70)
print("CHECK 5: Final top-100 composition health")
print("=" * 70)
top100_df = df[df["candidate_id"].isin(top100_ids)]
print(f"  built_real_system=True : {top100_df['built_real_system'].sum()} / 100  (target: 100)")
print(f"  is_eng_title=True      : {top100_df['is_eng_title'].sum()} / 100")
print(f"  product_vs_services=T  : {top100_df['product_vs_services'].sum()} / 100")
print(f"  India-based            : {top100_df['is_india_based'].sum()} / 100")
print(f"  open_to_work=True      : {top100_df['open_to_work_flag'].sum()} / 100")
print(f"  YOE sweet-spot (5-9)   : {((top100_df['years_of_experience']>=5)&(top100_df['years_of_experience']<=9)).sum()} / 100")
print(f"  YOE < 5                : {(top100_df['years_of_experience']<5).sum()} / 100")
print(f"  notice <=30d           : {(top100_df['notice_period_days']<=30).sum()} / 100")
print(f"  notice 31-60d          : {((top100_df['notice_period_days']>=31)&(top100_df['notice_period_days']<=60)).sum()} / 100")
print(f"  notice >=120d          : {(top100_df['notice_period_days']>=120).sum()} / 100")
print(f"  RRR >= 0.70            : {(top100_df['recruiter_response_rate']>=0.70).sum()} / 100")
print(f"  RRR < 0.50             : {(top100_df['recruiter_response_rate']<0.50).sum()} / 100")
print(f"  Score spread           : {scores.max():.4f} – {scores.min():.4f} = {scores.max()-scores.min():.4f}")
