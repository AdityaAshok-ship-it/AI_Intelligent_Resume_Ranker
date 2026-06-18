"""
Critical submission audit:
1. Ranks 101-150 — who got squeezed out; any stronger than rank 91-100?
2. Gate exclusion check — any strong candidates wrongly excluded?
3. Submission internal consistency — score monotonic, no dup IDs, reasoning flags
4. Distribution checks — notice, YOE, title, built_real_system across top-100
"""
import numpy as np
import pandas as pd
from pathlib import Path
from rubric import compute_career_score, compute_skills_score, compute_logistics_modifier, compute_base, apply_disqualifier_caps, compute_availability_multiplier
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

# Apply distinct-float (same as rank.py)
top150_idx = np.argsort(final)[::-1][:150]
N = len(top150_idx)
for k, idx in enumerate(top150_idx):
    final[idx] += (N - k) * 1e-9

sorted_idx = np.argsort(final)[::-1]

# ── 1. Ranks 101–130 snapshot ─────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Ranks 101–130 (squeezed-out candidates)")
print("=" * 70)
sub = pd.read_csv("submission.csv")
top100_ids = set(sub["candidate_id"].tolist())

print(f"{'Rank':>5}  {'CandID':>14}  {'Title':30}  {'YOE':>5}  {'Built':>5}  {'Eng':>5}  {'Score':>7}  {'Notice':>7}  {'RRR':>5}")
for rank, idx in enumerate(sorted_idx[100:130], 101):
    cid = str(ids[idx])
    r   = df.iloc[idx]
    sc  = float(final[idx])
    print(f"{rank:>5}  {cid:>14}  {str(r['current_title'])[:30]:30}  {float(r['years_of_experience']):>5.1f}  "
          f"{'Y' if r['built_real_system'] else 'N':>5}  {'Y' if r['is_eng_title'] else 'N':>5}  "
          f"{sc:>7.4f}  {int(r['notice_period_days']):>5}d  {float(r['recruiter_response_rate']):>5.2f}")

# ── 2. Gate exclusion — strong candidates wrongly caught? ─────────────────────
print()
print("=" * 70)
print("SECTION 2: Gate exclusions — strong candidates (built=True, eng=True, YOE>=5)")
print("=" * 70)

strong_excl = df[
    (honeypot_mask | anachronism_mask | education_mask) &
    df["built_real_system"].astype(bool) &
    df["is_eng_title"].astype(bool) &
    (df["years_of_experience"] >= 5)
].copy()
strong_excl["gate"] = (
    np.where(honeypot_mask & df["built_real_system"].astype(bool) & df["is_eng_title"].astype(bool) & (df["years_of_experience"]>=5), "HONEYPOT", "") +
    np.where(anachronism_mask & df["built_real_system"].astype(bool) & df["is_eng_title"].astype(bool) & (df["years_of_experience"]>=5), "ANACHRONISM", "") +
    np.where(education_mask & df["built_real_system"].astype(bool) & df["is_eng_title"].astype(bool) & (df["years_of_experience"]>=5), "EDUCATION", "")
)[strong_excl.index]

print(f"Total strong candidates excluded: {len(strong_excl)}")
print(f"  by honeypot:    {int((honeypot_mask & df['built_real_system'].astype(bool) & df['is_eng_title'].astype(bool) & (df['years_of_experience']>=5)).sum())}")
print(f"  by anachronism: {int((anachronism_mask & df['built_real_system'].astype(bool) & df['is_eng_title'].astype(bool) & (df['years_of_experience']>=5)).sum())}")
print(f"  by education:   {int((education_mask & df['built_real_system'].astype(bool) & df['is_eng_title'].astype(bool) & (df['years_of_experience']>=5)).sum())}")

# Sample 10 honeypot-excluded strong candidates for spot-check
hp_strong = df[
    honeypot_mask &
    df["built_real_system"].astype(bool) &
    df["is_eng_title"].astype(bool) &
    (df["years_of_experience"] >= 5)
]
if len(hp_strong) > 0:
    print(f"\nSample honeypot-excluded strong candidates (up to 10):")
    print(f"{'CandID':>14}  {'Title':30}  {'YOE':>5}  {'HFlag':>6}")
    for _, r in hp_strong.head(10).iterrows():
        print(f"  {r['candidate_id']:>14}  {str(r['current_title'])[:30]:30}  {float(r['years_of_experience']):>5.1f}  {r.get('honeypot_flag', '?')}")

# ── 3. Submission internal consistency ────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 3: Submission internal consistency")
print("=" * 70)
sub = pd.read_csv("submission.csv")
sub["score"] = sub["score"].astype(float)

print(f"Rows: {len(sub)}  (expected 100)")
print(f"Dup candidate_ids: {sub['candidate_id'].duplicated().sum()}")
print(f"Dup ranks: {sub['rank'].duplicated().sum()}")
print(f"Score strictly non-increasing: {all(sub['score'].iloc[i] >= sub['score'].iloc[i+1] for i in range(len(sub)-1))}")
print(f"Score range: {sub['score'].min():.4f} – {sub['score'].max():.4f}")
print(f"Score spread (max-min): {sub['score'].max() - sub['score'].min():.4f}")

# Check reasoning length distribution
sub["reason_len"] = sub["reasoning"].str.len()
print(f"Reasoning length: min={sub['reason_len'].min()}, median={int(sub['reason_len'].median())}, max={sub['reason_len'].max()}")

# ── 4. Distribution across top-100 ───────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 4: Top-100 distribution checks")
print("=" * 70)

top100_df = df[df["candidate_id"].isin(top100_ids)].copy()
print(f"built_real_system=True : {top100_df['built_real_system'].sum()} / 100")
print(f"is_eng_title=True      : {top100_df['is_eng_title'].sum()} / 100")
print(f"India-based            : {top100_df['is_india_based'].sum()} / 100")
print(f"open_to_work=True      : {top100_df['open_to_work_flag'].sum()} / 100")

print(f"\nNotice period distribution:")
for lo, hi, label in [(0,30,"<=30d"), (31,60,"31-60d"), (61,119,"61-119d"), (120,180,">=120d")]:
    n = ((top100_df["notice_period_days"] >= lo) & (top100_df["notice_period_days"] <= hi)).sum()
    print(f"  {label}: {n}")

print(f"\nYOE distribution:")
for lo, hi, label in [(0,4,"<5 (below sweet-spot)"), (5,9,"5-9 (sweet-spot)"), (10,99,"10+ (over)")]:
    n = ((top100_df["years_of_experience"] >= lo) & (top100_df["years_of_experience"] <= hi)).sum()
    print(f"  {label}: {n}")

print(f"\nRRR distribution (recruiter response rate):")
for lo, hi, label in [(0,0.3,"<0.30 (low)"), (0.3,0.6,"0.30-0.60 (medium)"), (0.6,1.0,">0.60 (good)")]:
    n = ((top100_df["recruiter_response_rate"] >= lo) & (top100_df["recruiter_response_rate"] <= hi)).sum()
    print(f"  {label}: {n}")

# ── 5. Weak tail — candidates with no-ship in top-100 ─────────────────────────
print()
print("=" * 70)
print("SECTION 5: No-ship candidates in top-100 (built_real_system=False)")
print("=" * 70)
no_ship = top100_df[~top100_df["built_real_system"].astype(bool)]
print(f"Count: {len(no_ship)}")
for _, r in no_ship.iterrows():
    idx = list(ids).index(r["candidate_id"])
    rank_pos = sub[sub["candidate_id"] == r["candidate_id"]]["rank"].values[0]
    print(f"  Rank {rank_pos:>3}: {r['candidate_id']}  {str(r['current_title'])[:30]}  YOE={r['years_of_experience']:.1f}  RRR={r['recruiter_response_rate']:.2f}  notice={int(r['notice_period_days'])}d")

# ── 6. Score boundary: rank 100 vs rank 101 gap ──────────────────────────────
print()
print("=" * 70)
print("SECTION 6: Boundary — rank 100 vs ranks 101-105")
print("=" * 70)
for rank, idx in enumerate(sorted_idx[98:106], 99):
    cid = str(ids[idx])
    r = df.iloc[idx]
    sc = float(final[idx])
    in_top = "IN" if cid in top100_ids else "OUT"
    print(f"  Rank {rank:>3} [{in_top}]: {cid}  {str(r['current_title'])[:28]:28}  score={sc:.5f}  built={r['built_real_system']}  eng={r['is_eng_title']}")
