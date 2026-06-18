"""Compact top-100 view for review: rank, candidate, score, YOE, staleness, rrr, open, tier, title."""
import csv, sys, io
import pandas as pd
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = Path(__file__).parent
feat = pd.read_parquet(D / "artifacts" / "features.parquet").set_index("candidate_id")

rows = []
with open(D / "submission.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append((int(r["rank"]), r["candidate_id"], float(r["score"])))

TIER = {"tier_1":"T1","tier_2":"T2","tier_3":"T3","tier_4":"T4","unknown":"--"}
print(f"{'rk':>3}  {'candidate':13} {'score':>7} {'YOE':>4} {'stale':>5} {'rrr':>4} {'op':>2} {'tier':>4}  title")
print("-"*96)
for rank, cid, score in rows:
    r = feat.loc[cid]
    yoe = float(r["years_of_experience"]); stale = int(r["staleness_days"])
    rrr = float(r["recruiter_response_rate"]); op = "Y" if bool(r["open_to_work_flag"]) else "n"
    tier = TIER.get(str(r["top_edu_tier"]), "--")
    title = str(r["current_title"])[:34]
    print(f"{rank:>3}  {cid:13} {score:>7.4f} {yoe:>4.1f} {stale:>4}d {rrr:>4.2f} {op:>2} {tier:>4}  {title}")

# quick distribution summary
import numpy as np
y = np.array([float(feat.loc[c]['years_of_experience']) for _,c,_ in rows])
s = np.array([int(feat.loc[c]['staleness_days']) for _,c,_ in rows])
print("-"*96)
print(f"YOE: min {y.min():.1f}  median {np.median(y):.1f}  max {y.max():.1f}  | "
      f"in sweet-spot 5-9: {int(((y>=5)&(y<=9)).sum())}/100  | below 5: {int((y<5).sum())}  above 9: {int((y>9).sum())}")
print(f"Staleness: min {s.min()}d  median {int(np.median(s))}d  max {s.max()}d  | "
      f"<=15d: {int((s<=15).sum())}  16-45d: {int(((s>15)&(s<=45)).sum())}  46-90d: {int(((s>45)&(s<=90)).sum())}  >90d: {int((s>90).sum())}")
