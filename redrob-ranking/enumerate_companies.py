"""Enumerate distinct company names across the full corpus (for H5 founding gate)."""
import json, sys, io
from collections import Counter, defaultdict
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CORPUS = Path(r"A:\side_hustle\IndiaRuns Hackathon") / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"

comp_count = Counter()
comp_min_start = defaultdict(lambda: "9999")  # earliest role start_date seen at each company
n_records = 0

with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n_records += 1
        # current_company
        cc = (rec.get("profile", {}).get("current_company") or "").strip()
        if cc:
            comp_count[cc] += 1
        # career_history companies
        for r in rec.get("career_history", []) or []:
            c = (r.get("company") or "").strip()
            if not c:
                continue
            comp_count[c] += 1
            sd = r.get("start_date") or ""
            if sd and sd < comp_min_start[c]:
                comp_min_start[c] = sd

print(f"Records: {n_records:,}")
print(f"DISTINCT company names: {len(comp_count)}")
print("="*78)
print(f"{'company':40} {'count':>8}  {'earliest_role_start'}")
print("-"*78)
for name, c in comp_count.most_common():
    print(f"  {name:38} {c:>8,}  {comp_min_start.get(name,'')}")
