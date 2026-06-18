"""Enumerate distinct skill names + degree types across the full corpus (for gate design)."""
import json, re, sys, io
from collections import Counter, defaultdict
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CORPUS = Path(r"A:\side_hustle\IndiaRuns Hackathon") / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"

skill_count = Counter()
skill_maxdur = defaultdict(int)
degree_count = Counter()
field_count = Counter()

with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for s in rec.get("skills", []) or []:
            nm = (s.get("name") or "").strip()
            if nm:
                skill_count[nm] += 1
                d = s.get("duration_months")
                if isinstance(d, int):
                    skill_maxdur[nm] = max(skill_maxdur[nm], d)
        for e in rec.get("education", []) or []:
            dg = (e.get("degree") or "").strip()
            if dg:
                degree_count[dg] += 1
            fld = (e.get("field_of_study") or "").strip()
            if fld:
                field_count[fld] += 1

print(f"DISTINCT SKILLS: {len(skill_count)}")
print("="*70)
for nm, c in skill_count.most_common():
    print(f"  {nm:32} n={c:>6}  max_dur={skill_maxdur[nm]:>3}mo")

print(f"\nDISTINCT DEGREES: {len(degree_count)}")
print("="*70)
for dg, c in degree_count.most_common():
    print(f"  {dg:24} n={c:>6}")

print(f"\nDISTINCT FIELDS OF STUDY: {len(field_count)}")
print("="*70)
for fld, c in field_count.most_common(40):
    print(f"  {fld:40} n={c:>6}")
