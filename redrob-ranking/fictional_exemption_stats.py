"""Measure the fictional-company exemption blast radius for the proposed H5 gate."""
import json, sys, io
from collections import Counter
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CORPUS = Path(r"A:\side_hustle\IndiaRuns Hackathon") / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"

FICTIONAL = {
    "wayne enterprises", "initech", "pied piper", "acme corp",
    "globex inc", "hooli", "dunder mifflin", "stark industries",
}

n = 0
with_fiction = 0       # resumes containing >=1 fictional company (anywhere)
fiction_per_resume = Counter()
only_fiction = 0       # resumes whose companies are ALL fictional
has_real = 0

with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n += 1
        comps = set()
        for r in rec.get("career_history", []) or []:
            c = (r.get("company") or "").strip().lower()
            if c:
                comps.add(c)
        cc = (rec.get("profile", {}).get("current_company") or "").strip().lower()
        if cc:
            comps.add(cc)
        fic = comps & FICTIONAL
        real = comps - FICTIONAL
        if fic:
            with_fiction += 1
        fiction_per_resume[len(fic)] += 1
        if real:
            has_real += 1
        if comps and not real:
            only_fiction += 1

print(f"Total resumes: {n:,}")
print(f"Resumes with >=1 FICTIONAL company (EXEMPTED under whole-resume rule): {with_fiction:,} ({100*with_fiction/n:.1f}%)")
print(f"Resumes with >=1 real company: {has_real:,} ({100*has_real/n:.1f}%)")
print(f"Resumes with ONLY fictional companies (no real co at all): {only_fiction:,} ({100*only_fiction/n:.1f}%)")
print(f"\n=> Under the whole-resume exemption, H5 can only act on {n - with_fiction:,} resumes ({100*(n-with_fiction)/n:.1f}%).")
print("\nDistribution of #fictional companies per resume:")
for k in sorted(fiction_per_resume):
    print(f"  {k} fictional co: {fiction_per_resume[k]:,}")
