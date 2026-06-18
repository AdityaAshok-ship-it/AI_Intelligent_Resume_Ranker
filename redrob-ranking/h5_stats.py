"""H5 impact: per-company impossible-role counts + interaction with H1-H4."""
import sys, io, json
from pathlib import Path
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
from precompute import COMPANY_FOUNDING, FICTIONAL_COMPANIES, parse_date
import pandas as pd

CORPUS = Path(r"A:\side_hustle\IndiaRuns Hackathon") / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
feat = pd.read_parquet(Path(__file__).parent / "artifacts" / "features.parquet")

# parquet-level flag tallies
print("="*70)
print("HONEYPOT FLAG TALLIES (features.parquet)")
print("="*70)
for c in ["h1_flag","h2_flag","h3_flag","h4_flag","h5_flag","honeypot_flag"]:
    print(f"  {c:16} {int(feat[c].sum()):>6,}")
# h5-only (not caught by h1-h4)
h1to4 = feat["h1_flag"]|feat["h2_flag"]|feat["h3_flag"]|feat["h4_flag"]
h5_only = feat["h5_flag"] & ~h1to4
print(f"\n  H5 adds {int(h5_only.sum()):,} NEW honeypots beyond H1-H4 "
      f"(total honeypots 68 -> {int(feat['honeypot_flag'].sum()):,})")

# corpus-level per-company breakdown of impossible roles
print("\n"+"="*70)
print("H5 per-company impossible-role breakdown (corpus scan)")
print("="*70)
co_resumes = Counter()      # resumes with >=1 impossible role at this company
co_roles = Counter()        # total impossible roles at this company
yr_range = {}
n_h5 = 0
with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        hit_companies = set()
        for role in rec.get("career_history", []) or []:
            comp = (role.get("company","") or "").strip().lower()
            if not comp or comp in FICTIONAL_COMPANIES:
                continue
            founded = COMPANY_FOUNDING.get(comp)
            if founded is None:
                continue
            sd = role.get("start_date") or ""
            try:
                y = parse_date(sd).year
            except Exception:
                continue
            if y < founded:
                co_roles[comp] += 1
                hit_companies.add(comp)
                lo, hi = yr_range.get(comp, (9999, 0))
                yr_range[comp] = (min(lo, y), max(hi, y))
        if hit_companies:
            n_h5 += 1
            for c in hit_companies:
                co_resumes[c] += 1

print(f"  Resumes flagged by H5: {n_h5:,}\n")
print(f"  {'company':14} {'founded':>7}  {'resumes':>7}  {'imposs.roles':>12}  {'role-yr range'}")
print("  " + "-"*64)
for comp, n in co_resumes.most_common():
    lo, hi = yr_range[comp]
    print(f"  {comp:14} {COMPANY_FOUNDING[comp]:>7}  {n:>7,}  {co_roles[comp]:>12,}  {lo}-{hi}")
