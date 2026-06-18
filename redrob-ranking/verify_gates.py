"""Verify the two new gates: enforcement in top-100 + blast-radius breakdown."""
import csv, sys, io
import pandas as pd
from pathlib import Path
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

D = Path(r"A:\side_hustle\IndiaRuns Hackathon\redrob-ranking")
feat = pd.read_parquet(D / "artifacts" / "features.parquet").set_index("candidate_id")

def load_ids(p):
    with open(p, newline="", encoding="utf-8") as f:
        return [r["candidate_id"] for r in csv.DictReader(f)]

new_top = load_ids(D / "submission.csv")
old_top = load_ids(D / "submission_run.csv")

print("="*72)
print("FULL-CORPUS gate blast radius")
print("="*72)
print(f"  honeypot_flag         : {int(feat['honeypot_flag'].sum()):>6,}")
print(f"  skill_anachronism_flag: {int(feat['skill_anachronism_flag'].sum()):>6,}")
print(f"  education_anomaly_flag : {int(feat['education_anomaly_flag'].sum()):>6,}")

# education reason breakdown
ed_cat = Counter()
for d in feat.loc[feat["education_anomaly_flag"], "education_anomaly_detail"]:
    kinds = set()
    for part in str(d).split(";"):
        part = part.strip()
        if part.startswith("reversal"): kinds.add("reversal")
        elif part.startswith("overlap"): kinds.add("overlap")
        elif part.startswith("neg-span"): kinds.add("neg-span")
    for k in kinds:
        ed_cat[k] += 1
print("\n  education breakdown (candidates with >=1 of each kind):")
for k, v in ed_cat.most_common():
    print(f"     {k:10} {v:>6,}")

# skill anachronism: which techs drive it
sk_cat = Counter()
for d in feat.loc[feat["skill_anachronism_flag"], "skill_anachronism_detail"]:
    for part in str(d).split(";"):
        part = part.strip()
        if part:
            sk_cat[part.split(":")[0]] += 1
print("\n  skill-anachronism breakdown (by technology):")
for k, v in sk_cat.most_common():
    print(f"     {k:26} {v:>5,}")

print("\n"+"="*72)
print("ENFORCEMENT: does any flagged record survive into the NEW top-100?")
print("="*72)
leaked = [c for c in new_top if bool(feat.loc[c, "honeypot_flag"]) or
          bool(feat.loc[c, "skill_anachronism_flag"]) or
          bool(feat.loc[c, "education_anomaly_flag"])]
print(f"  leaked flagged records in new top-100: {len(leaked)}  {'<-- BUG!' if leaked else '(clean)'}")
if leaked:
    for c in leaked[:20]:
        print(f"    {c}: honey={bool(feat.loc[c,'honeypot_flag'])} "
              f"anach={bool(feat.loc[c,'skill_anachronism_flag'])} "
              f"edu={bool(feat.loc[c,'education_anomaly_flag'])}")

print("\n"+"="*72)
print("SHIFT: what happened to the OLD top-100 under the new gates?")
print("="*72)
gone = [c for c in old_top if c not in set(new_top)]
why = Counter()
for c in gone:
    if bool(feat.loc[c, "skill_anachronism_flag"]): why["skill-anachronism"] += 1
    if bool(feat.loc[c, "education_anomaly_flag"]): why["education"] += 1
print(f"  old top-100 records dropped from new top-100: {len(gone)}")
print(f"    of which skill-anachronism: {why['skill-anachronism']}")
print(f"    of which education-anomaly: {why['education']}")
kept = len(set(old_top) & set(new_top))
print(f"  old top-100 still in new top-100: {kept}")

# a few examples of dropped, with detail
print("\n  examples of dropped old-top records (with reason):")
shown = 0
for c in old_top:
    if c in gone and shown < 12:
        a = str(feat.loc[c, "skill_anachronism_detail"])
        e = str(feat.loc[c, "education_anomaly_detail"])
        r = []
        if a and a != "nan": r.append(f"skill[{a}]")
        if e and e != "nan": r.append(f"edu[{e}]")
        print(f"    {c}: {' '.join(r)[:120]}")
        shown += 1
