"""Third pass: education-timeline integrity across top-100 (pipeline never checks this)."""
import json, re, csv, sys, io
from datetime import date
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(r"A:\side_hustle\IndiaRuns Hackathon")
CORPUS = ROOT / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
SUB = ROOT / "redrob-ranking" / "submission_run.csv"

DEGREE_RANK = {  # higher = more advanced
    "phd":4, "ph.d":4, "doctor":4, "m.tech":3, "mtech":3, "m.s":3, "ms":3,
    "m.sc":3, "msc":3, "master":3, "mba":3, "m.e":3, "me ":3,
    "b.tech":1, "btech":1, "b.e":1, "be ":1, "b.sc":1, "bsc":1, "bachelor":1,
}
def deg_rank(d):
    dl = (d or "").lower()
    for k,v in DEGREE_RANK.items():
        if k in dl:
            return v
    return 0

top = {}
with open(SUB, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        top[row["candidate_id"]] = (int(row["rank"]), float(row["score"]))
want = set(top)

raw = {}
with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        m = re.search(r'"candidate_id"\s*:\s*"(CAND_\d+)"', line)
        if m and m.group(1) in want:
            raw[m.group(1)] = json.loads(line)
            if len(raw) == len(want):
                break

findings = {}
for cid, rec in raw.items():
    edu = rec.get("education", []) or []
    fs = []
    # 1. degree-order reversal: a higher degree ENDS before a lower degree STARTS
    for i,a in enumerate(edu):
        for j,b in enumerate(edu):
            if i==j: continue
            ra, rb = deg_rank(a.get("degree")), deg_rank(b.get("degree"))
            ay, by = a.get("end_year"), b.get("start_year")
            if ra>rb>0 and isinstance(ay,int) and isinstance(by,int) and ay < by:
                fs.append(f"REVERSED: {a.get('degree')} ends {ay} but lower {b.get('degree')} starts {by}")
    # 2. negative / zero / absurd education span
    for e in edu:
        sy,ey = e.get("start_year"), e.get("end_year")
        if isinstance(sy,int) and isinstance(ey,int):
            if ey < sy:
                fs.append(f"NEG SPAN: {e.get('degree')} {sy}-{ey} (ends before starts)")
            elif ey - sy > 8:
                fs.append(f"LONG SPAN: {e.get('degree')} {sy}-{ey} ({ey-sy}y)")
    # 3. AI-named undergrad before it plausibly existed (~2019 in India)
    for e in edu:
        fld = (e.get("field_of_study","") or "").lower()
        deg = (e.get("degree","") or "").lower()
        sy = e.get("start_year")
        if "artificial intelligence" in fld and deg_rank(deg)==1 and isinstance(sy,int) and sy < 2018:
            fs.append(f"ANACHRON DEGREE: '{e.get('degree')} {e.get('field_of_study')}' starting {sy}")
    # 4. overlapping full-time degrees (two different institutions same years, big overlap)
    spans = [(e.get("start_year"),e.get("end_year"),e.get("institution"),e.get("degree")) for e in edu
             if isinstance(e.get("start_year"),int) and isinstance(e.get("end_year"),int)]
    spans.sort()
    for i in range(len(spans)-1):
        s1,e1,i1,d1 = spans[i]; s2,e2,i2,d2 = spans[i+1]
        if s2 < e1 and (min(e1,e2)-s2) >= 2 and i1!=i2:
            fs.append(f"OVERLAP DEGREES: {d1}@{i1}({s1}-{e1}) vs {d2}@{i2}({s2}-{e2})")
    if fs:
        findings[cid] = fs

print("="*78)
print(f"EDUCATION-TIMELINE problems in top-100: {len(findings)} candidates")
print("(degree-order reversals, impossible spans, anachronistic degrees — NONE of")
print(" which H1-H4 or the pipeline's audit can see; education is never timeline-checked)")
print("="*78)
for cid in sorted(findings, key=lambda c: top[c][0]):
    rk,sc = top[cid]
    print(f"\n  #{rk:>3} {cid} score={sc:.4f}")
    for f in findings[cid]:
        print(f"       {f}")

# breakdown
from collections import Counter
cat = Counter()
for fs in findings.values():
    for f in fs:
        cat[f.split(":")[0]] += 1
print("\n"+"="*78)
print("Category tally:")
for k,v in cat.most_common():
    print(f"  {k:18} {v}")
print("="*78)
n_rev = sum(1 for fs in findings.values() if any(f.startswith("REVERSED") for f in fs))
print(f"\nCandidates with an outright IMPOSSIBLE degree-order reversal: {n_rev}/100")
