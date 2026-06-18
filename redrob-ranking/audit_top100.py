"""
audit_top100.py — INDEPENDENT adversarial audit of submission_run.csv top-100.

Does NOT trust precompute's flags. Re-derives H1-H4 from the raw corpus, then runs
a battery of discrepancy checks the shipped pipeline never performs. Goal: answer
"are you SURE there are no honeypots in the top 100?" with evidence, not the
pipeline's self-audit.
"""
import json, re, csv, sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(r"A:\side_hustle\IndiaRuns Hackathon")
CORPUS = ROOT / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
SUB = ROOT / "redrob-ranking" / "submission_run.csv"

REFERENCE_DATE = date(2026, 5, 27)

FOUNDING_YEAR_MAP = {
    "swiggy":2014,"razorpay":2014,"paytm":2010,"flipkart":2007,"zomato":2008,
    "ola":2010,"meesho":2015,"cred":2018,"zepto":2021,"sharechat":2015,
    "dream11":2008,"groww":2016,"upstox":2011,"smallcase":2015,"dunzo":2015,
    "udaan":2016,"nykaa":2012,
}

# Famous fictional employers — the residual the pipeline ADMITS it cannot see (EC-33).
FICTIONAL = {
    "hooli","pied piper","aviato","raviga","initech","initrode","stark",
    "stark industries","wayne","wayne enterprises","oscorp","lexcorp","umbrella",
    "globex","soylent","cyberdyne","tyrell","weyland","weyland-yutani",
    "massive dynamic","dunder mifflin","dunder-mifflin","sabre","vandelay",
    "kramerica","wonka","acme","gringotts","prestige worldwide","bluth",
    "wernham hogg","nakatomi","buy n large","aperture","aperture science",
    "black mesa","virtucon","gekko","krusty krab","monsters inc","duff",
    "wernham","skynet","encom","los pollos","gizmonic","pizza planet",
    "abstergo","vault-tec","wallace corporation","rekall","omni consumer",
    "ocp","spectre","kobayashi",
}

def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()

def months_between(start, end):
    return max(0, (end.year - start.year)*12 + (end.month - start.month))

def safe_date(s):
    try:
        return parse_date(s)
    except Exception:
        return None

# ── load top-100 ──
top = {}
with open(SUB, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        top[row["candidate_id"]] = {
            "rank": int(row["rank"]),
            "score": float(row["score"]),
            "reasoning": row["reasoning"],
        }
want = set(top)
print(f"Loaded {len(want)} top candidates from {SUB.name}")

# ── pull raw records ──
raw = {}
with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # cheap pre-filter before full json parse
        if '"candidate_id"' not in line:
            continue
        m = re.search(r'"candidate_id"\s*:\s*"(CAND_\d+)"', line)
        if not m or m.group(1) not in want:
            continue
        raw[m.group(1)] = json.loads(line)
        if len(raw) == len(want):
            break
print(f"Matched {len(raw)}/{len(want)} raw records from corpus\n")
missing = want - set(raw)
if missing:
    print(f"!! {len(missing)} top-100 IDs NOT FOUND in corpus: {sorted(missing)[:10]}")

# ── per-candidate audit ──
findings = {cid: [] for cid in raw}

def add(cid, sev, code, msg):
    findings[cid].append((sev, code, msg))

for cid, rec in raw.items():
    prof = rec.get("profile", {})
    career = rec.get("career_history", []) or []
    edu = rec.get("education", []) or []
    skills = rec.get("skills", []) or []
    sig = rec.get("redrob_signals", {}) or {}
    yoe = float(prof.get("years_of_experience", 0))
    yoe_m = int(round(yoe*12))

    # ===== H1: role claims more months than elapsed since its start (+3) =====
    for r in career:
        sd = safe_date(r.get("start_date",""))
        if not sd:
            continue
        ed = safe_date(r.get("end_date") or "") or REFERENCE_DATE
        actual = months_between(sd, ed)
        claimed = int(r.get("duration_months",0))
        if claimed > actual + 3:
            add(cid,"HONEYPOT","H1",f"role '{r.get('title','')}'@{r.get('company','')} claims {claimed}mo but only {actual}mo elapsed (start {r.get('start_date')})")

    # ===== H2: Σ tenure > YOE*12 + 30 =====
    tot = sum(int(r.get("duration_months",0)) for r in career)
    if tot > yoe_m + 30:
        add(cid,"HONEYPOT","H2",f"Σ role tenure {tot}mo > YOE*12+30 = {yoe_m+30}mo (YOE={yoe})")

    # ===== H3: YOE*12 > career span + 18 =====
    starts = [safe_date(r.get("start_date","")) for r in career]
    starts = [s for s in starts if s]
    span = months_between(min(starts), REFERENCE_DATE) if starts else 0
    if yoe_m > span + 18:
        add(cid,"HONEYPOT","H3",f"YOE*12={yoe_m}mo > career span {span}mo+18 (earliest start {min(starts) if starts else 'NA'})")

    # ===== H4: expert/advanced skill with explicit duration 0 =====
    for s in skills:
        dur = s.get("duration_months")
        if dur == 0 and s.get("proficiency") in {"advanced","expert"}:
            add(cid,"HONEYPOT","H4",f"{s.get('proficiency')} skill '{s.get('name')}' with duration_months=0")

    # ===== Founding-date anomaly (real companies) =====
    for r in career:
        comp = (r.get("company","") or "").lower()
        sd = safe_date(r.get("start_date",""))
        if not comp or not sd:
            continue
        toks = set(re.split(r"[^a-z0-9]", comp))
        for known, yr in FOUNDING_YEAR_MAP.items():
            hit = (known in toks) if len(known)<=5 else (known in comp)
            if hit and sd.year < yr:
                add(cid,"HONEYPOT","FOUND",f"role at '{r.get('company')}' starts {sd.year} < founding {yr}")

    # ===== Fictional employer (the admitted residual) =====
    for r in career:
        comp = (r.get("company","") or "").lower().strip()
        toks = set(re.split(r"[^a-z0-9]", comp))
        for fic in FICTIONAL:
            hit = (fic in toks) if len(fic)<=5 else (fic in comp)
            if hit:
                add(cid,"HONEYPOT","FICT",f"fictional employer: '{r.get('company')}'")
                break

    # ===== date sanity: end<start, future dates =====
    for r in career:
        sd = safe_date(r.get("start_date",""))
        ed = safe_date(r.get("end_date") or "")
        if sd and ed and ed < sd:
            add(cid,"HARD","DATE_REV",f"role '{r.get('company')}' end {ed} < start {sd}")
        if sd and sd > REFERENCE_DATE:
            add(cid,"HARD","DATE_FUT",f"role '{r.get('company')}' starts in future {sd}")
        if ed and ed > REFERENCE_DATE:
            add(cid,"SOFT","END_FUT",f"role '{r.get('company')}' ends in future {ed}")

    # ===== is_current vs end_date consistency =====
    for r in career:
        isc = r.get("is_current")
        ed = r.get("end_date")
        if isc and ed:
            add(cid,"SOFT","CUR_END",f"is_current=True but end_date={ed} ('{r.get('company')}')")
        if (not isc) and (ed is None):
            add(cid,"SOFT","NOCUR_NOEND",f"is_current=False but end_date=null ('{r.get('company')}')")
    n_current = sum(1 for r in career if r.get("is_current"))
    if n_current > 1:
        add(cid,"SOFT","MULTI_CUR",f"{n_current} roles flagged is_current=True")

    # ===== per-role duration mismatch in BOTH directions (beyond H1) =====
    for r in career:
        sd = safe_date(r.get("start_date",""))
        if not sd:
            continue
        ed = safe_date(r.get("end_date") or "") or REFERENCE_DATE
        actual = months_between(sd, ed)
        claimed = int(r.get("duration_months",0))
        if actual - claimed > 12 and claimed>0:
            add(cid,"SOFT","DUR_UNDER",f"role '{r.get('company')}' claims {claimed}mo but {actual}mo elapsed (under-claim {actual-claimed}mo)")

    # ===== overlapping full-time roles =====
    intervals = []
    for r in career:
        sd = safe_date(r.get("start_date",""))
        ed = safe_date(r.get("end_date") or "") or REFERENCE_DATE
        if sd:
            intervals.append((sd, ed, r.get("company","")))
    intervals.sort()
    for i in range(len(intervals)-1):
        s1,e1,c1 = intervals[i]
        s2,e2,c2 = intervals[i+1]
        ov = months_between(s2, min(e1,e2))
        if s2 < e1 and ov >= 6:
            add(cid,"SOFT","OVERLAP",f"'{c1}' ({s1}..{e1}) overlaps '{c2}' ({s2}..{e2}) by ~{ov}mo")

    # ===== skill older than career =====
    for s in skills:
        dur = s.get("duration_months")
        if isinstance(dur,int) and dur > span + 18 and span>0:
            add(cid,"SOFT","SKILL_GT_CAREER",f"skill '{s.get('name')}' {dur}mo > career span {span}mo")

    # ===== age / graduation plausibility =====
    estarts = [e.get("start_year") for e in edu if isinstance(e.get("start_year"),int)]
    eends = [e.get("end_year") for e in edu if isinstance(e.get("end_year"),int)]
    if estarts and starts:
        first_edu = min(estarts)
        first_job = min(starts).year
        if first_job < first_edu - 1:
            add(cid,"SOFT","JOB_BEFORE_EDU",f"first job {first_job} predates first education start {first_edu}")
        # implied age at first job assuming undergrad start ~ age 18
        implied_age_now = (REFERENCE_DATE.year - first_edu) + 18
        if yoe > implied_age_now - 20:  # would imply starting career before ~20
            add(cid,"SOFT","YOE_AGE",f"YOE={yoe} vs implied age ~{implied_age_now} (edu start {first_edu}) — starts career <20yo")
    if eends:
        last_grad = max(eends)
        # graduated in the future is fine (ongoing). graduated long after heavy YOE is odd only if huge gap
        if starts and (last_grad - min(starts).year) > 6 and len(edu)==1:
            add(cid,"SOFT","GRAD_LATE",f"single-degree grad year {last_grad} is {last_grad-min(starts).year}y after first job {min(starts).year}")

    # ===== 'too perfect' synthetic engagement fingerprint =====
    pcs = float(sig.get("profile_completeness_score",0))
    rrr = float(sig.get("recruiter_response_rate",0))
    icr = float(sig.get("interview_completion_rate",0))
    oar = float(sig.get("offer_acceptance_rate",0))
    if pcs==100 and rrr==1.0 and icr==1.0 and oar==1.0:
        add(cid,"INFO","PERFECT",f"all engagement signals maxed (pcs=100,rrr=1,icr=1,oar=1)")

    # ===== reasoning-vs-fact: YOE stated in CSV vs raw =====
    reason = top[cid]["reasoning"]
    rm = re.search(r"([\d.]+)\s*YOE", reason)
    if rm:
        ry = float(rm.group(1))
        if abs(ry - yoe) > 0.6:
            add(cid,"SOFT","REASON_YOE",f"reasoning says {ry} YOE but raw profile YOE={yoe}")
    # title check
    ct = (prof.get("current_title","") or "").strip()
    if ct and ct.lower() not in reason.lower():
        add(cid,"INFO","REASON_TITLE",f"reasoning omits/alters current_title '{ct}'")

# ── report ──
HONEY = {"H1","H2","H3","H4","FOUND","FICT"}
honey_hits = {c:fs for c,fs in findings.items() if any(f[1] in HONEY for f in fs)}
hard_hits  = {c:fs for c,fs in findings.items() if any(f[0]=="HARD" for f in fs)}

print("="*78)
print(f"HONEYPOT-CLASS hits in top-100 (independent recompute): {len(honey_hits)}")
print("="*78)
for c in sorted(honey_hits, key=lambda x: top[x]["rank"]):
    print(f"\n  #{top[c]['rank']:>3} {c} score={top[c]['score']:.4f}")
    for sev,code,msg in honey_hits[c]:
        if code in HONEY:
            print(f"       [{code}] {msg}")

print("\n"+"="*78)
print(f"HARD data-integrity errors (impossible dates): {len(hard_hits)}")
print("="*78)
for c in sorted(hard_hits, key=lambda x: top[x]["rank"]):
    print(f"  #{top[c]['rank']:>3} {c}")
    for sev,code,msg in hard_hits[c]:
        if sev=="HARD":
            print(f"       [{code}] {msg}")

# soft-signal tally
from collections import Counter
tally = Counter()
for fs in findings.values():
    for sev,code,msg in fs:
        if code not in HONEY and sev!="HARD":
            tally[code]+=1
print("\n"+"="*78)
print("SOFT / discrepancy signal tally across top-100:")
print("="*78)
for code,n in tally.most_common():
    print(f"  {code:18} {n}")

# candidates with the most soft discrepancies (most suspicious non-honeypot)
susp = []
for c,fs in findings.items():
    soft = [f for f in fs if f[1] not in HONEY and f[0]!="HARD" and f[1]!="PERFECT" and f[1]!="REASON_TITLE"]
    if soft:
        susp.append((len(soft), c, soft))
susp.sort(reverse=True)
print("\n"+"="*78)
print("TOP-10 most discrepant non-honeypot profiles (eyeball these):")
print("="*78)
for n,c,soft in susp[:10]:
    print(f"\n  #{top[c]['rank']:>3} {c} score={top[c]['score']:.4f}  ({n} signals)")
    for sev,code,msg in soft:
        print(f"       [{code}] {msg}")

print("\n"+"="*78)
clean = len(raw) - len(honey_hits) - len(hard_hits)
print(f"VERDICT: {len(honey_hits)} honeypot-class, {len(hard_hits)} hard-integrity, "
      f"{len(raw)-len(honey_hits)} pass honeypot gate, of {len(raw)} audited.")
print("="*78)
