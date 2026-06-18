"""Second pass: technology-anachronism check + raw spot-checks for the top-100."""
import json, re, csv, sys, io
from datetime import date
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(r"A:\side_hustle\IndiaRuns Hackathon")
CORPUS = ROOT / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
SUB = ROOT / "redrob-ranking" / "submission_run.csv"
REF = date(2026, 5, 27)

# Public inception (year) of novel techniques/tools. Months of plausible usage by REF.
# Anything claiming substantially MORE months than (REF - inception) is anachronistic.
INCEPTION = {
    "lora": 2021, "qlora": 2023, "peft": 2023, "rag": 2020,
    "llamaindex": 2022, "langchain": 2022, "fine-tuning llms": 2020,
    "fine tuning llms": 2020, "llms": 2020, "vector search": 2019,
    "pinecone": 2021, "milvus": 2019, "weaviate": 2019, "qdrant": 2021,
    "chromadb": 2022, "sentence transformers": 2019, "faiss": 2017,
    "prompt engineering": 2021, "embeddings": 2018, "instructor": 2023,
    "vllm": 2023, "ollama": 2023, "gpt-4": 2023, "llama": 2023,
}

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

anach = {}  # cid -> list of (skill, claimed_mo, max_plausible_mo, inception_yr)
for cid, rec in raw.items():
    hits = []
    for s in rec.get("skills", []) or []:
        name = (s.get("name","") or "").lower().strip()
        dur = s.get("duration_months")
        if not isinstance(dur, int):
            continue
        yr = INCEPTION.get(name)
        if yr is None:
            continue
        max_mo = (REF.year - yr) * 12 + REF.month  # generous: whole inception year counts
        if dur > max_mo + 6:
            hits.append((s.get("name"), dur, max_mo, yr))
    if hits:
        anach[cid] = hits

n_with = len(anach)
total_hits = sum(len(v) for v in anach.values())
print("="*78)
print(f"TECHNOLOGY ANACHRONISMS in top-100: {n_with} candidates, {total_hits} skill-claims")
print("(skill duration_months exceeds the time the technology has existed)")
print("="*78)
for cid in sorted(anach, key=lambda c: top[c][0]):
    rk, sc = top[cid]
    print(f"\n  #{rk:>3} {cid} score={sc:.4f}")
    for name, dur, mx, yr in sorted(anach[cid], key=lambda x:-(x[1]-x[2])):
        yrs = dur/12
        print(f"       '{name}': claims {dur}mo (~{yrs:.1f}y) but tech exists ~{mx}mo (since {yr})  -> +{dur-mx}mo impossible")

# worst single offenders
flat = [(dur-mx, cid, name, dur, mx, yr) for cid,v in anach.items() for (name,dur,mx,yr) in v]
flat.sort(reverse=True)
print("\n"+"="*78)
print("Worst 12 individual anachronisms:")
print("="*78)
for gap, cid, name, dur, mx, yr in flat[:12]:
    print(f"  #{top[cid][0]:>3} {cid}  '{name}' {dur}mo vs {mx}mo possible (+{gap}mo)")

# ── raw spot-checks: rank #1 and the most-anachronistic top-50 record ──
def dump(cid):
    rec = raw[cid]; p = rec["profile"]; rk,sc = top[cid]
    print("\n"+"#"*78)
    print(f"RAW SPOT-CHECK  #{rk} {cid}  score={sc:.4f}")
    print("#"*78)
    print(f"  name={p.get('anonymized_name')}  title={p.get('current_title')}")
    print(f"  company={p.get('current_company')} ({p.get('current_company_size')})  YOE={p.get('years_of_experience')}")
    print(f"  location={p.get('location')}, {p.get('country')}  industry={p.get('current_industry')}")
    print(f"  headline={p.get('headline')}")
    print("  CAREER:")
    for r in rec.get("career_history",[]):
        print(f"    - {r.get('start_date')}..{r.get('end_date')}  {r.get('duration_months'):>3}mo  cur={str(r.get('is_current')):5}  {r.get('title')} @ {r.get('company')} [{r.get('industry')}]")
    print("  EDUCATION:")
    for e in rec.get("education",[]):
        print(f"    - {e.get('start_year')}-{e.get('end_year')}  {e.get('degree')} {e.get('field_of_study')} @ {e.get('institution')} (tier {e.get('tier')})")
    print("  SKILLS (name / prof / dur_mo / endorsements):")
    for s in rec.get("skills",[]):
        print(f"    - {s.get('name'):28} {s.get('proficiency'):12} {str(s.get('duration_months')):>4}mo  e={s.get('endorsements')}")
    sg = rec.get("redrob_signals",{})
    print("  SIGNALS:")
    print(f"    completeness={sg.get('profile_completeness_score')}  open_to_work={sg.get('open_to_work_flag')}  last_active={sg.get('last_active_date')}  signup={sg.get('signup_date')}")
    print(f"    recruiter_rr={sg.get('recruiter_response_rate')}  resp_time_h={sg.get('avg_response_time_hours')}  interview_cr={sg.get('interview_completion_rate')}  offer_ar={sg.get('offer_acceptance_rate')}")
    print(f"    github={sg.get('github_activity_score')}  notice_days={sg.get('notice_period_days')}  relocate={sg.get('willing_to_relocate')}  mode={sg.get('preferred_work_mode')}")
    print(f"    assessments={sg.get('skill_assessment_scores')}")

dump("CAND_0011687")  # rank 1
if flat:
    worst_cid = flat[0][1]
    dump(worst_cid)
