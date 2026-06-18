# Redrob Ranking Challenge — Project Context (Single Source of Truth)

> **What this file is.** The authoritative orienting document for the project. Two companion files:
> - **`decision.md`** — the running decision log (Decision / Rejected / Why / Watch). Interview script + methodology evidence.
> - **`redrob_build_playbook.md`** — build sequence and operating rules (phases, checkpoints).
>
> **Provenance legend — every factual claim is tagged so you always know its source at Stage 5:**
> - **`[SPEC]`** = stated in the original organizer bundle (`job_description.docx`, `submission_spec.docx`, `README.docx`, `redrob_signals_doc.docx`, `candidate_schema.json`, `validate_submission.py`). These `.docx` files are the **true source of truth**; `problemStatement.md` is a consolidation we made of them (verified faithful, but always cite the original).
> - **`[DATA]`** = measured directly from the real `candidates.jsonl` (100,000 records). Reference "now" for all date math = **2026-05-27** (max `last_active_date` in the file).
> - **`[CALL]`** = our engineering judgment / design decision. Defensible, but ours — not handed to us.

---

## 1. The challenge in one paragraph `[SPEC]`

Given **100,000 candidate profiles** and **one job description**, build a **CPU-only, offline** system that returns the **top 100 best-fit candidates** as a CSV (`candidate_id, rank, score, reasoning`). The ranking step must run in **≤5 min wall-clock, ≤16 GB RAM, no GPU, no network, ≤5 GB intermediate disk**. Output is scored once against a **hidden ground truth**. **No live leaderboard**, **max 3 submissions**, **last valid submission is final**. *(submission_spec.docx §1–4, 8; README.docx.)*

The win condition is **not the best code** — a basic ranker looks similar across teams. `[CALL]` It is the best **judgment, trap handling, and defensibility**, plus the ability to explain all of it in a 30-minute live interview (Stage 5).

---

## 2. The role and what "good fit" means `[SPEC]`

**Role:** Senior AI Engineer — Founding Team at Redrob AI (Series A talent-intelligence platform; Pune/Noida; hybrid). *(job_description.docx.)*

**The central thesis, stated verbatim in the JD:** the right answer is *the gap between what the JD says and what the JD means*. A candidate who **built a recommendation system at a product company is a fit even without writing "RAG" or "Pinecone."** A Marketing Manager whose skills list contains every AI keyword is **not**. **Career history and title outrank the skills list.**

### Hard requirements ("absolutely need") `[SPEC]`
1. Production **embeddings-based retrieval** (drift, index refresh, retrieval-quality regression).
2. Production **vector DB / hybrid search** (FAISS, Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, etc.).
3. Strong **Python** / code quality.
4. Hands-on **ranking evaluation frameworks** (NDCG, MRR, MAP, offline-to-online correlation, A/B interpretation).

### Primary disqualifiers — exact JD language and category `[SPEC]`
The JD has two distinct negative sections. Only **one** disqualifier carries the literal "we will not move forward." Severity must mirror the actual wording — getting this wrong is a Stage-5 credibility hit.

| # | Disqualifier | JD's actual language / category | Operationalization `[CALL]` | Treatment `[CALL]` |
|---|---|---|---|---|
| **D1** | Pure research, no production | **"we will not move forward. We are explicit about this."** — the single hard disqualifier | research-only career, zero production/deployment in any role | **hard floor** |
| **D2** | "AI experience" only recent (<12 mo) LangChain→OpenAI | **"probably not move forward, *unless* [substantial pre-LLM-era ML production]"** — conditional | ML evidence confined to a sub-12-month current role, no pre-LLM ML production in history | heavy penalty **unless** pre-LLM ML present |
| **D3** | Senior, no production code in 18 mo | **"probably not move forward. This role writes code."** — firm | current role is architect/lead/manager-only, no hands-on IC code in 18 mo | heavy penalty |
| **D4** | Entire-career consulting | **"Things we explicitly do NOT want"** + explicit exception: *"currently at one… but prior product-company experience, that's fine"* | **all** career companies are consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini + "etc." extensions: Mindtree, HCL, Tech Mahindra, Mphasis) | **cap — but NOT if any prior product-company role exists** |
| **D5** | CV / speech / robotics primary, no NLP/IR | **"Things we explicitly do NOT want"** + softer: *"we respect your work but you'd be re-learning"* | primary expertise CV/speech/robotics, no NLP/IR career evidence | strong penalty |

### Secondary negative signals (JD "do NOT want," weaker or harder to detect) `[SPEC]` + `[CALL]`
These were omitted from the earlier consolidation; the JD names them and the ground truth may weight them.
- **Title-chasers** — job-hopping for titles ("every 1.5 years"); JD wants a **3+ year** commitment. Detectable: 3+ roles with avg tenure <18 mo = **3.0% of pool** `[DATA]`. → minor penalty.
- **Framework enthusiasts** — LangChain-tutorial GitHub, demos over systems. Overlaps D2; folded into the recent-LLM signal. → no separate rule.
- **Closed-source-proprietary 5+ yrs, no external validation** (papers/talks/OSS). Largely **unrepresentable** in the schema; `github_activity_score` (−1 for 64.6% of pool `[DATA]`) is a weak proxy only. → acknowledged, not heavily weighted.

### Logistics / fit factors (JD-weighted; named in the Stage-4 honest-concerns check) `[SPEC]` + `[DATA]`
The Stage-4 reasoning check explicitly looks for honest concerns like **"long notice period, wrong work mode"** — so these must be scored *and* surfaced in reasoning.
- **Location:** **75.1% India-based** `[DATA]`; the other ~25% (USA 10%, plus ~2.5% each Australia/Canada/UK/Germany/Singapore/UAE) face *"case-by-case, but we don't sponsor work visas."* → India-based **or** `willing_to_relocate` = full credit; international-not-relocating = discount; mild bonus for Noida/Pune/Hyderabad/Mumbai/Delhi-NCR/Bangalore. `[CALL]`
- **Notice period:** p50 = **90 days**; only **13.8%** are sub-30 `[DATA]`. JD: sub-30 ideal, can buy out 30, *"30+ day… the bar gets higher."* → short = small bonus, long = small penalty + honest-concern flag; **never disqualifying.** `[CALL]`
- **Work mode:** ~uniform across onsite/hybrid/remote/flexible (~25% each) `[DATA]`. JD is hybrid. → mismatch (e.g., remote-only) = minor honest-concern flag. `[CALL]`

### The JD's "ideal candidate" `[SPEC]`
6–8 years total, 4–5 in applied ML/AI at **product** companies (not services); has **shipped ≥1 end-to-end ranking/search/recommendation system** at real scale; defensible opinions on retrieval/eval/LLM-integration tied to systems built; in or willing to relocate to Noida/Pune; **active on the platform.** The JD expects **few matches in 100K and is fine with that** — *"10 great matches over 1000 maybes."*

---

## 3. Scoring and pipeline — what determines placement `[SPEC]`

### Composite metric (Stage 2)
```
Composite = 0.50 × NDCG@10  +  0.30 × NDCG@50  +  0.15 × MAP  +  0.05 × P@10
```
*(submission_spec.docx §4.)* **Strategic implication** `[CALL]`: **NDCG@10 alone is half the score**, so the *ordering quality of your top 10* dominates; NDCG@10 + NDCG@50 together are 80% of weight (NDCG@50 encompasses the top 10). Getting the genuine best fits into the right order at the very top is where the score is won or lost. P@10 is measured against **relevance tiers** (tier 3+ = relevant); **honeypots are forced to tier 0.**

**Composite tiebreakers** (between submissions): higher **P@5** → higher **P@10** → earlier timestamp. `[SPEC]`

### Five-stage evaluation pipeline `[SPEC]`
| Stage | What happens | Elimination trigger |
|---|---|---|
| **1. Format validation** | `validate_submission.py` on every submission | any spec violation |
| **2. Scoring** | composite computed once on hidden ground truth | below cutoff |
| **3. Code reproduction + honeypot check** | ranking step reproduced in sandboxed Docker (5 min / 16 GB / no GPU / no network) | cannot reproduce; **honeypot rate >10% in top 100**; missing/fabricated repo |
| **4. Manual review** | 6 reasoning checks; methodology coherence; git authenticity; code quality | failed reasoning; flat/single-dump git; codebase entirely LLM API calls |
| **5. Defend-your-work interview** | 30-min video; walk architecture, defend choices | cannot explain; contradicts code; clearly didn't build it |

### Stage-4 reasoning checks (all six must pass) `[SPEC]`
Specific facts · JD connection · honest concerns · no hallucination · variation across rows · **rank-consistency** (tone matches rank). *Penalized:* empty, all-identical, name-insert templating, hallucinated skills, rank-contradicting tone.

### Validator scope — two traps `[SPEC]`
- **Tie-break trap (Stage-1 rejection).** Spec §3 says break ties "using a secondary signal from your model, **or** by `candidate_id` ascending." But `validate_submission.py` **rejects** any equal-score adjacent pair where the better-ranked candidate has a *larger* `candidate_id` — it enforces `candidate_id`-ascending on ties regardless of the spec's "secondary signal" wording. → **Mitigation** `[CALL]`: emit a **continuous, distinct float score** so no equal-score pairs exist and the check never fires (your model's ordering is then fully respected). If ties are unavoidable, sort equal-score groups `candidate_id`-ascending.
- **Existence is not checked locally.** The validator checks only the `^CAND_[0-9]{7}$` pattern; **existence in `candidates.jsonl` is enforced server-side**, not by the local validator. Passing locally ≠ valid IDs. → Moot for us (we rank real records), but never hand-edit IDs.

---

## 4. Data reality — measured on the full 100K `[DATA]` (this is our edge)

All figures from a complete pass over `candidates.jsonl`. Reference "now" = **2026-05-27**.

### 4.1 Honeypots — the highest-stakes finding
- **Schema-detectable detector (H1–H4, 4 deterministic rules) catches 68 records (0.068%)**, matching the organizer's stated *"~80"* `[SPEC: submission_spec.docx §7]` with near-zero collateral:
  - **H1** — a role's `duration_months` exceeds months since its `start_date` (+3 mo slack). `[CALL]`
  - **H2** — summed role tenure exceeds stated `years_of_experience` × 12 by >30 mo. `[CALL]`
  - **H3** — stated YOE × 12 exceeds total career span (earliest start → now) by >18 mo. `[CALL]`
  - **H4** — a skill at **{advanced, expert}** proficiency with **0 `duration_months`**. Maps to `[SPEC]` example *"expert proficiency in 10 skills with 0 years used."* `[CALL]`
- **H5 (founding-date, real companies)** `[CALL]` — for each role at a **real, non-fictional** company, flag if `role.start_date.year < company_founding_year`. Fictional employers (Hooli, Stark Industries, Dunder Mifflin, Wayne Enterprises, Initech, Pied Piper, Acme Corp, Globex Inc) are skipped per-role. All 55 real corpus companies are mapped; 5 have binding founding years: CRED (2018), Krutrim (2023), Sarvam AI (2023), Glance (2019), Rephrase.ai (2019). **Catches 250 additional records. Total honeypots: 318.** Founding years verified against primary/first-party sources only.
- **Residual ~12 (fictional-employer founding-date cases):** undetectable — the schema has **no company founding dates** and fictional employers have none to look up. H5 and the top-150 audit both cover only known real companies, so this residual stays uncatchable. Bounded by: DQ requires **>10 honeypots in the top 100**; fictional-employer residual only fatal if ~11 of ~12 both reach the shortlist and out-score genuine fits (improbable — they carry no career-evidence advantage).
- **The naive rule to AVOID — "skill `duration_months` > career length = honeypot":** fires on **13,436 records (13.4%)**, catches only **5 of the 68** schema-detectable honeypots, and **would delete `CAND_0000031`** (88-month Pinecone on a 72-month career = 7 yrs of normal pre-career vector-DB familiarity). Do not use it.

### 4.2 Population structure `[DATA]`
| Segment | Count | Share |
|---|---|---|
| Engineering / technical titles | 42,672 | 42.7% |
| Non-technical titles (not eng) | 45,912 | 45.9% |
| **Consulting-only** entire careers | 9,745 | 9.7% |
| **Keyword-stuffers** (non-eng title + ≥3 AI skills) | 3,618 | 3.6% |
| **Job-hoppers** (3+ roles, avg tenure <18 mo) | 1,707 of 57,306 multi-role | 3.0% |
| Product-industry engineers (non-consulting) | ~13,441 | — |
| Services (IT-Services) engineers | ~8,546 | — |
| India-based | 75,113 | 75.1% |

### 4.3 Behavioural signals (availability multiplier inputs) `[DATA]`
| Signal | Distribution |
|---|---|
| `last_active` staleness (days) | p10 = 20 · p50 = 105 · p90 = 206 · **max = 240** |
| `recruiter_response_rate` | p10 = 0.14 · p50 = 0.44 · p90 = 0.73 |
| `open_to_work_flag` = TRUE | **35.3%** (≈65% NOT open) |
| `github_activity_score` = −1 (missing) | 64.6% |
| `notice_period_days` | p10 = 30 · p50 = 90 · p90 = 150 (13.8% sub-30) |
| `willing_to_relocate` = TRUE | 28.8% |
| `preferred_work_mode` | ~uniform (~25% each onsite/hybrid/remote/flexible) |
| **Ghosts** (stale >120 d AND rrr <0.15 AND not open) | **3.4%** |

**Calibration consequence** `[CALL]`: ghosts are only ~3.4%. A multiplier that compounds three independent penalties (open × response × staleness) drives the **median candidate to 0.37×** and pushes **77% below 0.5×** — it flattens everyone instead of down-weighting ghosts, and since NDCG@10 is half the score it actively *demotes* good fits. The multiplier must stay gentle in the healthy range and go harsh only on genuine disengagement (§7, Phase 3).

### 4.4 The genuine tier-5 pool — what the top 10 competes for `[DATA]`
Strict definition (discipline-signalling title **AND** career text evidencing a real ranking/retrieval/recommendation/search system **AND** not consulting-only **AND** 4–11 YOE):
- **374 candidates (0.37%)** qualify; **244 are fully available** (open + responsive + active ≤180 d).
- **Archetype: `CAND_0000031`** — "Recommendation Systems Engineer @ Swiggy," shipped XGBoost/LightGBM ranking models for a discovery feed, evaluation-aware, response-rate healthy, availability ≈ 0.95.
- The pool is **internally graded** `[CALL]`: true tier-5s sit above a large tier-3/4 crowd whose career text is the dataset's lighter boilerplate ("built recommendation-style features at a mid-stage startup… lighter weight than ranking systems at FAANG"). Some apparent matches are actually **D5 disqualifiers** (e.g., a Computer Vision Engineer doing predictive modeling) and must be capped despite a plausible title. Ordering *within* this pool is the whole game.

---

## 5. Strategic thesis — what wins `[CALL]`

1. **Career-evidence-first scoring.** Title + career descriptions are high-signal; the **skills list is near-noise** (the dataset's first record lists Kubernetes and NLP on a Customer Support profile). Read what people *built*, not what they *listed*.
2. **The rubric is the biggest edge.** With a hidden ground truth, the team whose internal "good fit" model matches Redrob's labelling wins — a **reasoning artifact, not a code artifact**.
3. **Deciding factor = ordering the top ~374, especially the top 10** (50–80% of composite).
4. **Two calibration traps invert the sign of effort if missed:** the skill-duration honeypot rule (13% false-positive, deletes the archetype) and the compounding availability multiplier (median → 0.37×). Both addressed below.
5. **Disqualifiers are gates, not deductions.** The hard ones are binary; point-subtractions let keyword-rich consultants float up.

---

## 6. Architecture `[CALL]` (in-budget per `[SPEC]` constraints)

**Offline precompute (uncapped time, run once):**
- Stream `candidates.jsonl` line-by-line — **never load the full file into a list** (it's **487 MB** as delivered; README quotes ~465 MB).
- Embed all 100K candidate texts with **bge-base-en-v1.5** → store as a `float32` matrix on disk (100K × 768 ≈ **307 MB**).
- Persist parsed feature vectors (title/career flags, disqualifier gates, logistics factors, availability inputs, honeypot flags).

**Ranking step (must fit ≤5 min / ≤16 GB / no network):**
1. Embed the JD **once**.
2. Cosine similarity vs. the precomputed matrix (sub-second over 100K × 768; ~25 ms).
3. Apply **rubric feature scores** (career-first; disqualifier gates → `gated`; logistics factors).
4. **Hard-exclude H1–H5 honeypots** from the candidate set (before any further scoring).
4a. **Hard-exclude discrepancy-gate failures** (skill-anachronism + education-integrity) — applied after honeypots, before the multiplier.
5. Apply the **continuous availability multiplier** over surviving rows.
6. Sort by final (distinct) score descending → take top 100.
7. **Top-150 audit guard** (§7, Phase 2) over the shortlist.
8. Generate **fact-grounded reasoning** per candidate → write CSV → run `validate_submission.py`.

The dominant cost (embedding 100K) is in *precompute*, not the ranking step; the ranking step is a matrix multiply + feature scoring over 100K rows.

---

## 7. Settled decisions (the register)

`[confirmed]` = locked by you. `[refined]` = changed on re-evaluation. All `[CALL]` unless noted.

### Phase 0 — Foundations
- **Embedding model: bge-base-en-v1.5 (768-d, ~307 MB matrix), default.** `[confirmed: your machine can run local precompute]` Embeddings are a *secondary* signal (mainly to rescue plain-language tier-5s), but bge-base is a genuinely stronger retrieval model than small alternatives (MiniLM/bge-small), and the ranking-step cost is negligible — cosine similarity over 100K at 768-d is ~25 ms, easily in budget. The precompute wall-time is longer (your machine's patience), not a problem. If precompute ever runs unusually long, optimize the candidate-text ingestion or the parsing step, not the embedding dimension — dimension doesn't threaten the 5-minute ranking budget.
- **Repo: flat, legible, one reproduce command.** `precompute.py`, `rank.py`, `rubric.py`, `detectors.py`, `reasoning.py`, plus `artifacts/`, `decision.md`, `README.md`, `requirements.txt`, `submission_metadata.yaml`. A traceable pipeline, not a package.

### Phase 1 — The rubric (the differentiator)
- **Disqualifiers are GATES/CAPS, not subtractions** (D1–D6 per §2 + visa cap; severity matched to JD language; **D4 fires only on entire-career consulting** with the prior-product exception; **D6 title-chaser cap 0.50** on ≥3 employers with avg tenure <18 mo; **visa cap 0.55** on international-not-relocating).
- **"Product over services" → credit multiplier on the *relevant* experience** (IT-Services + consulting names = services; Software/Fintech/E-commerce/Food Delivery = product).
- **"Shipped a real ranking/search/rec system" → weighted heaviest, read from career descriptions, never the skills list** (rarest, highest signal; 374 candidates).
- **"Wrote code recently" → recency gate on current role** (hands-on IC engineering within 18 mo).
- **Secondary negatives** (title-chasers minor penalty; framework-enthusiasts folded into D2; closed-source acknowledged-but-light) and **logistics factors** (location/relocation, notice period, work mode per §2) included as minor weights.

### Phase 2 — Traps
- **Honeypot detector: H1–H5. Explicitly NOT skill-duration-vs-career.** Hard-exclude flagged candidates from the top 100. H1–H4 catch 68 schema-detectable honeypots; H5 (founding-date, real companies only) catches 250 additional. Total: 318. `[refined]`
- **Discrepancy gates (after honeypots): skill-anachronism (955) + education-integrity (7,967).** Hard-exclude on physical impossibilities — technology claimed for more months than it has existed; internally-impossible degree timelines. Applied before the availability multiplier. Combined with H1–H5: 9,108 of 100,000 excluded (90,892 surviving). `[CALL, new]`
- **Keyword-stuffer (3.6%): flag, do not double-penalize** (career-first already demotes; aggressive penalty false-positives genuine career-changers).
- **Plain-language tier-5: a rescue, not a penalty** (semantic embedding + career-evidence keyword pass).
- **Behavioral twins: handled by the availability multiplier**, not a bolt-on.
- **Phase 2 closeout — Top-150 audit guard `[new]`:** after producing the shortlist, run an intensive consistency audit over the **top ~150** (cheap vs. 100K): re-run H1–H4, scrutinize any profile whose fit looks "too good," and check founding-date plausibility for *known real* companies (Swiggy, Razorpay, Paytm, etc.). Plus a human eyeball before submission. This is where the >10% disqualification risk actually lives. It catches founding-date implausibilities at **known real** companies (which 100K-wide rules also miss — the schema has no founding dates); the **fictional-employer residual stays uncatchable even here** and is bounded by the >10-in-top-100 DQ threshold, not the audit.

### Phase 3 — The ranker
- **Feature weights: career-first** — starting point ≈ career-evidence + title-fit **70%**, embedding/semantic **20%**, skills list **10%** (education/certs dropped 2026-06-18 — the JD names zero degree/institution criteria; weight folded into career), with logistics as a small modifier. **These are a starting point to calibrate by hand-ranking the 50-candidate sample** (Phase 1 checkpoint), not fixed constants.
- **Availability multiplier: CONTINUOUS, gentle healthy slope (~0.7–1.0), steep ghost floor (~0.15). NOT a product of independent penalties.** Continuous so behavioral twins always separate; the floor triggers only on the *conjunction* of stale-beyond-**120**-days AND near-zero response AND not-open. (120-day threshold, calibrated to the 3.4% ghost population `[DATA]`; 180-day was too strict.) **Highest-leverage calibration in the ranker — tune against synthetic twins (the sample has no strong twin pair or high-base ghost; §9) before trusting it.** `[refined]`
- **Tie-breaking: continuous distinct float scores** so the validator's `candidate_id`-ascending tie check never fires (see §3). Spec-mandated fallback if ties occur: `candidate_id` ascending.

### Phase 4 — Reasoning
- **Option (a): fact-grounded templates. No local LLM.** `[confirmed]` 3–4 structural variants, each pulling *different real facts* per candidate (years, title, named skills, signal values, the actual gap). Satisfies all six Stage-4 checks deterministically, is in-budget, and is byte-stable for Stage-3 reproduction. (LLM rejected: nondeterminism = reproduction friction; hallucination = fastest Stage-4 fail.)
- **Rank-consistency + honesty policy.** `[confirmed]` Every string = **a real fact + a JD connection + an honest gap where one exists** (long notice, wrong work mode, services background, international/visa, stale activity). Tone scales to rank: top-10 leads with strongest evidence (may note one caveat); bottom names the disqualifying gap plainly.

### Phase 5 — Hardening, sandbox, metadata
- **Sandbox: HuggingFace Spaces (Streamlit) primary, Docker `docker run` recipe in README as backup.** CPU-only / no-network / ≤100-sample fits the HF free tier; the spec accepts an unmodified `docker run` substitute. Doing both de-risks the Stage-1 sandbox flag.
- **`submission_metadata.yaml`** complete and honest, including the AI-tools declaration (*declared use is not penalized; contradicting your interview is* — `[SPEC: submission_spec.docx §10.4]`).

---

## 8. Dataset traps → handling

| Trap | Source | Detection `[CALL]` | Handling `[CALL]` |
|---|---|---|---|
| **Keyword stuffers** | `[SPEC: README]` | non-eng title + ≥3 AI skills, no career evidence (3.6%) | demoted by career-first weighting; flagged for honest reasoning, not double-penalized |
| **Plain-language tier-5** | `[SPEC: README]` | semantic embedding match + career-evidence keyword pass | rescued — surfaced despite no buzzwords |
| **Behavioral ghosts** | `[SPEC: JD + redrob_signals_doc]` (availability concept) | conjunction rule (3.4%) | crushed by the steep tail of the continuous availability multiplier |
| **Behavioral twins** | `[SPEC: README]` | n/a — handled by design | separated by the continuous (non-bucketed) availability multiplier |
| **Honeypots (~80)** | `[SPEC: submission_spec §7, README]` | H1–H4 (68 caught, 0 collateral) + top-150 audit | hard-excluded from top 100; forced toward tier 0 |

*(Note: README explicitly names keyword-stuffers, plain-language tier-5s, behavioral twins, and honeypots; "behavioral ghosts" is the availability/inactive-candidate concept from the JD and `redrob_signals_doc`. The detailed taxonomy and all detection logic are `[CALL]`.)*

---

## 9. Risks and watch items `[CALL]`

- **Top-10 ordering is reasoned, not tuned (highest)** — NDCG@10 + NDCG@50 are 80% of the composite, but the 50-sample holds **one** tier-5 (`CAND_0000031`), so calibration can separate it from non-fits, not order the top 10. Make the tier-5 ordering rules an explicit written, JD-traced deliverable and exercise them on synthetic variants; the honest Stage-5 line is "reasoned, not tuned."
- **D2/D3/D5 gates can delete a top-10 candidate, FP rate unmeasured** — measure each gate's firing rate on the full 100K and inspect who it caps; soften any that catch a plausible fit (the same archetype-deletion failure the skill-duration rule was rejected for). D2 is fuzziest (proxy: ML only in a sub-12-mo current role, no pre-LLM ML); don't overclaim precision.
- **Availability multiplier calibration** — the one high-leverage number; validate the 0.7–1.0 band and ~0.15 floor against **synthetic twins** (the sample has no tier-5 twin pair and no high-base ghost, only two low-tier ghosts). Too steep → unseen NDCG@10 loss.
- **Reasoning template collisions** — 3–4 variants over 100 rows let similar candidates collide into near-identical strings (the Stage-4 "name-insert templating" penalty); run a pairwise near-duplicate check across all 100, not a 5-string spot-read.
- **Honeypot residual (~12 uncaught)** — founding-date cases the schema can't represent (fictional employers); the top-150 audit covers only known real companies, so the residual is bounded by the >10-in-top-100 DQ threshold, not the audit. Stated honestly at Stage 5.
- **Parser field noise** — in the sample, `career_history[].description`/`.industry` don't reliably track the title (mismatched/repeated descriptions); eyeball the parsed `product_vs_services`/`built_real_system` flags on the 50 before calibrating.
- **False-positive risk on genuine career-changers** — the dataset has real backend→ML transitioners; don't let keyword overlap or a single services employer cap them.
- **Weights are hand-tuned on a 50-sample, not learned** — defensible for this task; say so plainly, don't imply a trained learning-to-rank model you didn't build.
- **`problemStatement.md` is a derivative** — always trace facts to the four original `.docx` files; this file's `[SPEC]` tags cite them.

---

## 10. Interview-defense anchors (Stage 5 = win condition) `[CALL]`

Answer these **cold**; each traces to a decision above and a `decision.md` entry:

1. **Walk one candidate from raw JSON → final rank**, narrating every transformation. (Pre-rehearse `CAND_0000031` and one mid-pack candidate.)
2. **Why a *gentle, continuous* availability multiplier, not a hard multiplicative crush?** (Ghosts are 3.4%; a compounding crush flattens the median to 0.37× and demotes good fits.)
3. **Why reject the skill-duration honeypot rule?** (False-positives 13.4% of the pool, catches only 5 of 68 real honeypots, and would delete `CAND_0000031`; we used four timeline/proficiency rules instead.)
4. **Why does career history outrank the skills list?** (The JD's explicit thesis + the keyword-stuffer trap.)
5. **Why fact-grounded templates over an LLM?** (Reproducibility + hallucination risk vs. marginal prose gain.)
6. **How do two near-identical candidates rank differently?** (The continuous availability multiplier separates twins by their signal delta.)
7. **You only catch 68 of ~80 honeypots — why is that acceptable?** (Tested six more signatures: five find nothing, the sixth is a false-positive trap; the residual is founding-date cases the schema can't represent. The top-150 audit covers only known real companies, and the fictional-employer residual is bounded by the >10-in-top-100 DQ threshold.)
8. **Your top-10 ordering is 80% of the score but the 50-sample has one tier-5 — how do you know it's right?** (You can't validate it empirically; the order is *reasoned*, not tuned — here are the written, JD-traced tier-5 ordering rules and the synthetic-variant tests. The honest answer beats pretending the sample validated it.)

---

*Grounded against the full 100K dataset on 2026-06-14. Source-of-truth files: `job_description.docx`, `submission_spec.docx`, `README.docx`, `redrob_signals_doc.docx`, `candidate_schema.json`, `validate_submission.py`. Reference "now" for all date math: 2026-05-27.*
