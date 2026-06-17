# Architecture — Redrob Intelligent Candidate Discovery & Ranking Challenge

> **What this file is.** The technical architecture of the ranking system: components, data flow, the scoring composition, the compute budget, and the precompute/ranking split. Derived from `context.md` (the project's single source of truth), corrected against the original organizer bundle, and verified to be in-budget per `[SPEC]`.
>
> **Provenance legend** (mirrors `context.md`):
> - `[SPEC]` — stated in the original organizer bundle (`job_description`, `submission_spec`, `README`, `redrob_signals_doc`, `candidate_schema.json`, `validate_submission.py`). These are the true source of truth.
> - `[DATA]` — measured over the full `candidates.jsonl` (100,000 records). Reference "now" = **2026-05-27** (max `last_active_date`).
> - `[CALL]` — engineering judgment / design decision. Defensible, ours.
>
> **Two clarifications vs. `context.md`'s prose** (see §11 for the full verification ledger):
> 1. Embedding dimension is **768** (`bge-base-en-v1.5`), matrix **≈307 MB** — now consistent across `decision.md` and `context.md` §6/§7. (Earlier `context.md` §6 carried a stale "384 / 154 MB"; reconciled in this pass.)
> 2. The JD embedding is treated as a **precompute artifact**, so the in-budget ranking step loads no model and needs no network. This closes a no-network failure mode the prose left implicit.

---

## 1. Design constraints (the envelope everything must fit) `[SPEC]`

The ranking **step** that produces the CSV must satisfy, simultaneously (`submission_spec.md` §3):

| Constraint | Limit |
|---|---|
| Wall-clock runtime | ≤ 5 minutes |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only — no GPU |
| Network | Off — no hosted LLM / API calls during ranking |
| Intermediate disk | ≤ 5 GB |

**Pre-computation is exempt** from the 5-minute cap (`submission_spec.md` §10.3): embeddings/indexes may be built offline; only the ranking step is timed and reproduced at Stage 3. The entire architecture is organized around moving the one expensive operation — embedding 100,000 texts — into precompute, leaving the ranking step as a vectorized matrix-multiply plus feature scoring.

**Output** `[SPEC]`: a UTF-8 CSV, columns `candidate_id,rank,score,reasoning` in that order, exactly 100 data rows, ranks 1–100 each used once, `score` non-increasing with rank, top-1 = best fit. Validated by `validate_submission.py` before every submission.

---

## 2. System overview — two stages, one boundary

```
                      OFFLINE PRECOMPUTE                         IN-BUDGET RANKING STEP
                  (uncapped time, run once)                    (≤5 min / ≤16 GB / CPU / no net)
   ┌───────────────────────────────────────────┐      ┌──────────────────────────────────────────┐
   │ candidates.jsonl  (487 MB [DATA] /          │      │  load artifacts:                          │
   │   ~465 MB [SPEC: README])                   │      │    • candidate matrix  (100K×768, 307 MB) │
   │            │ stream line-by-line            │      │    • parsed feature store                 │
   │            ▼                                │      │    • JD vector (768-d)                    │
   │  parse + build candidate text               │      │            │                              │
   │            │                                │      │            ▼                              │
   │   ┌────────┴────────┐                       │ ───▶ │  1. cosine(JD, matrix)  → sim[100K]       │
   │   ▼                 ▼                       │      │  2. rubric base score   (career-first)    │
   │ bge-base-en-v1.5   feature parser           │      │  3. disqualifier gates  (cap/floor)       │
   │ embeddings         (title/career flags,     │      │  4. honeypot H1–H4      (hard-exclude)    │
   │   │                 disqualifier gates,      │      │  5. availability multiplier (continuous)  │
   │   ▼                 logistics, availability  │      │  6. distinct-float final score            │
   │ candidate matrix    inputs, honeypot flags)  │      │  7. sort ↓ → take top 100                 │
   │   (307 MB on disk)        │                  │      │  8. top-150 audit guard                   │
   │            └──────────────┘                  │      │  9. fact-grounded reasoning (templates)   │
   │  + embed the fixed JD once → JD vector       │      │ 10. write CSV → validate_submission.py    │
   └───────────────────────────────────────────┘      └──────────────────────────────────────────┘
                                                                          │
                                                                          ▼
                                                                  <team_id>.csv  (top 100)
```

**The boundary is the whole trick** `[CALL]`: embedding 100K candidates is the only heavy operation, and it lives entirely in precompute. The ranking step touches no model and no network — it reads three artifacts and does arithmetic. This is what buys the large margin under the 5-minute cap (see §10).

---

## 3. Embedding model `[CALL, confirmed in decision.md]`

- **Model:** `bge-base-en-v1.5`, **768-dimensional** sentence embeddings.
- **Candidate matrix:** 100,000 × 768 × 4 bytes (`float32`) = **307.2 MB** on disk and in RAM. Trivial against 16 GB / 5 GB caps.
- **JD vector:** 768 × 4 = ~3 KB, precomputed once (the JD is fixed and released ahead of time).
- **Why bge-base over bge-small/MiniLM (384-d):** the machine runs precompute without time pressure, so there is no reason to trade retrieval quality for speed. bge-base is a genuinely stronger retrieval model and surfaces **plain-language tier-5** candidates better. The ranking-step cost difference is invisible — cosine similarity over 100K is ~25 ms at 768-d vs ~14 ms at 384-d, both negligible against 5 minutes. The only cost is precompute wall-time, which is uncapped.
- **Role of embeddings:** a **secondary** signal (≈20% weight, §6). Its specific job is the plain-language rescue — catching candidates who *built* the right systems but describe them without buzzwords. It does **not** drive ranking; the career rubric does.

**Offline-model requirement** `[CALL]` (closes a no-network failure mode): `bge-base-en-v1.5` weights must be available locally with no Hub call.
- **Precompute** loads the model from a bundled local path (`HF_HUB_OFFLINE=1`) to embed candidates and the JD.
- **The 100K ranking step loads no model** — it consumes the precomputed candidate matrix and the precomputed JD vector. This is why the ranking step trivially satisfies "network off."
- **The sandbox demo** (≤100 sample, §9) *does* load the model to embed the sample live, so the model weights ship as a repo artifact for that path and for Stage-3 regeneration.

---

## 4. Precompute stage (offline, run once)

**Objective:** produce three artifacts that make the ranking step a pure arithmetic pass.

1. **Stream `candidates.jsonl` line-by-line** `[CALL]` — never load the full file into a list. It is **487 MB** as delivered `[DATA]` (`README` quotes ~465 MB `[SPEC]`); a full in-memory list of 100K parsed dicts risks the 16 GB ceiling and is unnecessary.
2. **Build candidate text** for embedding from the high-signal fields only — `current_title`, `headline`, `summary`, and each `career_history[].description`. The **skills list is deliberately excluded or down-weighted** from the embedding text (it is the keyword-stuffer attack surface; see §5.1).
3. **Embed** each candidate text with bge-base → append to the `float32` matrix.
4. **Parse a feature vector** per candidate (deterministic, no model): title/career-evidence flags, disqualifier-gate booleans, logistics factors, availability-signal inputs, and honeypot flags. Persist this **feature store** — so the ranking step never re-parses 487 MB.
5. **Embed the fixed JD once** → persist the 768-d JD vector.

**Precompute artifacts:**

| Artifact | Size | Purpose |
|---|---|---|
| `candidate_matrix.npy` (100K×768 float32) | ~307 MB | cosine similarity in ranking step |
| `features.parquet` / `.npz` (parsed per-candidate fields) | < ~100 MB | rubric scoring, gates, availability, honeypot flags |
| `jd_vector.npy` (768 float32) | ~3 KB | JD side of cosine, no model needed at rank time |
| `bge-base-en-v1.5/` (model weights) | ~440 MB | precompute + sandbox only (not the 100K rank step) |

All comfortably inside the 5 GB intermediate-disk cap `[SPEC]`. Artifacts ship in the repo (or via a documented regeneration script) so Stage-3 can reproduce the ranking step (`submission_spec.md` §10.3).

---

## 5. Scoring model — the ranking step in detail

The final score is a **gated, availability-modulated rubric**, computed in a fixed order so behavior is unambiguous and defensible.

### 5.0 Composition order `[CALL]` — precedence matters

```
base   = 0.65·career_title + 0.20·embedding + 0.10·skills + 0.05·education + logistics_modifier
gated  = apply_disqualifier_caps(base)          # most-restrictive cap wins; D1 ≈ hard floor
gated  = drop_honeypots(gated)                  # H1–H4 = HARD row-exclusion, before any further scoring
final  = gated × availability_multiplier         # over surviving rows; multiplier ∈ [~0.15, 1.0]
final  = make_distinct(final)                    # unique floats → then sort ↓
```

**Why this order:** caps apply to `base` *before* the multiplier, so a disqualified candidate cannot be rescued by high availability (the multiplier is ≤1). A genuine tier-5 who is a behavioral ghost is correctly multiplied *down* from a high base. Reversing the order lets an available-but-disqualified candidate float into the top 100 — a Stage-5 credibility failure. **Honeypots are dropped immediately after gating**, before the multiplier and sort: a dropped row cannot resurface wherever the drop sits, so the early position is a legibility/defensibility choice ("a honeypot never enters scoring") rather than a hole being closed — and it keeps this document, the implementation plan, and the code on one identical order.

### 5.1 Rubric base — career-first `[CALL]`, calibrated, not fixed

Starting weights, to be tuned by hand-ranking the 50-candidate sample (Phase-1 checkpoint), **not** sacred constants:

| Component | Weight | Source read | Notes |
|---|---|---|---|
| Career evidence + title fit | **65%** | `current_title` + `career_history[].description` | Highest signal; rule/keyword extraction for *built a real ranking/search/recommendation system*. **Never the skills list.** |
| Embedding / semantic | 20% | `cosine(JD, candidate)` | Plain-language tier-5 rescue (§3). |
| Skills list | 10% | `skills[]` | Near-noise; low weight by design. |
| Education / certs | 5% | `education[]`, `certifications[]` | Minor. |
| Logistics modifier | small | location/relocation, notice, work mode | Modifies, never dominates (§5.4). |

> **Primary calibration risk** `[CALL]`: the 65% career component is the highest-weighted part, the main keyword-stuffer defense, *and* the least-specified — it is a keyword/rule extraction over free-text descriptions, which can itself be gamed in the `description` field (though the title gate and the difficulty of stuffing prose guard it). The score is won or lost in calibrating this 65% against the hand-ranked sample. Do not under-invest here in favor of the 20% embedding.

### 5.2 Disqualifier gates — caps/floors, not subtractions `[SPEC language + CALL operationalization]`

Severity mirrors the JD's *actual* wording (getting this wrong is a Stage-5 hit). Only **D1** carries the literal "we will not move forward."

| # | Disqualifier | JD category | Operationalization `[CALL]` | Treatment |
|---|---|---|---|---|
| **D1** | Pure research, no production | **hard** ("we will not move forward") | research-only career, zero production in any role | **hard floor** (cap ≈ 0) |
| **D2** | "AI experience" only recent (<12 mo) LangChain→OpenAI | **conditional** | ML evidence confined to sub-12-mo current role, no pre-LLM ML in history | heavy penalty **unless** pre-LLM ML present |
| **D3** | Senior, no production code 18 mo | **firm** | current role architect/lead/manager-only, no hands-on IC code in 18 mo | heavy penalty |
| **D4** | Entire-career consulting | **"do NOT want"** + exception | **all** companies are consulting firms (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini + extensions) | **cap — NOT if any prior product role** |
| **D5** | CV/speech/robotics primary, no NLP/IR | **softer "do NOT want"** | primary expertise CV/speech/robotics, no NLP/IR evidence | strong penalty |

Caps are applied to `base`; the most restrictive firing cap wins. D4's product-company exception is part of its firing condition: current-consulting-with-prior-product does **not** fire.

### 5.3 Availability multiplier — continuous, gentle, ghost-harsh `[CALL, refined]`

Applied as a **multiplier on `gated`**, not a stack of independent penalties.

- **Healthy band:** ~**0.7–1.0** for engaged candidates.
- **Ghost floor:** drops steeply toward ~**0.15** only on the **conjunction** of: `last_active` stale **>180 days** AND `recruiter_response_rate` near-zero AND `open_to_work_flag = false`.
- **Continuous (not bucketed)** so near-identical **behavioral twins** separate on their signal delta alone.

**Why not a multiplicative stack** (`open × response × staleness`) `[DATA]`: the naive stack drives the **median candidate to 0.37×** and pushes **77% below 0.5×** — it flattens everyone instead of down-weighting the **3.4% ghosts**, and since NDCG@10 is half the score it actively demotes good fits. This is the single highest-leverage number in the ranker; calibrate the band and floor against **synthetic twins** — the sample has no tier-5 twin pair and no high-base ghost (§12) — before trusting a full run.

### 5.4 Logistics factors `[SPEC + DATA]`

Minor weights, also surfaced as honest concerns in reasoning (Stage-4 explicitly checks for "long notice period, wrong work mode"):
- **Location:** India-based **or** `willing_to_relocate` → full credit; international-not-relocating → discount (no visa sponsorship); mild bonus for Noida/Pune/Hyderabad/Mumbai/Delhi-NCR/Bangalore. (75.1% India-based `[DATA]`.)
- **Notice period:** short → small bonus; long → small penalty + honest-concern flag; **never disqualifying** (JD: a great fit with 120-day notice is still a great fit). (p50 = 90 days; 13.8% sub-30 `[DATA]`.)
- **Work mode:** remote-only vs. hybrid → minor honest-concern flag. (~uniform across modes `[DATA]`.)

### 5.5 Honeypot detection — four impossibility rules, hard-exclude `[CALL, refined]`

Flagged candidates are **removed from the candidate set** before sorting (not score-adjusted), keeping them out of the top 100.

| Rule | Test | Slack rationale |
|---|---|---|
| **H1** | a role's `duration_months` > months since its `start_date` | +3 mo absorbs date rounding |
| **H2** | summed role tenure > `years_of_experience`×12 by >30 mo | +30 mo allows concurrent roles |
| **H3** | `years_of_experience`×12 > total career span by >18 mo | +18 mo allows early internships |
| **H4** | a skill at {advanced, expert} with **0 `duration_months`** | maps to the spec's "expert proficiency, 0 years" example |

- **Catches 68 of ~80** `[DATA]` with near-zero collateral. The ~12 missed are the *"8 years at a company founded 3 years ago"* type — **undetectable**: the schema has no company founding dates and most employers are fictional (Hooli, Stark Industries, Dunder Mifflin). The top-150 audit (§7) re-checks founding dates only for **known real** companies, so it does **not** cover this residual (which is the fictional-employer cases). What actually bounds the risk is the arithmetic: a Stage-3 DQ needs **>10 honeypots in the top 100**, so the residual is only fatal if ~11 of the ~12 uncaught ones both survive to the shortlist *and* out-score genuine fits — improbable, since they carry no career-evidence advantage. State it honestly: ~12 uncatchable, low residual risk — *not* "the audit catches them."
- **The naive rule to AVOID** `[DATA]`: *"skill `duration_months` > career length = honeypot"* fires on **13,436 records (13.4%)**, catches only **5 of 68** real honeypots, and **would delete `CAND_0000031`** (verified: Recommendation Systems Engineer @ Swiggy, 6.0 YOE, Pinecone expert at 88 months — 7 years of normal pre-career vector-DB familiarity). Do not use it.

---

## 6. Ranking step — exact sequence

1. **Load artifacts** — candidate matrix (307 MB), feature store, JD vector. No model load.
2. **Cosine similarity** `cosine(jd_vector, candidate_matrix)` → `sim[100K]` (~25 ms via BLAS).
3. **Rubric base** (§5.1) — vectorized over 100K rows.
4. **Disqualifier caps** (§5.2) → `gated`.
5. **Honeypot exclusion** (§5.5) — drop H1–H4 flagged rows from the candidate set entirely (hard exclusion, before any further scoring).
6. **Availability multiplier** (§5.3) over the surviving rows → `final = gated × multiplier`.
7. **Distinct-float guarantee** (§8) — ensure no two `final` scores are exactly equal.
8. **Sort `final` descending → take top 100.** Sorting makes `score` non-increasing with rank automatically (validator requirement).
9. **Top-150 audit guard** (§7).
10. **Fact-grounded reasoning** for the top 100 (§7).
11. **Write CSV → run `validate_submission.py`.**

Steps 2–11 run in seconds (§10). The dominant cost lives in precompute.

---

## 7. Post-ranking guards and reasoning

### Top-150 audit guard `[CALL, new]`
After the shortlist, run an intensive consistency audit over the **top ~150** (cheap vs. 100K): re-run H1–H4, scrutinize any "too good to be true" profile, and check founding-date plausibility for *known real* companies (Swiggy, Razorpay, Paytm) whose founding dates are known. Plus a human eyeball before submission. **This is where the >10% disqualification risk actually lives** — the DQ fires on honeypots *in the top 100*, not in the 100K pool. The audit catches founding-date implausibilities at **known real** companies (which 100K-wide rules also miss, since the schema has no founding dates); the **fictional-employer residual remains uncatchable even here**, and is bounded instead by the >10-in-top-100 threshold (§5.5). Keep it **deterministic and logged**; the human eyeball is a *check*, never a manual CSV edit (manual edits between code and output are a Stage-3/4 red flag).

### Reasoning generation — fact-grounded templates, no LLM `[CALL, confirmed]`
3–4 structural variants, each pulling **different real facts** per candidate (years, title, named skills, signal values, the specific gap). Satisfies all six Stage-4 checks deterministically and is byte-stable for Stage-3 reproduction.
- **Why not a local LLM:** non-determinism (reproduction friction) + hallucination risk (the fastest Stage-4 fail) for marginal prose gain. Local LLM is *allowed* (local ≠ hosted API) but rejected on these grounds.
- **Honesty + rank-consistency policy:** every string = a real fact + a JD connection + an honest gap where one exists (long notice, wrong work mode, services background, international/visa, stale activity). Tone scales to rank — top-10 leads with strongest evidence (may note one caveat); bottom names the disqualifying gap plainly. Avoids the "templated, just inserts the name" penalty by varying structure *and* facts.

---

## 8. Validator-driven design `[SPEC]`

`validate_submission.py` enforces more than the headline rules; two behaviors shape the design:

- **Tie-break trap.** The spec text says break ties "by a secondary signal **or** `candidate_id` ascending," but the **validator code rejects** any equal-score adjacent pair where the better-ranked candidate has a *larger* `candidate_id`. → **Mitigation:** emit **continuous, distinct float scores** so no equal-score pairs exist and the check never fires; the model's true ordering stands. If an exact tie ever survives, sort that group `candidate_id`-ascending. (The validator *allows* ties on `score` — it only forbids `score` *increasing* with rank — so distinctness is a safety choice, not strictly required.)
- **Existence is server-side.** The validator checks only the `^CAND_[0-9]{7}$` pattern; existence in `candidates.jsonl` is enforced server-side. Moot here (we rank real records) — but never hand-edit IDs.

---

## 9. Sandbox vs. full-reproduction paths `[SPEC §10.5 + CALL]`

Two distinct execution paths, often conflated — separating them resolves the no-network/model tension:

| Path | Input | Embeddings | Model loaded? | Network |
|---|---|---|---|---|
| **Full ranking step** (Stage-3 reproduction, 100K) | precomputed matrix + JD vector | precomputed | **No** | Off |
| **Sandbox demo** (Stage-1 requirement, ≤100 sample) | ≤100 raw candidates | embedded **live** | Yes (bundled weights) | Off |

The sandbox must accept a ≤100-candidate sample, run end-to-end, and finish within budget — for 100 candidates, live bge-base embedding on CPU is a few seconds. **Deploy:** HuggingFace Spaces (Streamlit) primary + a `docker run` recipe in the README as backup (the spec accepts an unmodified `docker run` substitute). Doing both removes the single point of failure at Stage 1.

---

## 10. Budget accounting `[CALL]` — why the ranking step fits with margin

| Ranking-step operation | Time | Peak RAM |
|---|---|---|
| Load artifacts (≈307 MB matrix + features + JD vector) | ~1–2 s | ~0.4 GB |
| Cosine similarity 100K×768 (BLAS) | ~25 ms | + ~0.3 GB transient |
| Rubric + gates + multiplier (vectorized over 100K) | < 1 s | negligible |
| Honeypot exclusion + sort 100K + top-150 audit | < 0.5 s | negligible |
| Reasoning (100 template fills) + CSV write + validate | < 1 s | negligible |
| **Total** | **seconds** (vs. 5-minute cap) | **~1 GB** (vs. 16 GB cap) |

No model in this path → "network off" is satisfied structurally. The expensive operation (embedding 100K) is in precompute, which is uncapped. The margin is large by design, not by luck — and it is robust even if feature parsing were (suboptimally) moved into the ranking step (~20–30 s to re-stream 487 MB, still well inside budget).

---

## 11. Verification ledger — every parameter traced to source

| Architecture claim | Value | Source | Status |
|---|---|---|---|
| Composite metric | 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10 | `submission_spec` §4 | ✅ verified |
| Compute envelope | 5 min / 16 GB / CPU / no net / 5 GB disk | `submission_spec` §3 | ✅ verified |
| Output schema | `candidate_id,rank,score,reasoning`, 100 rows | `submission_spec` §2–3 | ✅ verified |
| Validator: ties allowed, `candidate_id`-asc enforced; score non-increasing | — | `validate_submission.py` | ✅ verified in code |
| Honeypots ~80, forced tier 0, >10% in top 100 = DQ at Stage 3 | — | `submission_spec` §7, §5 | ✅ verified |
| 5-stage pipeline + 6 Stage-4 reasoning checks | — | `submission_spec` §4–5 | ✅ verified |
| Max 3 submissions, last valid final, no live leaderboard | — | `submission_spec` §3, §8 | ✅ verified |
| 23 redrob signals | — | `redrob_signals_doc`, `candidate_schema.json` | ✅ verified |
| Embedding model / dim / matrix size | bge-base-en-v1.5 / 768 / 307 MB | `decision.md`, `context.md` §6–§7 | ✅ verified — `context.md` §6 reconciled to 768/307 MB this pass |
| Archetype CAND_0000031 | Rec-Sys Eng @ Swiggy, 6.0 YOE, Pinecone expert/88 mo | `sample_candidates.json` (real data) | ✅ verified against data |
| Honeypot H1–H4 catch 68/80; naive rule flags 13,436 | — | `context.md` §4.1 `[DATA]` | ⚠ `[DATA]` claim — relies on full 100K (not re-derivable from the 50-sample bundle); core logic confirmed by CAND_0000031 |
| Population/availability percentages | various | `context.md` §4 `[DATA]` | ⚠ `[DATA]` — taken as given from full-file pass |
| Between-submission tiebreaks | P@5 → P@10 → timestamp | `submission_spec` §4 | ✅ verified |
| File size as delivered | 487 MB (`README` says ~465 MB) | `context.md` §6 `[DATA]` / `README` `[SPEC]` | ✅ both noted |

**Corrections this document makes to `context.md`:** (1) embedding dim 768, matrix 307 MB, reconciled across §6/§7 (§3); (2) JD vector precomputed so the ranking step needs no model/network (§3, §6, §9); (3) explicit gate→honeypot→multiplier→distinct ordering (§5.0); (4) parsed features made an explicit precompute artifact (§4).

---

## 12. Known risks and data ceilings `[CALL]`

- **Top-10 ordering cannot be validated, only reasoned (highest).** NDCG@10 + NDCG@50 are 80% of the composite, won on the *order* of the genuine tier-5s. But the 50-sample contains **one** tier-5 (`CAND_0000031`) — re-derived on review — so calibration can separate the one fit from the 49 non-fits (trivial) yet **cannot** tune the top-10 ordering that decides the score. Treat the tier-5 ordering rules as a first-class, **written**, JD-traced deliverable (the Stage-5 substitute for validation), and exercise the rubric on **synthetic tier-5 variants** (perturbations of the archetype). The honest interview line is "top-10 order is reasoned, not tuned."
- **Career-rubric (65%) extraction:** keyword/rule over free-text descriptions — the main thing to hand-tune against the sample, and gameable in the `description` field (guarded by the title gate + the difficulty of stuffing prose). Most of the *separable* score lives here; invest here, not in the 20% embedding.
- **Disqualifier gates D2/D3/D5 can silently delete a top-10 candidate, and their false-positive rate is unmeasured.** A mis-firing D5 (a real NLP engineer who once shipped vision) or D3 (a Staff/Principal IC read as "manager-only" from the title) caps exactly the high-value candidates NDCG@10 rewards — the same archetype-deletion failure the skill-duration rule was rejected for. **Measure each gate's firing rate on the full 100K and inspect who it caps**; soften any gate that catches a plausible fit. D2 is the fuzziest (proxy: ML only in a sub-12-mo current role, no pre-LLM ML) — don't overclaim its precision.
- **Availability-multiplier calibration:** the one high-leverage number, but the sample has **no tier-5 twin pair and no high-base ghost** — only two low-tier ghosts (`CAND_0000029`, `CAND_0000043`), which confirm the floor *fires* but don't tune the case that matters (a strong candidate who is also disengaged). Calibrate the 0.7–1.0 band and ~0.15 floor against **synthetic twins** (the archetype cloned, identical but for the signal). Too steep → unseen NDCG@10 loss.
- **Reasoning template collisions:** 3–4 structural variants over 100 rows means similar candidates can collide into near-identical strings that read as the "name-insert templating" Stage-4 penalizes. A 5-string spot-read won't catch a collision between rows #37 and #61 — run a **pairwise near-duplicate check across all 100** and force divergence on any flagged pair.
- **Honeypot residual (~12 uncaught):** founding-date cases the schema can't represent (fictional employers); the top-150 audit covers only known real companies, so the residual stays uncatchable and is bounded by the >10-in-top-100 DQ threshold (§5.5), not by the audit. Stated honestly at Stage 5.
- **Parser field noise:** in the sample, `career_history[].description` and `.industry` don't reliably track the role title (a "Marketing Manager" with a mechanical-engineering description; repeated descriptions across roles). The parser reads these as clean signal for `product_vs_services` and `built_real_system` — **eyeball the parsed flags against raw text on the 50 before calibrating.**
- **False positives on genuine career-changers:** the dataset has real backend→ML transitioners; don't let keyword overlap or a single services employer cap them.
- **Weights are hand-tuned on a 50-sample, not learned:** defensible for this task — say so plainly; don't imply a trained learning-to-rank model.
- **`problemStatement.md` is a derivative:** always trace facts to the original organizer files (done in §11).

---

*Architecture derived from `context.md` and verified against the original organizer bundle (`job_description`, `submission_spec`, `README`, `redrob_signals_doc`, `candidate_schema.json`, `validate_submission.py`) and `sample_candidates.json`. Reference "now" for date math: 2026-05-27.*
