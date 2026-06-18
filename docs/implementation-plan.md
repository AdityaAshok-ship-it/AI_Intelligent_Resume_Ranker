# Redrob Ranking Challenge — Phase-Wise Implementation Plan

> **What this file is.** The executable build sequence that operationalizes the three design docs:
> - **`context.md`** — single source of truth (the *what* and *why*).
> - **`architecture.md`** — technical design (components, scoring composition, budget).
> - **`redrob_build_playbook.md`** — methodology, operating rules, understanding checkpoints.
>
> This plan adds the layer none of those carry on its own: the concrete build order, file-level tasks, the **artifact contracts** that flow between phases, the **calibration procedures**, and the **exit gate** for each phase. Nothing here overrides a decision in the design docs; it sequences them.
>
> **Conventions.** Reference "now" for all date math = **2026-05-27** (max `last_active_date`, **independently confirmed against the file**). Field paths use the *real nested schema* (`candidate_schema.json`), not the loose names used in architecture.md prose. Tags: `[SPEC]` organizer bundle · `[DATA]` measured on full 100K · `[CALL]` our decision.
>
> **Verification status (2026-06-15).** The full `candidates.jsonl` is now on disk and the design's `[DATA]` claims have been re-derived from it. The high-stakes numbers hold exactly (see §0.5 ledger). One real correction (notice-period) and a cluster of definition-sensitive `[CALL]` counts are flagged below and folded into the relevant phases.

---

## 0. Pre-flight — must be true before Phase 0 starts

These are blockers, not tasks. Do not begin precompute until all four hold.

| # | Pre-condition | Why it blocks | Status |
|---|---|---|---|
| P1 | **Full `candidates.jsonl` (100K) is on disk.** | The entire precompute stage embeds 100K texts and parses 100K feature rows. The 50-record `sample_candidates.json` is the **calibration/test set only** — it cannot stand in for the corpus. | ✅ **Satisfied** — 487,259,903 bytes (≈487 MB; README "~465 MB" is loose), 100,000 records, 0 duplicate IDs, structure matches schema. |
| P2 | **`bge-base-en-v1.5` weights available locally** (`HF_HUB_OFFLINE=1`, bundled path). | Precompute and the sandbox load the model; a Hub call is a no-network failure mode. The 100K ranking step loads no model. | Confirm weights cached + offline flag works. |
| P3 | **`validate_submission.py` runs in the env.** | Wired in from Phase 0; every CSV is validated before it leaves your machine. | `python validate_submission.py sample_submission.csv`. |
| P4 | **Repo + git + `decision.md` + `submission_metadata.yaml` initialized.** | The 50% (methodology) runs from commit 1; a flat git history is a Stage-4 elimination trigger. | First commit is the scaffold, not a code dump. |

---

## 0.5 Data verification ledger `[DATA, re-derived 2026-06-15]`

Every load-bearing `[DATA]` claim was recomputed directly from `candidates.jsonl`. This matters for two reasons: (1) the design rested on a prior pass nobody could re-check, and (2) Stage 5 fails on "contradicts the code/data" — so what you quote in the interview must match what the file says.

### Confirmed exactly — quote these with confidence

| Claim | Doc value | Re-derived | Note |
|---|---|---|---|
| Records / file size | 100K / ~487 MB | **100,000 / 487 MB** | 0 duplicate IDs |
| Reference "now" (max `last_active_date`) | 2026-05-27 | **2026-05-27** | anchors all date math |
| **Honeypots caught by H1–H4 (union)** | **68** of ~80 | **68** | the number the >10%-DQ insurance hinges on — exact |
| Ghosts (stale>120 ∧ rrr<0.15 ∧ not-open) | 3.4% | **3,372 / 3.4%** | small tail — the case for a *gentle* multiplier holds |
| Consulting-only careers | 9,745 / 9.7% | **9,745 / 9.7%** | exact |
| India-based | 75,113 / 75.1% | **75,113 / 75.1%** | exact |
| Staleness days p10/p50/p90/max | 20/105/206/240 | **20/105/206/240** | exact |
| `recruiter_response_rate` p10/p50/p90 | 0.14/0.44/0.73 | **0.14/0.44/0.73** | exact |
| `open_to_work` TRUE | 35.3% | **35.3%** | exact |
| `github_activity_score` == −1 | 64.6% | **64.6%** | exact |
| `willing_to_relocate` TRUE | 28.8% | **28.8%** | exact |
| Work mode | ~uniform | **25.0/25.1/24.9/25.0%** | exact |
| Job-hoppers | 1,707 of 57,306 (3+-role pool) | **1,707 of 57,306** | denominator = candidates with ≥3 roles — confirmed |
| **Archetype `CAND_0000031`** | Rec-Sys Eng @ Swiggy, 6.0 YOE, Pinecone expert @ 88 mo | **exact** | + `open=true`, `rrr=0.91`, last-active 3 days stale → availability ~0.95 confirmed |
| Naive rule **deletes the archetype** | yes | **yes** | the single sharpest argument for rejecting it — confirmed |

### One real correction — fix before Phase 1

- **Notice period "13.8% sub-30" is wrong as written.** Strictly `<30` is **22 candidates (0.0%)**. The 13.8% are at **exactly 30** (`<=30` = 13,809 / 13.8%). Notice is **quantized** to {0, 15, 30, 45, 60, 90, 120, 150}, mode **90** (31%), then 60 (24%), 120 (18%), 30 (14%), 150 (13%). **Implication:** any "short-notice bonus" keyed on `notice < 30` fires on ~nobody — a dead branch. Operationalize the bonus on **`<=30`** (the populated buyout tier) and treat 120/150 as the honest-concern band. (Fixed in Phase 0 parser + Phase 1 logistics below.)

### Definition-sensitive — do NOT quote the context.md figure; re-derive from your shipped detector

These depend on `[CALL]` keyword/rule choices. My independent reproduction lands in the same ballpark but not on the doc's exact count, which is expected — there is no canonical definition. **Quoting "3.6% keyword-stuffers" at Stage 5 while your final detector flags 4.7% is a self-contradiction.** Lock each definition in code, then quote *that* number.

| Claim | Doc | My reproduction | Why it differs |
|---|---|---|---|
| Eng/technical titles | 42.7% | 41.8% | title keyword list |
| Keyword-stuffers | 3.6% | 4.7% | AI-skill list + "no career evidence" test |
| Tier-5 pool | 374 (244 avail) | 287 (192 avail) | "discipline title" + "built-a-system" evidence regex |
| Naive-rule false-positive count | 13,436 / 13.4% | 18,588 / 18.6% | definition of "career length" in the comparison |

The naive-rule discrepancy does **not** weaken the decision to reject it: at either count it deletes 13–19k legitimate people, the archetype included, while adding **zero** honeypots H1–H4 doesn't already catch. The strategic conclusion is direction-stable.

### New findings the docs never flagged

1. **Data is pristine on every parser-critical field.** Zero nulls in `current_title`, `career_history` (all ≥1 role), `summary`, `skills` (all ≥1), and all role/skill `duration_months`. → **The Phase 0 parser needs no defensive null-handling on dense fields.** The *only* sparse fields are `skill_assessment_scores` (present for 24%), `github_activity_score` (−1 for 65%), and `offer_acceptance_rate` (−1 sentinel) — any rule touching these must handle absence explicitly.
2. **Honeypots wear attractive titles.** `CAND_0010770` is titled **"Recommendation Systems Engineer"** — identical to the archetype — and is caught **only by H3 (timeline math)**, not by title or keywords. Others in the 68 include "Frontend Engineer", ".NET Developer", "Business Analyst", "Graphic Designer". → Hard confirmation that honeypot exclusion **must** be a pre-sort hard gate; the rubric alone cannot see these, and a keyword-stuffed honeypot with a perfect title will score high. This is the strongest single piece of evidence for Phase 2 and sharpens the top-150-audit rationale.
3. **`career_history` length** runs 1–9 (schema allows 10); **18.5% have a single role.** Single-role candidates are the main H3 population (span = current tenure; inflated YOE trips it). The H3 slack (+18 mo) is what keeps legitimate single-role people out.
4. **The calibration set is a trap-detection set, not a ranking set `[DATA]`.** The 50-sample contains **exactly one** genuine tier-5 (`CAND_0000031`; all 23 eng-titled candidates were checked — only it shows real ranking/retrieval evidence), **no tier-5 twin pair**, and **no high-base ghost** (only two low-tier ghosts, `CAND_0000029` and `CAND_0000043`). It can separate the one fit from 49 non-fits and exercise the trap/disqualifier gates, but it **cannot** calibrate top-10 *ordering* (1 tier-5) or the availability multiplier on a *strong* candidate (no twin, no high-base ghost) — the two things the score is actually won on. The sample's keyword-stuffer is `CAND_0000021` (Project Manager, AI buzzwords in skills *and* summary, no AI career evidence). Also: `career_history[].description`/`.industry` do **not** reliably track the role title in the sample (mismatched and repeated descriptions), so the parser's `product_vs_services`/`built_real_system` flags run on noisier text than assumed. → Reframes Phase 1/Phase 3 calibration (below) and Phase 0's parser exit gate.

---

## 1. Target repo layout (end state)

Flat and legible — a traceable pipeline, not a package (`context.md` §7).

```
redrob-ranking/
├── precompute.py          # offline, uncapped: embeddings + feature store + JD vector
├── rubric.py              # base score composition (career-first)
├── detectors.py           # H1–H4 honeypots, keyword-stuffer, twins, top-150 audit
├── reasoning.py           # fact-grounded templates + 6-check self-test
├── rank.py                # in-budget ranking step → top-100 CSV (single reproduce cmd)
├── calibrate.py           # (or a notebook) hand-rank harness + weight tuning on the 50-sample
├── artifacts/
│   ├── candidate_matrix.npy     # 100K×768 float32 (~307 MB)
│   ├── candidate_ids.npy        # row-index → candidate_id map (ALIGNMENT CONTRACT)
│   ├── features.parquet         # parsed per-candidate features (< ~100 MB)
│   └── jd_vector.npy            # 768 float32 (~3 KB)
├── sandbox/app.py         # Streamlit demo (≤100 sample, embeds live)
├── README.md              # single reproduce command
├── requirements.txt
├── decision.md            # running decision log (most important file)
├── submission_metadata.yaml
└── validate_submission.py # organizer-provided, unmodified
```

---

## 2. Artifact contracts (the glue between phases)

Make these explicit *before* writing scorers — every downstream phase reads them. Getting the alignment contract wrong silently corrupts every rank.

- **`candidate_ids.npy`** — the ordered list of `candidate_id`s. **Row `i` of `candidate_matrix.npy` ⇔ `candidate_ids[i]` ⇔ the feature row keyed by that id.** This is the single most important invariant in the build; assert it at load time in `rank.py`.
- **`candidate_matrix.npy`** — `float32`, shape `(100000, 768)`. Optionally L2-normalized at write time so cosine reduces to a dot product.
- **`features.parquet`** — one row per candidate, keyed by `candidate_id`. Columns enumerated in §Phase-0. Pandas/PyArrow; small.
- **`jd_vector.npy`** — `float32`, shape `(768,)`. Normalized consistently with the matrix.

> **Rule:** scorers (`rubric.py`, `detectors.py`) consume `features.parquet` columns by name and the matrix by row index. They never re-parse JSON. If a scorer needs a field that isn't a column, the fix is to add it to the parser, not to read the corpus at rank time.

---

## Phase 0 — Foundations & precompute infrastructure
*Part of the 50%. Sets the evidence trail from commit 1; builds the artifacts everything else consumes.*

**Objective:** clean repo, streaming loader, the deterministic feature parser, the embedding precompute, JD vector, validator wired in.

### Tasks
1. **Scaffold + git + empty `decision.md` + metadata from template.** First commit.
2. **Streaming loader** `[CALL]` — read `candidates.jsonl` line-by-line; never build a 100K list of dicts. (487 MB as delivered `[DATA]`; README quotes ~465 MB `[SPEC]`. A full in-memory parse risks the 16 GB ceiling.)
3. **Deterministic feature parser** (no model) — the heart of Phase 0. Emits one `features.parquet` row per candidate. Enumerated below.
4. **Embedding precompute** — build candidate text (high-signal fields only), embed with bge-base, append to the `float32` matrix; write `candidate_matrix.npy` + `candidate_ids.npy` in lockstep.
5. **Embed the fixed JD once** → `jd_vector.npy`.
6. **Validator wired into the workflow** (`Rule 4`) — a throwaway 100-row CSV must pass before Phase 0 closes.

### Feature parser specification (exact schema paths)

**Embedding text builder** `[CALL]` — concatenate **`profile.current_title` + `profile.headline` + `profile.summary` + each `career_history[].description`**. **Exclude `skills[]`** — it is the keyword-stuffer attack surface (§5.1 architecture).

| Feature group | Columns (→ from schema path) | Use |
|---|---|---|
| **Identity** | `candidate_id` | join key |
| **Title / career evidence** | `is_eng_title` (← `profile.current_title`), `built_real_system` flag (regex/keyword pass over `career_history[].description` for shipped ranking/search/recommendation/retrieval systems — **never `skills[]`**), `product_vs_services` (← `career_history[].industry` + company-name list: TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/Mindtree/HCL/Tech Mahindra/Mphasis = services) | 70% rubric component (§Phase-1) |
| **Recency (D3 input)** | `hands_on_code_18mo` (current role is IC-engineering, not architect/lead/manager-only, within 18 mo of 2026-05-27) | disqualifier gate |
| **Disqualifier inputs D1–D5** | D1: research-only career, zero production · D2: ML evidence confined to sub-12-mo current role, no pre-LLM ML in history · D3: see recency · D4: **all** companies consulting (with prior-product exception flag) · D5: primary CV/speech/robotics, no NLP/IR evidence | gates/caps (§Phase-1) |
| **Availability (ghost conjunction)** | `staleness_days` (← 2026-05-27 − `redrob_signals.last_active_date`), `recruiter_response_rate`, `open_to_work_flag` | availability multiplier (§Phase-3) |
| **Logistics** | `country` / `profile.location`, `willing_to_relocate`, `notice_period_days` (**quantized {0,15,30,45,60,90,120,150}; bonus keyed on `<=30` — `<30` is empty, see §0.5**), `preferred_work_mode` | small modifier + honest-concern flags |
| **Honeypot inputs H1–H4** | per-role `duration_months` vs months-since-`start_date`; Σ`duration_months` vs `profile.years_of_experience`×12; `years_of_experience`×12 vs career span (earliest `start_date`→now); any `skills[]` at `proficiency`∈{advanced,expert} with `duration_months`==0 | hard-exclude (§Phase-2) |

> **Null-handling reality `[DATA]`:** every field above except three is **dense (zero nulls across 100K)** — no defensive coding needed on `current_title`, `career_history`, `summary`, `skills`, or any `duration_months`. The three sparse fields needing explicit absence handling: `skill_assessment_scores` (present 24%), `github_activity_score` (−1 for 65%), `offer_acceptance_rate` (−1 sentinel). Don't let a sparse-field rule silently coerce a sentinel into a real value.

**Exit gate (checkpoint, `Rule 1`):**
- Stream all 100K under 16 GB; `features.parquet`, `candidate_matrix.npy`, `candidate_ids.npy`, `jd_vector.npy` all materialized.
- Assert row-alignment invariant holds.
- **Eyeball the parsed flags on the 50-sample:** dump `product_vs_services` and `built_real_system` next to the raw title/description/industry for all 50 and confirm they aren't corrupted by the sample's title↔description mismatches (§0.5 finding 4). If a flag disagrees with what a human reads, the *extraction* is wrong — fix it here, before it poisons calibration.
- You can **explain plainly** why streaming beats loading 465 MB into a list, and **predict** what RAM does if you don't.
- `decision.md`: embedding-model choice (bge-base over bge-small/MiniLM — quality, precompute is uncapped), repo structure.

---

## Phase 1 — The relevance rubric + calibration (THE differentiator)
*The 30%. Pure judgment. This is what separates you from every generic-ranker team.*

**Objective:** a written model of what Redrob's *hidden* ground truth rewards — reverse-engineered from the JD's subtext — implemented as the base scorer, then **calibrated against the 50-sample**.

> **Sequencing note (deviation from the playbook, deliberate):** the playbook has Phase 1 produce a *spec only* and defers code to Phase 3. But you cannot calibrate the 20% embedding component or the availability multiplier (against synthetic twins — the sample has no strong twin pair; §0.5 finding 4) without the parser + sample-embedding existing. So Phase 0's parser runs first, and Phase 1 calibration is **empirical**, not paper. The rubric *spec* still comes first within this phase.

### Tasks
1. **Write the tiering rubric** (tier 5 perfect-fit → tier 0 honeypot/DQ) in plain language. This is the most-cited artifact at Stage 5.
2. **Implement `rubric.py` base scorer** with the starting weights — **calibration targets, not constants:**
   ```
   base = 0.70·career_title + 0.20·embedding + 0.10·skills + logistics_modifier   # education dropped 2026-06-18 (no JD basis)
   ```
3. **Define the composition order** `[CALL]` (precedence matters — **aligned with architecture.md §5.0**; honeypots are excluded right after gating, before the multiplier):
   ```
   gated = apply_disqualifier_caps(base)      # most-restrictive cap wins; D1 ≈ hard floor
   gated = drop_honeypots(gated)              # H1–H4 = HARD row-exclusion, before any further scoring
   final = gated × availability_multiplier    # over surviving rows; multiplier ∈ [~0.15, 1.0]
   final = make_distinct(final)               # unique floats → then sort ↓
   ```
   Two ordering rules, each load-bearing for a different reason:
   - **Caps before the multiplier** — caps apply to `base`, and the multiplier is ≤1, so an available-but-disqualified candidate can never be lifted back up. (This is the one that changes scores; reversing it is a Stage-5 credibility failure.)
   - **Honeypots excluded immediately after gating** — a flagged candidate is dropped from the set entirely, so it never reaches the multiplier, distinct-float, or sort. A dropped row cannot resurface no matter where the drop sits, so this is a **legibility/defensibility** choice rather than a hole being closed: "a honeypot never enters scoring" is cleaner to narrate and audit than "a honeypot is scored, then removed before sort." Outcome is identical; the early position is easier to defend cold.
4. **Operationalize disqualifiers as gates/caps** (D1–D5, §2 context) with severity matched to JD wording — only D1 carries "we will not move forward"; D4 fires only on *entire-career* consulting (prior-product exception).
5. **Calibrate** (the actual work — see procedure below).

### Calibration procedure (do not skip; do not hand-wave)

> **What calibration here can and cannot do `[DATA]`.** The 50-sample has **one** tier-5 (`CAND_0000031`) — §0.5 finding 4. So hand-ranking can teach the rubric to *separate* the one fit from 49 non-fits (easy; any career-first rubric does it on the first pass) but **cannot** tune the top-10 *ordering* among genuine tier-5s, which is 80% of the composite. There is no in-loop ground-truth signal anywhere (no leaderboard). Conclusion: stop treating calibration as de-risking the score. The top-10 ordering is **reasoned, not tuned**, and its only Stage-5 backstop is the **written tier-5 ordering rules** (step 1b). Build those; don't expect the sample to validate them.

1. **Hand-rank the 50-sample** using the *rubric document alone* — before looking at scorer output.
   - **1b. Write the tier-5 ordering rules explicitly** (JD-traced): given two genuine tier-5s, what breaks the tie — product-scale of the shipped system, eval-framework depth, recency of hands-on ranking work, availability? This document *is* the substitute for the validation the sample can't provide, and it is the answer to the hardest Stage-5 question ("how do you know your top-10 order is right?").
2. Run `rubric.py` over the 50; compare its ordering to your hand-ranking.
   - **2b. Exercise the rubric on synthetic tier-5 variants.** Since real tier-5s are absent bar one, construct 3–4 by lightly perturbing the archetype (swap company, vary YOE within 4–11, add/remove one signal) and confirm the rubric orders them the way your step-1b rules say it should. This is the only way to test top-10 ordering on this data.
3. Where they diverge, decide whether the *rubric* or the *weight* is wrong; adjust; **log every change in `decision.md`** with reasoning.
4. **Hold out ~10–15 of the 50 as a check set.** Tune on the rest; confirm held-out ordering also improves. Note what this does and doesn't guard: it catches overfitting your weights to your *own* hand-ranks — it does **not** guard against your hand-ranks themselves diverging from the hidden labels (nothing can, here). Stop when it stops improving — do not chase the last point (there is no leaderboard to catch overfitting; see §Submission-strategy).
5. **Measure the disqualifier gates before trusting them `[DATA]`.** Compute the firing rate of D2/D3/D5 on the full 100K and inspect a sample of who each one caps. A false-positive gate silently deletes a top-10 candidate (a mis-firing D5 on a real NLP engineer who once shipped vision; a D3 on a Staff/Principal IC read as "manager-only" by title) — the same archetype-deletion failure that killed the skill-duration rule, but unmeasured. Soften any gate that catches a plausible fit.
6. Primary calibration risk `[CALL]`: the **70% career component** is highest-weighted, the main keyword-stuffer defense, *and* the least-specified. Most of the *separable* score lives here — invest here, not in the 20% embedding.

**Exit gate (checkpoint):** (a) you can hand-rank 5 sample candidates from the rubric alone, explain each placement, and a second reader reproduces it; (b) the rubric ranks `CAND_0000031` above every distractor *for the right reason* (career evidence, not a cosine/keyword accident) and orders the synthetic tier-5 variants per your written rules; (c) the written tier-5 ordering rules exist in the repo; (d) D2/D3/D5 firing rates measured and inspected. Calibrated weights + every change logged in `decision.md`.

---

## Phase 2 — Trap & honeypot detection
*The 30%. Where binary disqualification and blind-scoring risk live. A naive cosine ranker walks into every trap.*

**Objective:** independently testable detectors for every documented trap, each tested on the 50-sample.

### Build order (by risk)
1. **Honeypot detector `detectors.py` — FIRST (binary DQ insurance).** Five impossibility rules; flagged candidates **removed from the set before sort** (not score-adjusted):

   | Rule | Test | Slack |
   |---|---|---|
   | **H1** | a role's `duration_months` > months since its `start_date` | +3 mo (date rounding) |
   | **H2** | Σ role tenure > `years_of_experience`×12 by >30 mo | +30 mo (concurrent roles) |
   | **H3** | `years_of_experience`×12 > total career span by >18 mo | +18 mo (early internships) |
   | **H4** | a skill at {advanced, expert} with `duration_months`==0 | maps to spec's "expert, 0 years" |
   | **H5** | a role at a **real** company starts before that company's founding year | fictional/unknown companies skipped per-role |

   **Verified `[DATA, 2026-06-15]`:** H1–H4 catch **exactly 68** records (per-rule: H1≈19, H2≈22, H3≈25, H4≈21). **H5 catches 250 additional** using a hardcoded founding-year map for the 55 real corpus companies (8 fictional are exempt). **Total: 318 honeypots.** Near-zero collateral throughout. All 5 binding companies (CRED 2018, Krutrim 2023, Sarvam AI 2023, Glance 2019, Rephrase.ai 2019) verified against primary/first-party sources only.

   **Critical finding — honeypots wear attractive titles:** one H3 honeypot is titled "Recommendation Systems Engineer" (identical to the archetype) — caught only by timeline math, not by title or keywords. Exactly why honeypot exclusion is a **pre-sort hard gate**, not a score penalty.

   **Explicitly DO NOT use** the naive "skill `duration_months` > career length" rule (fires 13–19% of the pool, adds **zero** honeypots over H1–H4, and **deletes `CAND_0000031`**). The ~12 that remain uncaught are "8 years at a company founded 3 years ago" at **fictional** employers — undetectable (no founding dates for fictional companies). The top-150 audit covers only known real companies, so it does NOT cover this fictional-employer residual; it's bounded by the >10-in-top-100 DQ threshold.

1a. **Skill-anachronism gate** — hard-exclude if a candidate claims a named, datable technology for more months than it has existed (+6 mo slack). Gates 15 technologies (LoRA, QLoRA, PEFT, RAG, LangChain, LlamaIndex, Pinecone, Milvus, Qdrant, Weaviate, pgvector, Sentence-Transformers, Haystack, OpenSearch, HF Transformers); all inception dates primary-sourced. Generic concepts and pre-2018 tools are excluded to avoid false positives. **Impact: 955 résumés.**

1b. **Education-integrity gate** — hard-exclude on internally-impossible degree timelines: (a) higher degree ends before lower starts, (b) Bachelor's concurrent with PhD (rank-gap ≥ 2), (c) degree ends before it starts. Degree names never judged. **Impact: 7,967 résumés.**

   Combined with H1–H5: **9,108 of 100,000 excluded** (90,892 surviving).
2. **Keyword-stuffer detector** — non-eng title + ≥3 AI skills, no career evidence (**~3.6–4.7%, definition-sensitive — see §0.5; lock the AI-skill list in code and quote that count, not the doc's**). **Threshold held at ≥3 by decision** (lowering to ≥2 adds only 166 candidates, disproportionately genuine 2-skill transitioners — the modal count for real career-changers — for zero ranking gain since the flag is cosmetic; revisit against ranked results during calibration before locking). **Flag, do not double-penalize** — career-first weighting already demotes them; aggressive penalty false-positives genuine career-changers.
3. **Plain-language tier-5 rescue** — the 20% embedding + a career-evidence keyword pass surfaces candidates who *built* the right systems without buzzwords. A *rescue*, not a penalty.
4. **Behavioral twins** — handled by the continuous availability multiplier (Phase 3), not a bolt-on. No separate rule.
5. **Top-150 audit guard** `[CALL]` — after the shortlist, re-run H1–H4 over the top ~150, scrutinize "too good to be true" profiles, check founding-date plausibility for *known real* companies (Swiggy, Razorpay, Paytm). **This is where the >10% DQ risk actually lives** (the rule fires on honeypots *in the top 100*) — and §0.5 confirms it is real: a honeypot titled "Recommendation Systems Engineer" exists in the pool and only timeline math catches it, so a near-miss could surface in your shortlist. (The fictional-employer founding-date residual stays uncatchable even here; it is bounded by the >10-in-top-100 threshold, not the audit.) Deterministic + logged; the human eyeball is a *check*, never a manual CSV edit.

**Exit gate (checkpoint):** for any given sample candidate you can state which detectors it trips and why. Threshold choices logged with the false-positive/false-negative tradeoff in `decision.md`.

---

## Phase 3 — The ranker
*The 20%. The easy, solved-shape part. Do not over-invest here.*

**Objective:** combine rubric (P1), traps (P2), and the availability multiplier into a valid top-100 CSV, in budget, behind one reproduce command.

### `rank.py` — exact sequence (matches architecture §6; honeypot exclusion precedes discrepancy gates and the multiplier)
1. **Load artifacts** — matrix (307 MB), feature store, JD vector. **No model load.** Assert row-alignment.
2. **Cosine** `cosine(jd_vector, candidate_matrix)` → `sim[100K]` (~25 ms via BLAS).
3. **Rubric base** (§Phase-1) — vectorized over 100K.
4. **Disqualifier caps** (D1–D6 + visa cap) → `gated`.
5. **Honeypot exclusion** — drop H1–H5 flagged rows from the set entirely (hard exclusion, before any further scoring).
6. **Discrepancy gates** — drop skill-anachronism and education-integrity flagged rows. `surviving = ~honeypot & ~anachronism & ~education`.
7. **Availability multiplier** over the surviving rows → `final = where(surviving, gated × multiplier, −∞)`.
8. **Sort `final` ↓ → top 150** (audit window). Sorting makes `score` non-increasing with rank automatically.
9. **Distinct-float guarantee** — apply post-sort on the top-150 window: descending offsets `(150 − k) × 1e-9` so the best candidate keeps the largest offset and sort order is preserved (EC-47, EC-48). Post-sort to avoid the pre-sort risk of flipping pairs closer than `(max_index_gap × ε)`.
10. **Top-150 audit guard** (§Phase-2) → **take top 100.**
11. **Fact-grounded reasoning** for the top 100 (§Phase-4).
12. **Write CSV → run `validate_submission.py`.**

### Availability multiplier `[CALL, refined]` — the highest-leverage number
- Applied as a **multiplier on `gated`**, not a stack of independent penalties.
- **Healthy band ~0.7–1.0** for engaged candidates.
- **Ghost floor ~0.15**, only on the **conjunction**: `staleness_days` > **120** **AND** `recruiter_response_rate` near-zero **AND** `open_to_work_flag` == false. (120-day threshold; 180-day was too strict, targeting only 0.8% vs. the measured 3.4% ghost population.)
- **Continuous (not bucketed)** so behavioral twins separate on their signal delta alone.
- **Why not a multiplicative stack** `[DATA]`: `open × response × staleness` drives the **median to 0.37×** and pushes **77% below 0.5×** — it flattens everyone instead of down-weighting the **3.4% ghosts**, and since NDCG@10 is half the composite it actively demotes good fits. **Calibrate the band and floor against synthetic twins before trusting a full run** — the 50-sample has no tier-5 twin pair and no high-base ghost (only two low-tier ghosts; §0.5 finding 4), so clone the archetype into a twin pair identical but for the availability signal and confirm the ghost lands below the engaged twin without dropping below an unrelated weaker-but-engaged candidate. Log the chosen band/floor and this test in `decision.md` — it's the only evidence the number was reasoned.

### Validator traps to build around `[SPEC, verified in code]`
- **Tie-break trap:** the validator rejects any equal-score adjacent pair where the better-ranked candidate has a *larger* `candidate_id`. → distinct floats make the check never fire; fallback is `candidate_id`-ascending within any surviving tie.
- **Score non-increasing by rank** is enforced — sorting in step 8 satisfies it structurally.

### Budget check (must hold; architecture §10)
Load ~1–2 s · cosine ~25 ms · rubric/gates/multiplier <1 s · exclusion+sort+audit <0.5 s · reasoning+write+validate <1 s → **seconds vs. 5-min cap**, **~1 GB vs. 16 GB cap**. No model in this path → "network off" satisfied structurally.

**Exit gate (checkpoint — the single most likely interview question):** trace one candidate from raw JSON → final rank, narrating every transformation. The reproduce command runs end-to-end in budget.

---

## Phase 4 — Reasoning generation
*Engineering + Stage-4 prep.*

**Objective:** 100 reasoning strings that pass all six Stage-4 checks: specific facts · JD connection · honest concerns · no hallucination · variation across rows · rank-consistency.

### `reasoning.py` `[CALL, confirmed]`
- **Fact-grounded templates, no LLM.** 3–4 structural variants, each pulling **different real facts** per candidate (years, title, named skills, signal values, the specific gap). Deterministic (Stage-3 reproducible) and hallucination-proof.
  - *Why not a local LLM:* non-determinism (reproduction friction) + hallucination risk (the fastest Stage-4 fail) for marginal prose gain. Local LLM is *allowed* (local ≠ hosted API) but rejected on these grounds.
- **Honesty + rank-consistency policy:** every string = a real fact + a JD connection + an honest gap where one exists (long notice, wrong work mode, services background, international/visa, stale activity). Tone scales to rank — top-10 leads with strongest evidence (may note one caveat); bottom names the disqualifying gap plainly.
- **Self-check:** score your own 100 outputs against the six criteria before submission. **Include a pairwise near-duplicate check across all 100** (n-gram overlap or edit distance between every pair): 3–4 structural variants over 100 rows guarantees that similar candidates collide into near-identical strings that read as the "name-insert templating" Stage-4 penalizes. Flag any pair above a threshold and force a structural divergence — a 5-string spot-read cannot catch a collision between rows #37 and #61.

**Exit gate (checkpoint):** (a) read 5 generated reasonings; each cites a *real* fact from that candidate's profile (no hallucination); (b) the pairwise duplicate check passes — no two of the 100 are near-identical.

---

## Phase 5 — Hardening, sandbox, metadata
*The 50%. The mandatory pieces most teams scramble on.*

**Objective:** survive Stage 1 (format + mandatory sandbox) and Stage 3 (clean reproduction).

### Checklist
- **Local reproduction test** — ranking step runs end-to-end in ≤5 min, ≤16 GB, no network. Confirmed on your machine.
- **Sandbox deployed** `[SPEC §10.5 + CALL]` — **HuggingFace Spaces (Streamlit) primary + a `docker run` recipe in the README as backup** (the spec accepts an unmodified `docker run` substitute). The sandbox accepts a ≤100-candidate sample and embeds **live** (loads bundled weights; a few seconds on CPU). Doing both removes the single point of failure at Stage 1. *Missing sandbox is flagged at Stage 1 — do not let it ambush you at the end.*
- **`submission_metadata.yaml`** complete and honest, including the **AI-tools declaration** (`[SPEC §10.4]` — declared use is not penalized; *contradicting your interview* is).
- **README** with the single reproduce command.
- **Final `decision.md` pass** — read end to end; it should tell the whole story.

**Exit gate (checkpoint):** you run the reproduce command yourself and watch it finish in budget.

---

## Phase 6 — Interview consolidation
*The 50%. The actual win condition (Stage 5 = 30-min live defense).*

**Objective:** defend every choice cold, without the code in front of you.

### Drills
- **Marketing-manager test** — explain each component in plain language.
- **Predict-the-output** — given candidates, call the ranks before running.
- **"Why not X" gauntlet** — defend each decision from `decision.md`.
- **Walk the pipeline cold** — one candidate, raw JSON → final rank, no notes.

### The eight defense anchors (answer cold; each traces to a decision + a `decision.md` entry — `context.md` §10)
1. Walk one candidate raw JSON → final rank (pre-rehearse `CAND_0000031` + one mid-pack).
2. Why a *gentle continuous* availability multiplier, not a multiplicative crush? (3.4% ghosts; a crush flattens the median to 0.37× and demotes good fits.)
3. Why reject the skill-duration honeypot rule? (Fires on 13–19% of the pool, adds zero honeypots over H1–H4, and deletes the archetype `CAND_0000031` — verified.)
4. Why does career history outrank the skills list? (The JD's explicit thesis + the keyword-stuffer trap.)
5. Why fact-grounded templates over an LLM? (Reproducibility + hallucination risk vs. marginal prose gain.)
6. How do two near-identical candidates rank differently? (The continuous multiplier separates twins by signal delta.)
7. You catch only 68 of ~80 honeypots — why acceptable? (68 confirmed exact; six more signatures tested find nothing clean; the residual is founding-date cases the schema can't represent. The top-150 audit covers only known real companies; the fictional-employer residual stays uncatchable but is bounded by the >10-honeypots-in-top-100 DQ threshold — improbable for ~11 of ~12 to both surface and out-score real fits.)
8. Your top-10 ordering is 80% of the score and the 50-sample has **one** tier-5 — how do you know the order is right? (You can't validate it empirically; no team can without the hidden labels. The order is **reasoned**, not tuned: here are the written, JD-traced tier-5 ordering rules — product-scale of the shipped system, eval-framework depth, recency, availability — and the synthetic-variant tests that exercise them. That honest answer is stronger than pretending the sample validated the order.)

> **Defensibility note on numbers `[DATA]`:** §0.5 splits the stats into *confirmed-exact* (quote freely: 68 honeypots, 3.4% ghosts, the archetype facts, ref date, all signal percentiles) and *definition-sensitive* (re-derive from your shipped code before quoting: eng-title %, keyword-stuffer %, tier-5 pool size, naive-rule false-positive count). Two figures inherited from the docs are **form-dependent and not independently verifiable from the file** — the multiplicative-stack "median → 0.37×, 77% < 0.5×" depends on the exact penalty curves you compare against. In the interview, present those as *illustrative of the collapse direction* (which the 3.4% ghost rate + 0.44 median response rate fully support), not as exact measurements — or re-derive them from the specific stack form in `decision.md`. Stage 5 fails on "contradicts the code"; an unverifiable precise number is a self-inflicted wound.

**Exit gate:** you pass the gauntlet cold. If you can, the interview is a formality.

---

## 3. Submission strategy `[SPEC + CALL]` — because there is no leaderboard

**`max 3 submissions · no live leaderboard · last valid submission is final` `[SPEC]`.** This inverts normal Kaggle instincts: **a submission buys you no scoring feedback** — there is nothing to learn between attempts. The only thing a submission confirms is **Stage-1 format validity + server-side ID existence**.

Therefore:
- **All score-improving work happens offline**, against the 50-sample with a held-out check (§Phase-1). The hidden ground truth gives zero in-loop signal.
- **Submission 1:** once local calibration + the top-150 audit are clean — purely to confirm the format passes server-side (validates that your IDs exist in the corpus). Not a probe; there's no probe value.
- **Submissions 2–3:** reserve. Use only if you find a genuine defect after submission 1. The **last valid** one stands.
- **Never spend a submission to "see how it does."** You can't see.

---

## 4. Critical path — where effort actually flows

The 20/30/50 split (ranker / rubric+traps / engineering+methodology) is **not** sequential — the 50% runs through everything via the operating rules. The genuine critical path, by risk:

**data acquisition (P0 ✅ done) → feature parser (P0) → rubric calibration (P1) → availability-multiplier calibration (P3)**

Everything else — the cosine multiply, the sort, the templates, the sandbox — is solved-shape and low-risk. But note what the two calibration steps can and can't do (§0.5 finding 4, Phase 1): with **one** tier-5 in the sample and no strong twin/ghost, they can be *executed* but not *validated* — the top-10 ordering and the multiplier's effect on strong candidates are **reasoned, not tuned**. So the real critical path is: **(a) principled, written tier-5 ordering rules, (b) measured D2/D3/D5 gates, (c) the 70% career extractor** — the places where reasoning quality and false-positive control actually move (or protect) the score. The 20% mechanics remain the place *not* to spend time.

---

## 5. Phase summary

| Phase | Deliverable | Exit gate |
|---|---|---|
| **0** | repo, streaming loader, feature parser, matrix + ids + JD vector, validator wired | stream <16 GB; artifacts aligned; parser flags eyeballed on the 50; explain streaming |
| **1** | rubric doc + `rubric.py` + calibrated weights + written tier-5 ordering rules | hand-rank 5 reproducibly; tier-5 ordering rules written; D2/D3/D5 firing rates measured |
| **2** | `detectors.py` (H1–H4, stuffer, rescue, twins, top-150 audit) | predict which traps a candidate trips |
| **3** | `rank.py` → valid top-100 CSV in budget | trace one candidate raw → rank; repro in budget |
| **4** | `reasoning.py` + 6-check self-test | 5 reasonings each cite a real fact; pairwise near-duplicate check passes |
| **5** | repro test, sandbox (HF + docker), metadata, README | run the repro command, finish in budget |
| **6** | interview readiness | pass the "why not X" gauntlet cold |

*Plan derived from `context.md`, `architecture.md`, and `redrob_build_playbook.md`; field paths verified against `candidate_schema.json`; validator behavior verified against `validate_submission.py`; `[DATA]` claims independently re-derived from `candidates.jsonl` on 2026-06-15 (§0.5). Reference "now": 2026-05-27.*
