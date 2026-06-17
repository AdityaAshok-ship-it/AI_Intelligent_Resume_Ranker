# edgecases.md — Redrob Ranking Challenge

> **What this file is.** The consolidated catalog of edge cases the ranking pipeline must survive — the boundary conditions, sentinel values, false-positive traps, and validator corner cases that a "happy-path" build silently gets wrong. It is the operational complement to the design docs: where `architecture.md` says *what* the system does and `implementation-plan.md` says *in what order*, this file enumerates *where it breaks* and *what the correct behavior is at the break*.
>
> **Derived from** `implementation-plan.md`, `architecture.md`, `eval.md`, and `job_description.docx` (the JD's stated disqualifiers and "between the lines" profile), cross-checked against `candidate_schema.json` and `validate_submission.py` (the two files that define the hard input/output contracts).
>
> **Conventions.** Reference "now" = **2026-05-27** (max `last_active_date`). Provenance tags mirror the other docs: `[SPEC]` organizer bundle · `[DATA]` measured on the 100K / 50-sample · `[CALL]` our decision. Each case has an ID (`EC-n`) so it can be cited from `decision.md` and the exit gates.
>
> **Handling status legend:**
> - **HANDLED** — a specific code path already neutralizes it; the entry says where.
> - **FLAG** — detect and surface (reasoning honest-concern or audit log), do not silently drop.
> - **VERIFY** — no automatic guarantee; an exit-gate check must confirm it before submission.
> - **RESIDUAL** — known-uncatchable; bounded by argument, stated honestly at Stage 5.

---

## 0. Input / schema / parser edge cases (Phase 0)

The parser is the upstream of everything; a coercion bug here poisons the 65% rubric, the gates, and the honeypots at once. The data is **dense on every parser-critical field** `[DATA]` — but three fields are sparse and carry **sentinels that are not real values**.

| ID | Edge case | Where it bites | Correct behavior | Status |
|---|---|---|---|---|
| **EC-1** | `github_activity_score == −1` (65% of pool `[DATA]`) — sentinel for "no GitHub linked", **not** a score of −1. | Any arithmetic/threshold using the score (e.g. an engagement signal) reads −1 as "very inactive". | Treat −1 as **absent**: exclude from any mean/threshold; never let it push a candidate toward the ghost floor. Absence ≠ disengagement. | **HANDLED** (explicit sentinel branch) |
| **EC-2** | `offer_acceptance_rate == −1` sentinel ("no offer history"). | Same coercion risk; a real 0.0 acceptance vs −1 sentinel are opposite meanings. | Branch on `== −1` → absent. Do not average the sentinel into anything. | **HANDLED** |
| **EC-3** | `skill_assessment_scores` present for only **24%** `[DATA]` (empty `{}` for the rest). | A rule that reads "no assessment" as "failed assessment". | Absent dict = no signal, **not** a low signal. Any rule touching it handles the empty case explicitly. | **HANDLED** |
| **EC-4** | `career_history[].end_date` is `null` for the current role (`is_current == true`) `[SPEC schema]`. | H1 timeline math (`duration_months` vs months-since-`start_date`) and career-span computation dereference `end_date`. | For the current role, treat `end_date` as **now (2026-05-27)**; never `None`-arithmetic. The span's right edge is "now". | **VERIFY** (assert no null reaches date math) |
| **EC-5** | `career_history` length **1** for **18.5%** of pool `[DATA]`; single role → career span = current tenure, so an inflated `years_of_experience` trips H3. | H3 false positives on legitimate single-role people. | H3's **+18 mo slack** is exactly the buffer that keeps legit single-role candidates out. Do not tighten it without re-measuring this 18.5% population. | **HANDLED** ([[EC-26]]) |
| **EC-6** | `description` / `.industry` **do not reliably track the role title** in the sample `[DATA]` — a "Marketing Manager" with a mechanical-engineering description; identical text repeated across `CAND_0000031`'s two roles. | `built_real_system` (regex over `description`) and `product_vs_services` (over `industry`) ingest noise/contradiction → corrupts the 65% inputs silently. | The flags run on noisier text than assumed. **Eyeball all 50 sample flags against raw text before calibrating** (Phase 0 exit gate). If a flag disagrees with a human read, the extraction is wrong — fix here, cheaply. | **VERIFY** (Phase 0 gate) |
| **EC-7** | `skills[]` is the keyword-stuffer attack surface. | If embedded, buzzword skill lists inflate cosine for non-fits. | **Exclude `skills[]` from embedding text** and from `built_real_system`; embed only `current_title + headline + summary + career_history[].description`. | **HANDLED** ([[EC-12]]) |
| **EC-8** | `education` may be **empty** (`minItems: 0`); `certifications` / `languages` are **optional** (absent key, not null). | A scorer that assumes the keys exist KeyErrors on a sparse profile. | Education = 5% weight; treat empty as zero-contribution, not a penalty. Guard optional keys with `.get()`. | **HANDLED** |
| **EC-9** | `notice_period_days` is **quantized** to {0,15,30,45,60,90,120,150} (schema allows 0–180), mode **90** `[DATA]`. Strict `<30` = **22 records (0.0%)** — an empty branch. | A "short-notice bonus" keyed on `notice < 30` fires on **~nobody** — a dead branch that looks active in code. | Key the buyout-tier bonus on **`<= 30`** (= 13.8%, the populated tier the JD's "buy out up to 30 days" actually means). Treat 120/150 as honest-concern band, never disqualifying. | **HANDLED** (corrected from "13.8% sub-30") |
| **EC-10** | File is **487 MB** as delivered `[DATA]` (README says ~465 MB `[SPEC]`); 100K parsed dicts in a list risks the 16 GB ceiling. | Precompute OOM. | **Stream line-by-line**; never materialize a 100K list. | **HANDLED** |
| **EC-11** | Row-alignment drift: `candidate_matrix[i]` must map to `candidate_ids[i]` must map to the `features` row for that id. | A single off-by-one silently corrupts **every** rank — undetectable in the output, fatal in the interview. | Write matrix + ids in lockstep; **assert the invariant at load time in `rank.py`** before any scoring. | **VERIFY** (load-time assert) |

---

## 1. Embedding / semantic edge cases (20% component)

The embedding is a *secondary* signal whose job is the **plain-language tier-5 rescue**. Its failure modes are symmetric: it can boost buzzword prose, and it can be fooled by keyword homonyms.

| ID | Edge case | Correct behavior | Status |
|---|---|---|---|
| **EC-12** | **Keyword homonym false-match** `[DATA, sample-verified]`: a Marketing Manager's *"ranked on the first page of search for high-competition keywords"* (SEO) trips a naive `rank`+`search` pass. | The 65% extractor must disambiguate "ranked **on** search results" (SEO outcome) from "shipped **a** ranking/search **system**". Title gate + career-evidence regex (shipped ranking/retrieval/recommendation *systems*) — not bare keyword presence. | **VERIFY** ([[EC-18]]) |
| **EC-13** | **Plain-language tier-5**: a candidate who *built* a recommendation system at a product company but never writes "RAG" or "Pinecone" `[SPEC: JD final note]`. | The 20% embedding + career-evidence pass must **rescue** them above keyword-rich non-fits. This is the embedding's whole reason to exist. | **VERIFY** (calibration) |
| **EC-14** | **Buzzword-stuffed prose** scores high on cosine even with no real career evidence (e.g. a summary full of "Building with LLMs", "RAG"). | The career-first 65% weight + keyword-stuffer flag must **dominate** the topical boost the embedding gives buzzword prose. Confirm `CAND_0000021` (PM, AI buzzwords in skills *and* summary) is demoted far below `CAND_0000031`. | **VERIFY** (Phase 2 gate) |
| **EC-15** | **Repeated/duplicated `description` text across roles** (`CAND_0000031`'s two roles carry identical text) `[DATA]`. | Embedding double-counts the same prose; acceptable for cosine but means description-based evidence regex must not "count" the same achievement twice. | **FLAG** (de-dupe evidence hits per candidate) |

---

## 2. Rubric & career-evidence edge cases (65% component — the differentiator)

This is the highest-weighted, least-specified component. Most score-separation and most false-positive risk live here.

| ID | Edge case | Correct behavior | Status |
|---|---|---|---|
| **EC-16** | **The archetype `CAND_0000031`** — 88-month Pinecone proficiency on a 72-month career `[DATA]`. The "skill duration > career length" pattern looks like a honeypot but is **7 years of normal pre-career familiarity**. | Must **survive** and rank #1-class for the *right reason* (career evidence, not a cosine/keyword accident). The naive skill-duration rule deletes it — that is the single sharpest argument for rejecting that rule. | **HANDLED** ([[EC-28]]) |
| **EC-17** | **Keyword-stuffer**: non-eng title + ≥3 AI skills + **no career evidence** (sample: `CAND_0000021`, PM with AI buzzwords) `[DATA]`. | **Flag, do not double-penalize** — career-first weighting already demotes them; an aggressive penalty false-positives genuine career-changers. Lock the AI-skill list in code; quote *that* firing rate (~3.6–4.7%, definition-sensitive), never the doc figure. | **HANDLED** |
| **EC-18** | **Genuine career-changer** (real backend→ML transitioner; the modal real changer has **2** AI skills) `[SPEC: JD "we have real transitioners"]`. | Threshold held at **≥3** AI skills by decision — lowering to ≥2 adds only 166, disproportionately genuine 2-skill transitioners, for zero ranking gain. One services employer or keyword overlap must **not** cap a real transitioner. | **HANDLED** |
| **EC-19** | **Honeypot wearing a perfect title** `[DATA]`: `CAND_0010770` is titled *"Recommendation Systems Engineer"* — identical to the archetype — and is caught **only** by H3 timeline math, not by title or keywords. A keyword-stuffed honeypot with a perfect title **will** score high on the rubric. | Exactly why honeypot exclusion is a **pre-sort hard gate**, not a score penalty — the rubric alone cannot see these. | **HANDLED** ([[EC-25]]) |
| **EC-20** | **Title-chaser** `[SPEC: JD "do NOT want"]`: Senior→Staff→Principal by switching companies every ~1.5 years; JD wants 3+ year tenure. | Surface as a soft signal / honest concern; the JD lists it under "do not want" but not as a hard "we will not move forward" — do not over-weight into a disqualifier. | **FLAG** |
| **EC-21** | **`product_vs_services` classification** — all-consulting career (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/Mindtree/HCL/Tech Mahindra/Mphasis). | Services classification feeds D4, **but** the JD's explicit exception: *"currently at one of these but prior product-company experience → that's fine"*. Classify per-role; do not cap on current-company alone. | **HANDLED** ([[EC-24]]) |

---

## 3. Disqualifier-gate edge cases (D1–D5) — the under-guarded score-killers

`eval.md` punch-list item #2: these gates are **deep semantic judgments approximated by title/keyword heuristics with no measured false-positive rate**, and a false positive **zeroes out exactly the top-10 candidate you compete on** — the same archetype-deletion failure the skill-duration rule was rejected for. Severity must mirror the JD's *actual* wording: only **D1** carries "we will not move forward".

| ID | Gate | False-positive edge case (the danger) | Correct behavior | Status |
|---|---|---|---|---|
| **EC-22** | **D1** — pure research, zero production (**hard floor**, cap≈0). | Research-heavy career that *does* have a production deployment → must **not** floor. | Fire only on **zero production in any role**. Research-with-production survives. D1 is the one literal "we will not move forward". | **VERIFY** (firing inspection) |
| **EC-23** | **D2** — "AI experience" only in a sub-12-mo current LangChain/OpenAI role. | A real pre-LLM ML engineer whose *most recent* role happens to be a LangChain project → wrongly penalized. | **Exception: pre-LLM ML anywhere in history → do not fire.** D2 is the fuzziest gate; don't overclaim its precision. | **VERIFY** (firing rate on 100K + inspect) |
| **EC-24** | **D3** — senior, no IC code in 18 mo (inferred from title). | **Staff/Principal Engineer who still codes**, or a "coding manager", read as "architect/lead/manager-only" from the title → heavy penalty on a genuine senior IC. Titles do not reveal who writes code. | Measure firing rate; inspect who it caps; **soften if any caught profile reads as a plausible IC**. This is the archetype-deletion risk, unmeasured. | **VERIFY** (punch-list #2) |
| **EC-25** | **D4** — entire-career consulting (cap). | Currently at TCS/Infosys **but with a prior product role** → JD explicitly says fine. | The product-company exception is **part of the firing condition**: current-consulting-with-prior-product does **not** fire. Fires only on *all-consulting* careers. | **HANDLED** |
| **EC-26** | **D5** — CV/speech/robotics primary, no NLP/IR (strong penalty, softer JD "do NOT want"). | A real **NLP/IR engineer who once shipped a vision feature** → wrongly capped. | Fire only on *primary* CV/speech/robotics with **no** NLP/IR evidence. Inspect captures; soften if it catches an NLP person. | **VERIFY** |
| **EC-27** | **Multiple gates fire** on one candidate. | Ambiguous final cap. | **Most-restrictive firing cap wins.** Caps apply to `base` **before** the multiplier, so an available-but-disqualified candidate can never be lifted back up (multiplier ≤ 1). Reversing this order is a Stage-5 credibility failure. | **HANDLED** ([[EC-40]]) |

---

## 4. Honeypot edge cases (H1–H4) — binary DQ insurance

The H1–H4 union catches **exactly 68** `[DATA, 2026-06-15]` — the number the >10%-in-top-100 DQ insurance rests on. Per-rule (overlapping): H1≈19, H2≈22, H3≈25, H4≈21. Flagged rows are **dropped from the set before sort**, never score-adjusted.

| ID | Edge case | Correct behavior | Status |
|---|---|---|---|
| **EC-28** | **The naive rule to AVOID**: "skill `duration_months` > career length = honeypot". | **Do not use.** Fires on **13–19%** of the pool (def-sensitive), **deletes `CAND_0000031`**, and adds **zero** honeypots H1–H4 don't already catch. The classic over-aggressive-detector trap. | **HANDLED** (rejected, logged) |
| **EC-29** | **H1 boundary** — `duration_months` exactly equals months-since-`start_date`. | **+3 mo slack** absorbs date rounding; fire only when `duration_months > months_since_start + 3`. Boundary candidates (within slack) survive. | **HANDLED** |
| **EC-30** | **H2 — legitimate concurrent roles** inflate summed tenure above `years_of_experience×12`. | **+30 mo slack** allows genuine overlap; fire only when `Σtenure > YOE×12 + 30`. | **HANDLED** |
| **EC-31** | **H3 — single-role candidate** with early internships inflating YOE above career span. | **+18 mo slack** allows early internships; protects the 18.5% single-role population ([[EC-5]]). Fire only when `YOE×12 > span + 18`. | **HANDLED** |
| **EC-32** | **H4** — skill at {advanced, expert} with `duration_months == 0` ("expert, 0 years"). | Hard impossibility → flag. Distinguish from a skill with a *positive* small duration (legit) and from `duration_months` absent (handle as not-firing, not zero). | **HANDLED** |
| **EC-33** | **Residual ~12 honeypots** — "8 years at a company founded 3 years ago" at **fictional** employers (Hooli, Stark, Dunder Mifflin). Schema has **no company founding dates**. | **RESIDUAL, uncatchable.** The top-150 audit checks founding dates only for **known real** companies, so it does **not** cover this. Bounded by arithmetic: DQ needs **>10 honeypots in top 100**, so fatal only if ~11 of ~12 both surface *and* out-score real fits — improbable (they carry no career-evidence advantage). **Do not claim the audit catches them.** | **RESIDUAL** (stated honestly) |
| **EC-34** | **Honeypot reaches the shortlist** (a near-miss perfect-title honeypot like `CAND_0010770` in the top 100). | **Top-150 audit guard**: re-run H1–H4 over the top ~150, scrutinize "too good to be true", founding-date check for *known real* companies. **Deterministic + logged; the human eyeball is a check, never a manual CSV edit** (manual edits between code and output are a Stage-3/4 red flag). | **HANDLED** |

---

## 5. Availability-multiplier edge cases (the highest-leverage number)

A **multiplier on `gated`** (∈ ~[0.15, 1.0]), continuous not bucketed. Its calibration target — a tier-5 twin pair and a high-base ghost — **does not exist in the 50-sample** `[DATA]`, so it is **reasoned against synthetic twins**, not tuned.

| ID | Edge case | Correct behavior | Status |
|---|---|---|---|
| **EC-35** | **Ghost = a conjunction, not any single signal.** Only `staleness_days > 180` **AND** `recruiter_response_rate` near-zero **AND** `open_to_work_flag == false` → floor ~0.15. | A candidate meeting **only one or two** of the three must **not** be floored. The floor fires on the 3.4% true ghosts only. | **HANDLED** |
| **EC-36** | **`open_to_work_flag == false` but active and responsive** (35.3% are open; 65% are not — being "not open" is the norm, not a ghost). | Not a ghost. The flag alone is weak; only the full conjunction floors. Healthy band 0.7–1.0. | **HANDLED** |
| **EC-37** | **High-base ghost** — a strong tier-5 who is disengaged. | The gentle slope must demote them **below the engaged twin** but **not below an unrelated weaker-but-engaged** candidate. This is the case the multiplier exists for — and the sample **cannot** test it. | **VERIFY** (synthetic twins) |
| **EC-38** | **Behavioral twins** — two candidates identical but for the availability signal. | Continuous (not bucketed) multiplier so they **separate on the signal delta alone**. Clone the archetype into a twin pair, confirm the ghost lands below the engaged twin without over-crushing. **Log the band/floor + this test in `decision.md`** — the only evidence the number was reasoned. | **VERIFY** |
| **EC-39** | **Multiplicative-stack pitfall** `[DATA]`: `open × response × staleness` drives **median to 0.37×**, **77% below 0.5×** — flattens the 96.6% healthy majority to fight the 3.4% ghosts, and since NDCG@10 is half the composite it **demotes good fits**. | **Reject the stack.** Use one gentle multiplier. Present the 0.37×/77% figures as *illustrative of collapse direction* (form-dependent), not as exact measurements, at Stage 5. | **HANDLED** |
| **EC-40** | **`recruiter_response_rate` near-zero with no recruiter messages** vs genuine non-response, and **`github −1`** ([[EC-1]]) feeding a "disengaged" read. | Don't conflate "no inbound to respond to" with "ignores recruiters"; the conjunction (stale + not-open) guards against a lone spurious 0. Never let `github == −1` push toward the ghost floor. | **HANDLED** |

---

## 6. Logistics edge cases (small modifier — must never dominate)

The JD is explicit that logistics are *honest concerns*, **not disqualifiers** ("a great fit with 120-day notice is still a great fit").

| ID | Edge case | Correct behavior | Status |
|---|---|---|---|
| **EC-41** | **Long notice (120/150 days)** `[DATA]` — the honest-concern band. | Small penalty **+ honest-concern flag in reasoning**; **never disqualifying**. JD buys out up to 30 days; 30+ raises the bar, doesn't close the door. | **HANDLED** |
| **EC-42** | **International candidate, not relocating** `[SPEC: "we don't sponsor work visas"]`. | Discount (no visa sponsorship) **+ honest concern**. But `willing_to_relocate == true` (28.8% `[DATA]`) → **full credit**; India-based (75.1%) → full credit. | **HANDLED** |
| **EC-43** | **Tier-1 city** (Noida/Pune/Hyderabad/Mumbai/Delhi-NCR/Bangalore) `[SPEC]`. | Mild bonus only; offices are Pune/Noida. Must not dominate career evidence. | **HANDLED** |
| **EC-44** | **Work-mode mismatch** (remote-only vs hybrid; ~uniform across modes `[DATA]`). | Minor honest-concern flag; Stage-4 explicitly checks reasoning surfaces "wrong work mode". Never a score driver. | **FLAG** |

---

## 7. Composition-order & doc-consistency edge cases

| ID | Edge case | Correct behavior | Status |
|---|---|---|---|
| **EC-45** | **Caps vs multiplier order.** If the multiplier applied *before* caps, an available-but-disqualified candidate could float into the top 100. | **Caps before multiplier** — the one ordering choice that actually changes scores. ([[EC-27]]) | **HANDLED** |
| **EC-46** | **Honeypot-exclusion position.** `architecture.md` §5.0/§6 places the multiplier (step 5) **before** honeypot exclusion (step 6); `implementation-plan.md` Phase 3 places exclusion (step 5) **before** the multiplier (step 6). | Outcome is **identical** (a dropped row never resurfaces) — but **two canonical docs stating the pipeline in two different orders is a Stage-5 "contradicts the docs" trap**. **Fix: make `architecture.md` and the code agree on one order** (exclusion-before-multiplier, per Phase-1 composition). Don't re-prove equivalence in the interview — remove the contradiction. | **VERIFY** (eval.md punch-list #4) |

---

## 8. Output / validator edge cases (`validate_submission.py`, verified in code)

The validator enforces more than the headline rules. These are hard rejections — every CSV must clear them before it leaves the machine.

| ID | Edge case | Validator behavior (from code) | Correct behavior | Status |
|---|---|---|---|---|
| **EC-47** | **Tie-break trap.** Equal `score` on an adjacent pair where the better-ranked candidate has a **larger** `candidate_id` → **rejected** (lines 136–144). | `if s1 == s2 and c1 > c2: error`. | Emit **distinct floats** so no equal-score pair exists and the check never fires. Fallback: sort any surviving tie `candidate_id`-ascending. | **HANDLED** |
| **EC-48** | **Precision-collapse ties** — scores distinct in memory but **formatted to too few decimals** in the CSV print equal → the tie-break trap ([[EC-47]]) fires on rows that weren't tied. | Validator parses the **string** `float(score_s)`; equality is on the written value. | Write **enough precision** that distinctness survives serialization (don't round `final` to 4 dp before writing). This is the subtle one — distinctness in memory is necessary but **not sufficient**. | **VERIFY** (check written CSV, not in-memory array) |
| **EC-49** | **Score non-increasing by rank** — `s1 < s2` for adjacent ranks → rejected (lines 127–134). | Enforced structurally. | **Sort `final` descending** (step 8) satisfies it automatically. | **HANDLED** |
| **EC-50** | **`rank` must be a clean integer string.** `str(int(rank_s)) != rank_s` → rejected (line 98). So `"1.0"`, `"01"`, `" 1"` (pre-strip aside), `"1 "` fail. | `int("1.0")` raises; `int("01")==1` but `"1"!="01"`. | Write ranks as **plain integers 1–100**, each exactly once, no leading zeros, no decimals. | **HANDLED** |
| **EC-51** | **Exactly 100 data rows** — not 99, not 101 (lines 58–64). **Whitespace-only rows are skipped** by the reader (line 48), so a stray blank line won't inflate the count — but a genuinely missing row will fail. | `any(cell.strip() for cell in row)` filters blank rows. | Emit precisely 100 non-empty rows; ranks 1–100 each used once (the `missing` check, line 119). | **HANDLED** |
| **EC-52** | **`candidate_id` format & uniqueness** — must match `^CAND_[0-9]{7}$`, no duplicates (lines 87–94). | Regex + `seen_ids`. | Rank only real corpus IDs; never hand-edit IDs. Existence is enforced **server-side** (validator only checks the pattern). | **HANDLED** |
| **EC-53** | **`reasoning` with commas / quotes / newlines** breaks naive CSV writing → column-count mismatch (`len(cells) != 4`, line 73). | `csv.reader` respects RFC-4180 quoting. | Use a proper CSV **writer** (`csv.writer`, default quoting) so reasoning text is quoted/escaped. Never hand-concatenate with commas. | **VERIFY** |
| **EC-54** | **Encoding** — file must be **UTF-8** (line 51) and **`.csv`** extension with a non-empty stem = participant ID (lines 22–25). | `UnicodeDecodeError` → rejected. | Write UTF-8; name the file `<participant_id>.csv`. On Windows, beware tools that default to UTF-16/BOM — write UTF-8 explicitly. | **VERIFY** (Windows encoding) |
| **EC-55** | **Header must be exactly** `candidate_id,rank,score,reasoning` in that order (line 38). | Exact list compare. | Emit that header verbatim; no extra/renamed/reordered columns. | **HANDLED** |

---

## 9. Reasoning-generation edge cases (Phase 4 — the 6 Stage-4 checks)

Fact-grounded templates, no LLM (determinism + hallucination-proof). The failure modes are collisions and hallucination, neither of which a 5-string spot-read catches.

| ID | Edge case | Correct behavior | Status |
|---|---|---|---|
| **EC-56** | **Template collision** — 3–4 structural variants over 100 rows ⇒ ~25 candidates/skeleton; two genuinely similar candidates (two available product-company Rec-Sys Engineers) pull near-identical facts into the same skeleton → read as "name-insert templating" (an explicit Stage-4 penalty). | **Pairwise near-duplicate check across all 100** (n-gram overlap / edit distance between **every** pair); force structural divergence on any flagged pair. The collision is a pairwise property — verify it pairwise, not by spot-read. | **VERIFY** (Phase 4 gate) |
| **EC-57** | **Hallucination** — a string cites a fact not in that candidate's profile. The **fastest Stage-4 fail**. | Every string pulls **only real facts** (years, title, named skills, signal values, the specific gap) from that candidate. Templates make this structural. | **VERIFY** (5-fact spot-read + provenance) |
| **EC-58** | **Sparse-field citation** — reasoning says "0 GitHub activity" when the value is the −1 sentinel ([[EC-1]]), or cites a `skill_assessment_score` that's absent. | Never render a sentinel as a real number. Omit the signal when absent rather than fabricate a value. | **HANDLED** |
| **EC-59** | **Missing honest concern** where a gap exists (long notice, wrong mode, services background, international/visa, stale activity). Stage-4 explicitly checks for these. | Every string = a real fact + a JD connection + an **honest gap where one exists**. Tone scales to rank: top-10 leads with strongest evidence (may note one caveat); bottom names the disqualifying gap plainly. | **HANDLED** |
| **EC-60** | **Rank-inconsistency** — a low-ranked candidate's reasoning reads more positive than a high-ranked one's. | Rank-consistency is one of the six checks; tone must monotonically track rank. | **VERIFY** |

---

## 10. Calibration / methodology / defensibility edge cases (the score-defining ones)

These are not code bugs — they are the places where the *plan's central claim* meets data that can't support it, and where Stage 5 is won or lost. `eval.md` punch-list items #1, #3.

| ID | Edge case | Correct behavior | Status |
|---|---|---|---|
| **EC-61** | **Top-10 ordering can't be calibrated** — the 50-sample has **exactly one** tier-5 (`CAND_0000031`), and top-10 ordering is **80%** of the composite (`0.50·NDCG@10 + 0.30·NDCG@50`). Calibration can separate 1 fit from 49 non-fits (trivial) but **cannot** order tier-5 #3 above tier-5 #7 — there is no #3. | **Stop selling calibration as de-risking.** The top-10 order is **reasoned, not tuned**. Its only Stage-5 backstop is the **written, JD-traced tier-5 ordering rules** (product-scale of the shipped system, eval-framework depth, recency of hands-on ranking work, availability). **Build those rules; exercise them on 3–4 synthetic tier-5 variants** (perturb the archetype). | **VERIFY** (Phase 1 deliverable) |
| **EC-62** | **Held-out check set is misread as validating the order.** It guards against overfitting your weights to your *own* hand-ranks — it does **nothing** about your hand-ranks diverging from the hidden labels (no leaderboard). | Use it for overfitting only. State plainly that nothing here validates against ground truth. Stop tuning when held-out ordering stops improving; don't chase the last point. | **HANDLED** |
| **EC-63** | **Definition-sensitive number quoted as fact** — quoting "3.6% keyword-stuffers" while your shipped detector flags 4.7% is a **self-contradiction** at Stage 5 (fails on "contradicts the code/data"). | **Re-derive from your shipped detector and quote that.** Split confirmed-exact (quote freely: 68 honeypots, 3.4% ghosts, archetype facts, ref date, all percentiles) from definition-sensitive (eng-title %, stuffer %, tier-5 pool, naive-rule FP count — re-derive). | **HANDLED** |
| **EC-64** | **Form-dependent figure presented as a measurement** — the multiplicative-stack "median→0.37×, 77%<0.5×" depends on the exact penalty curve compared against; not independently re-derivable from the file. | Present as **illustrative of the collapse direction** (which the 3.4% ghost rate + 0.44 median response rate fully support), or re-derive from the specific stack in `decision.md`. An unverifiable precise number is a self-inflicted wound. | **HANDLED** |
| **EC-65** | **Submission spent "to see how it does."** Max 3 submissions, **no live leaderboard**, last valid is final `[SPEC]`. A submission returns **zero scoring feedback** — only format validity + server-side ID existence. | All score-improving work is **offline**. Submission 1 = confirm format passes server-side. 2–3 = reserve for a genuine post-submission defect. **Never probe — you can't see.** | **HANDLED** |

---

## 11. Budget / runtime / sandbox edge cases (Phase 3 / 5)

| ID | Edge case | Correct behavior | Status |
|---|---|---|---|
| **EC-66** | **Model accidentally loaded in the ranking step** → breaks "network off" / risks the 5-min cap. | The 100K rank step **loads no model** — it consumes the precomputed matrix + JD vector. Network-off is satisfied **structurally**. Only precompute and the sandbox load weights. | **HANDLED** |
| **EC-67** | **Two execution paths fail differently** — the full repro path loads **no** model; the sandbox embeds **live** (loads bundled weights). | **Exercise both before submission.** A bug in the live-embed path won't show in the no-model path and vice versa. | **VERIFY** (Phase 5) |
| **EC-68** | **Offline model resolution** — a Hub call is a no-network failure mode. | `HF_HUB_OFFLINE=1` + bundled local weights path; confirm the offline flag actually prevents a network call before relying on it. | **VERIFY** (pre-flight P2) |
| **EC-69** | **Sandbox single point of failure at Stage 1** (missing sandbox is flagged at Stage 1). | **HF Spaces (Streamlit) primary + a `docker run` recipe in the README** as backup — the spec accepts an unmodified `docker run` substitute. Doing both removes the SPOF. | **HANDLED** |

---

## Boundary-value quick reference

The threshold cases most likely to hide an off-by-one or sentinel bug — verify each at its exact edge:

| Quantity | Threshold | Boundary rule |
|---|---|---|
| H1 | `duration_months − months_since_start` | fire only when `> +3 mo` |
| H2 | `Σtenure − YOE×12` | fire only when `> +30 mo` |
| H3 | `YOE×12 − career_span` | fire only when `> +18 mo` |
| H4 | skill duration | fire when `== 0` at {advanced, expert} only |
| Ghost floor | conjunction | `staleness > 180` **AND** `rrr ≈ 0` **AND** `open == false` (all three) |
| Notice bonus | `notice_period_days` | key on **`<= 30`** (not `< 30` — empty) |
| Honeypot DQ | honeypots in top 100 | DQ at **`> 10`** |
| Sentinels | `github_activity_score == −1`, `offer_acceptance_rate == −1` | branch to **absent**, never arithmetic |
| `end_date` | `is_current == true` | substitute **now (2026-05-27)**, never `None` |
| Output rows | data rows | exactly **100** (blank rows skipped by validator) |
| Output rank | integer string | `str(int(r)) == r` — no `"1.0"`, no `"01"` |
| Output score | adjacent equal | distinct floats **after serialization** ([[EC-48]]) |

---

## Priority — which edge cases actually cost the score

Ordered by `eval.md`'s severity ranking, mapped to the cases above:

1. **[Score-defining]** Top-10 ordering can't be calibrated → **EC-61, EC-62**. Write the tier-5 ordering rules; reasoned-not-tuned defense.
2. **[Score-damaging]** D2/D3/D5 gates can zero your top-10 with unmeasured FP rate → **EC-23, EC-24, EC-26**. Measure firing rates on 100K, inspect, soften.
3. **[Defensibility]** Residual-honeypot mitigation is circular → **EC-33**. Honest arithmetic, don't claim the audit covers it.
4. **[Defensibility]** Two docs state pipeline order differently → **EC-46**. Reconcile `architecture.md` + code to one order.
5. **[Stage-4]** Template collisions vs a "read 5" gate → **EC-56**. Pairwise near-duplicate check.
6. **[Input quality]** Parser reads noisy description/industry as clean → **EC-6**. Eyeball all 50 before calibrating.
7. **[Silent output reject]** Precision-collapse ties + CSV escaping + Windows UTF-8 → **EC-48, EC-53, EC-54**. Check the *written* file, not the in-memory array.

Items 3, 4, 6 and the EC-48/53/54 output checks are cheap and should be done before any calibration. Items 1 and 2 are where the remaining judgment-hours belong — not in the 20% mechanics the plan already, correctly, calls easy.

---

*Edge cases derived from `implementation-plan.md`, `architecture.md`, `eval.md`, and `job_description.docx`; input/output boundaries verified against `candidate_schema.json` and `validate_submission.py`. Reference "now": 2026-05-27. Each `EC-n` is citable from `decision.md` and the phase exit gates.*
