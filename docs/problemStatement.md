# Problem Statement
## Redrob — Intelligent Candidate Discovery & Ranking Challenge
### IndiaRun Hackathon · Track 1

---

## 1. Context

Redrob AI is a Series A, AI-native talent intelligence platform. Their current candidate-to-JD matching pipeline is BM25 + rule-based scoring — functional but not intelligent. The challenge asks participants to build the next-generation ranking layer: a system that understands *what a JD means*, not just what keywords it contains, and that incorporates behavioural signals to distinguish genuinely available candidates from paper-perfect ghosts.

---

## 2. Objective

Given a pool of **100,000 candidate profiles** and **one job description**, build a CPU-only, offline ranking system that:

- Identifies the **top 100 best-fit candidates** for the role.
- Assigns each a rank (1 = best), a continuous score, and a 1–2 sentence reasoning.
- Runs end-to-end within the compute constraints below.
- Produces a submission CSV that passes the format validator (`validate_submission.py`).

---

## 3. The Role Being Hired For

**Senior AI Engineer — Founding Team** at Redrob AI (Pune / Noida, India · Hybrid).

### Hard requirements (JD language: "absolutely need")
- Production experience with **embeddings-based retrieval** (sentence-transformers, BGE, E5, OpenAI embeddings, or equivalent) — specifically: handling embedding drift, index refresh, retrieval-quality regression in production.
- Production experience with **vector databases or hybrid search** (Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, or equivalent).
- Strong **Python** and code quality standards.
- Hands-on **evaluation framework** experience: NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.

### Preferred but not disqualifying
- LLM fine-tuning (LoRA, QLoRA, PEFT).
- Learning-to-rank models (XGBoost-based or neural).
- HR-tech / marketplace / distributed systems background.

### Explicit disqualifiers (JD-defined)
1. Pure-research background with no production deployments.
2. "AI experience" consisting only of recent (<12 months) LangChain-to-OpenAI projects, with no pre-LLM-era ML production work.
3. Senior/Staff/Principal engineers who have not written production code in the last 18 months.
4. Full-career consulting-firm background (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.).
5. Primary expertise in CV / speech / robotics with no NLP/IR exposure.

### Key hiring signal (JD language)
> The "right answer" to this JD is not keyword-matching. A candidate whose career history shows they built a recommendation system at a product company is a fit even if they don't write "RAG" or "Pinecone" in their profile. A Marketing Manager whose skills list contains every AI keyword is not a fit, no matter how complete the list.

---

## 4. Input Data

### 4.1 Candidate Pool
- **File:** `candidates.jsonl` (~465 MB, 100,000 records, one JSON per line)
- **Quick-inspect copy:** `sample_candidates.json` (first 50 candidates, pretty-printed)
- **Schema:** `candidate_schema.json`

Each candidate record contains:

| Section | Key fields |
|---|---|
| `profile` | `candidate_id` (CAND_XXXXXXX), headline, summary, location, country, `years_of_experience`, `current_title`, `current_company`, `current_company_size`, `current_industry` |
| `career_history` | Array of up to 10 roles — `company`, `title`, `start_date`, `end_date`, `duration_months`, `is_current`, `industry`, `company_size`, `description` |
| `education` | `institution`, `degree`, `field_of_study`, `start_year`, `end_year`, `grade`, `tier` (tier_1 / tier_2 / tier_3 / tier_4 / unknown) |
| `skills` | `name`, `proficiency` (beginner/intermediate/advanced/expert), `endorsements`, `duration_months` |
| `certifications` | `name`, `issuer`, `year` |
| `redrob_signals` | 23 behavioural signals (see §4.2) |

### 4.2 Behavioural Signals (`redrob_signals`)
23 platform-derived signals, intended as **multipliers / modifiers** on top of skill-match scores:

| # | Signal | Range | What it measures |
|---|---|---|---|
| 1 | `profile_completeness_score` | 0–100 | Profile fill-in percentage |
| 2 | `signup_date` | date | When they joined Redrob |
| 3 | `last_active_date` | date | Last login date |
| 4 | `open_to_work_flag` | bool | Self-declared availability |
| 5 | `profile_views_received_30d` | int ≥ 0 | Recruiter views in last 30 days |
| 6 | `applications_submitted_30d` | int ≥ 0 | Applications filed recently |
| 7 | `recruiter_response_rate` | 0.0–1.0 | Fraction of recruiter messages replied to |
| 8 | `avg_response_time_hours` | number ≥ 0 | Median time to reply to a recruiter |
| 9 | `skill_assessment_scores` | dict[str, 0–100] | Per-skill Redrob platform assessment scores |
| 10 | `connection_count` | int ≥ 0 | Redrob network connections |
| 11 | `endorsements_received` | int ≥ 0 | Total endorsements received |
| 12 | `notice_period_days` | 0–180 | Stated notice period |
| 13 | `expected_salary_range_inr_lpa` | {min, max} | Salary expectations (INR LPA) |
| 14 | `preferred_work_mode` | onsite/hybrid/remote/flexible | Stated work-mode preference |
| 15 | `willing_to_relocate` | bool | Relocation willingness |
| 16 | `github_activity_score` | −1 to 100 | GitHub activity (−1 = no GitHub linked) |
| 17 | `search_appearance_30d` | int ≥ 0 | Recruiter search appearances in last 30 days |
| 18 | `saved_by_recruiters_30d` | int ≥ 0 | Recruiter bookmarks in last 30 days |
| 19 | `interview_completion_rate` | 0.0–1.0 | Fraction of scheduled interviews actually attended |
| 20 | `offer_acceptance_rate` | −1 to 1.0 | Historical offer acceptance rate (−1 = no history) |
| 21 | `verified_email` | bool | Email verified on platform |
| 22 | `verified_phone` | bool | Phone verified on platform |
| 23 | `linkedin_connected` | bool | LinkedIn account linked |

---

## 5. Output Format

### 5.1 Submission CSV
**Filename:** `<your_registered_participant_id>.csv`  
**Encoding:** UTF-8

**Required columns (exact order):**

```
candidate_id,rank,score,reasoning
```

| Column | Type | Rules |
|---|---|---|
| `candidate_id` | string | Must match `CAND_[0-9]{7}` pattern; must exist in `candidates.jsonl` |
| `rank` | int (1–100) | Each integer 1–100 used exactly once |
| `score` | float | Monotonically non-increasing with rank (score@rank_1 ≥ score@rank_2 ≥ … ≥ score@rank_100) |
| `reasoning` | string | Optional but **strongly recommended** — 1–2 sentences per candidate; used at Stage 4 scoring |

**Tie-breaking rule:** When two candidates share the same score, the one with the **lower `candidate_id` string** (ascending alphabetical) gets the better rank.

**Exactly 100 data rows** (plus 1 header row = 101 rows total).

### 5.2 Common Rejection Causes
- 99 or 101 rows (not exactly 100)
- Ranks starting at 0, or duplicate ranks
- Duplicate `candidate_id` values
- `candidate_id` values not present in `candidates.jsonl`
- All scores equal (not differentiating)
- Scores increasing as rank increases
- File submitted as `.xlsx` or `.json` instead of `.csv`

Run `python validate_submission.py <file>.csv` before every submission.

---

## 6. Compute Constraints

| Constraint | Limit |
|---|---|
| Total runtime (ranking step) | ≤ 5 minutes wall-clock |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only — **no GPU** |
| Network | **Off** — no external API calls during ranking (OpenAI, Anthropic, Cohere, Gemini, or any hosted model) |
| Intermediate disk | ≤ 5 GB |

> **Note:** Pre-computation (generating embeddings, building indexes) may exceed 5 minutes. Only the **ranking step** that produces the CSV must fit within the budget. Document pre-computation steps clearly in your README.

---

## 7. Evaluation

### 7.1 Scoring Formula (Stage 2)
Your top-100 ranking is scored against a **hidden ground truth** using a composite metric:

```
Composite = 0.50 × NDCG@10  +  0.30 × NDCG@50  +  0.15 × MAP  +  0.05 × P@10
```

| Metric | Weight | What it measures |
|---|---|---|
| NDCG@10 | 0.50 | Quality of your top-10 picks — most heavily weighted |
| NDCG@50 | 0.30 | Quality of your top-50 picks |
| MAP | 0.15 | Mean Average Precision across all relevance levels |
| P@10 | 0.05 | Fraction of top-10 that are "relevant" (relevance tier 3+) |

**Important:** P@10 is defined against **relevance tiers** (0–N) assigned by the ground truth. Honeypot candidates are forced to relevance tier 0. There is no partial-score feedback during the competition — scoring happens once, after submissions close.

**Composite score tiebreakers** (between two submissions with identical composites): Higher P@5 → Higher P@10 → Earlier submission timestamp.

### 7.2 Evaluation Pipeline (5 Stages)

| Stage | What happens | Elimination trigger |
|---|---|---|
| **1. Format validation** | Auto-validator (`validate_submission.py`) runs on every submission | Any spec violation |
| **2. Scoring** | Composite computed once on full hidden ground truth | Score below cutoff for Stage 3 advancement |
| **3. Code reproduction + honeypot check** | Full code repo requested; ranking step run in sandboxed Docker (5 min, 16 GB, no GPU, no network); honeypot rate computed | Cannot reproduce within limits; honeypot rate >10% in top 100; missing/fabricated repo |
| **4. Manual review** | Reasoning quality on 6 checks (see §7.3); methodology coherence; git history authenticity; code quality | Failed reasoning checks; flat git history (single dump); codebase is entirely LLM API calls |
| **5. Defend-your-work interview** | 30-minute video call with Redrob engineering — walk through architecture, defend design choices | Cannot explain architecture; contradicts code; clearly didn't build it |

### 7.3 Stage 4 Reasoning Quality Checks
10 random rows are sampled. Each reasoning string is checked for:

| Check | What evaluators look for |
|---|---|
| **Specific facts** | References specific profile data: years of experience, title, named skills, signal values |
| **JD connection** | Connects to specific JD requirements, not generic praise |
| **Honest concerns** | Acknowledges obvious gaps (e.g., long notice period, wrong work mode) |
| **No hallucination** | Every claim corresponds to something in the candidate's actual profile |
| **Variation** | 10 sampled reasonings are substantively different from each other — not templated |
| **Rank consistency** | Tone of reasoning matches rank position (rank-5 shouldn't sound critical; rank-95 shouldn't sound glowing) |

---

## 8. Dataset Traps (Critical)

### 8.1 Keyword Stuffers
Candidates whose `skills` section is loaded with AI/ML keywords but whose `current_title`, `career_history`, and `profile.summary` reveal an entirely different background (e.g., Marketing Manager). A naive embedding or keyword-similarity system will rank these highly. **Down-weight skills list in isolation; weight title and career history heavily.**

### 8.2 Plain-Language Tier 5 Candidates
Candidates who are strong fits for the JD but describe their work in plain language (e.g., "built a recommendation engine" not "deployed a vector database"). A system over-relying on keyword or embedding similarity to the JD text will miss these. **Semantic understanding of career descriptions is required.**

### 8.3 Behavioral Ghosts
Candidates with strong static profiles but dead engagement signals: `last_active_date` months or years ago, `recruiter_response_rate` near 0, `open_to_work_flag = false`. These are effectively unavailable. **Apply `redrob_signals` as a multiplicative modifier, not an additive one — a ghost should be penalised regardless of skill score.**

### 8.4 Behavioral Twins
Pairs (or clusters) of candidates with nearly identical static profiles — same title, similar experience, similar skill lists — who differ primarily in their `redrob_signals`. A system that ignores behavioural signals will assign them the same rank; the ground truth distinguishes them based on availability and engagement. **Signal-agnostic scoring will fail here.**

### 8.5 Honeypot Candidates (~80 records)
Candidates with subtly impossible profiles: e.g., 8 years of experience at a company founded 3 years ago; "expert" proficiency in 10 skills with 0 `duration_months` each. These are forced to **relevance tier 0** in the ground truth.

**Disqualification rule:** If your top-100 submission contains more than 10 honeypots (>10%), your submission is disqualified at Stage 3, regardless of composite score.

A well-designed ranking system catches these naturally via career timeline validation and skill proficiency vs. duration cross-checks. You do not need to special-case them, but you must not reward impossible profiles.

---

## 9. Submission Limit

**Maximum 3 submissions.** Your **last valid submission** is your final entry. Earlier submissions are not preserved. There is no live leaderboard; scores are revealed only after the competition closes.

---

## 10. Full Submission Package (All Three Parts Required)

### 10.1 CSV File
The top-100 ranking, as specified in §5.

### 10.2 GitHub Repository
Must include:
- `README.md` — setup instructions and **a single command** that produces the submission CSV from `candidates.jsonl`
- Full source code (no hidden steps, no manual edits between code and CSV)
- Pre-computed artifacts (embeddings, indexes, model weights) **or** a script that generates them
- `requirements.txt` (or `pyproject.toml`) with all dependencies and versions
- `submission_metadata.yaml` at repo root (use the provided `submission_metadata_template.yaml`)

### 10.3 Sandbox / Demo Link (Mandatory)
A hosted environment where your ranker can be run on a small sample (≤100 candidates). Accepted platforms: HuggingFace Spaces, Streamlit Cloud, Replit (public), Google Colab, Docker public image (`docker pull` + `docker run`), Binder.

The sandbox must: accept a small candidate sample as input, run your system end-to-end, produce a ranked CSV, complete within the compute budget.

> **Warning:** Submissions without a working sandbox link are flagged at **Stage 1**. This is not optional. If a hosted sandbox is impractical, a self-contained `docker run` recipe in your README is an acceptable substitute — but the Dockerfile must build and run unmodified.

---

## 11. AI Tools

AI tools (Claude, GPT-4, Copilot, Cursor, etc.) are **permitted**. Declare them honestly in `submission_metadata.yaml`. The evaluation pipeline is designed so that AI-assisted submissions where the human did real engineering pass Stages 3–5; AI-only paste-and-pray submissions fail at code reproduction, manual review, or the defend-your-work interview.

---

## 12. Core Engineering Challenge (Summary)

Building a high-scoring ranker requires solving these problems simultaneously:

1. **Semantic JD understanding** — extracting what the JD *means* (required skills, experience profile, cultural signals), not just its keywords.
2. **Career-history-first scoring** — title and career trajectory are higher-signal than the skills list. A skills-list-only scorer is explicitly a trap.
3. **Behavioural availability modelling** — skill-match score × availability multiplier derived from `redrob_signals`. A ghost candidate with perfect skills should rank behind an available candidate with slightly weaker skills.
4. **Anomaly rejection** — detecting honeypot candidates through timeline inconsistencies and skill-proficiency vs. duration mismatches.
5. **CPU-only latency** — the full ranking of 100K candidates must complete in ≤5 minutes on CPU with no network. This rules out per-candidate LLM inference and mandates either precomputed embeddings + ANN search or a compact feature-based scoring pipeline.
6. **Reasoning generation** — per-candidate justifications must be specific, grounded in the actual profile, and consistent with rank position. They are evaluated by human reviewers and will be verified against the Stage 5 interview.
