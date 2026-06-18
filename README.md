# Redrob Ranking Challenge — Intelligent Resume Ranker

A career-first candidate ranking system for Redrob's AI Engineer job description. Ranks a corpus of ~100K candidates and outputs a top-100 CSV with scores and fact-grounded reasoning — fully reproducible offline, no LLM at inference time.

---

## Quick start

```bash
# One-time setup
pip install -r requirements.txt
python precompute.py --download-model        # downloads bge-base-en-v1.5 (~440 MB)
python precompute.py --candidates /path/to/candidates.jsonl   # ~45 min, run once

# Reproduce submission (≤5 min, CPU only, no network)
python rank.py --out submission.csv
```

---

## Pipeline overview

```
candidates.jsonl (~100K records)
    │
    ▼ precompute.py  (run once, uncapped time)
    │   • stream line-by-line
    │   • parse deterministic features → features.parquet
    │   • embed with bge-base-en-v1.5 (768-d, L2-normalised) → candidate_matrix.npy
    │   • write row-aligned candidate_ids.npy
    │   • embed fixed JD → jd_vector.npy
    │
    ▼ rank.py  (≤5 min, CPU only, no network)
        1. load artifacts + assert row alignment
        2. cosine(jd_vector, matrix) → sim[100K]          (~25 ms)
        3. rubric base score
           0.70 × career  +  0.20 × cosine  +  0.10 × skills  +  logistics
        4. disqualifier caps D1–D6 + visa (most-restrictive wins; applied before multiplier)
        5. honeypot exclusion H1–H5 + skill-anachronism & education gates (hard pre-sort exclusion)
        6. availability multiplier (continuous; ghost-floor ~0.15)
        7. distinct-float guarantee
        8. sort ↓ → top-150 audit window → take top 100
        9. fact-grounded reasoning (template-based, no LLM)
       10. write CSV → validate_submission.py
```

---

## Scoring architecture

### Base score

| Component | Weight | Signal |
|---|---|---|
| Career evidence | **0.70** | Engineering title, shipped production systems, product-company experience, YOE sweet spot (5–9), IC recency |
| Semantic similarity | **0.20** | Cosine similarity to JD embedding (bge-base-en-v1.5) |
| Skills | **0.10** | Redrob assessment scores + GitHub activity (low weight to resist keyword stuffing) |
| ~~Education~~ | **removed** | Dropped 2026-06-18 — the JD names no education/institution-tier criterion (only Indian cities); its 0.05 was folded into career. |
| Logistics | additive | Notice period, India-based, Tier-1 city, relocation, work-mode preference |

### Disqualifier caps (D1–D6 + location)

Applied to base score before the availability multiplier. Most-restrictive cap wins.

| Code | Description | Cap |
|---|---|---|
| D1 | Pure research, zero production deployment | 0.02 |
| D2 | AI experience only in recent LLM role, no prior ML history | 0.40 |
| D3 | No hands-on IC engineering in last 18 months | 0.45 |
| D4 | Entire career in IT services / consulting, no product-company role | 0.35 |
| D5 | CV / speech / robotics primary domain, no NLP or IR evidence | 0.50 |
| D6 | Title-chaser / job-hopper (≥3 employers, <18-mo avg tenure) — JD wants a 3+ yr commitment | 0.50 |
| location | Non-India **and** not willing to relocate — JD: "we don't sponsor work visas" | 0.55 |

### Honeypot exclusion (H1–H5) + discrepancy gates

Five timeline-impossibility rules (H5 = a role at a real company starting before its founding year) hard-exclude fabricated candidates before any score is computed. Two further impossibility gates also hard-exclude: **skill-anachronism** (a named technology claimed for longer than it has existed) and **education-integrity** (reversed/overlapping degree timelines). All pre-computed in `precompute.py` and stored in `features.parquet`; `detectors.py` reads the pre-computed flags.

### Availability multiplier

Continuous multiplier ∈ [0.15, 1.0] derived from recruiter response rate, staleness, and open-to-work flag. Ghost conjunction (stale + unresponsive + not open) floors the multiplier at ~0.15; genuinely engaged candidates receive ≥ 0.70.

---

## Repo structure

```
precompute.py           offline: embeddings + features + JD vector
rubric.py               base score, disqualifier caps, availability multiplier
detectors.py            H1–H5 honeypot gate + anachronism/education gates, keyword-stuffer flag, top-150 audit
reasoning.py            fact-grounded reasoning templates + pairwise duplicate check
rank.py                 in-budget ranking step → top-100 CSV
calibrate.py            hand-ranking harness (50-sample) for weight validation
validate_submission.py  submission format check

artifacts/
  candidate_matrix.npy  float32 (100K × 768)  ~307 MB
  candidate_ids.npy     row-index → candidate_id mapping
  features.parquet      parsed per-candidate features
  jd_vector.npy         float32 (768,)

sandbox/app.py          Streamlit demo (≤100 sample, embeds live)
Dockerfile              Docker backup for sandbox
submission_metadata.yaml  submission metadata
```

---

## Development mode

Run the full pipeline without the precomputed embedding matrix (cosine component zeroed):

```bash
python rank.py --out submission.csv --no-embed
```

Useful for testing the rubric, disqualifier, and reasoning logic without the ~307 MB matrix.

---

## Exit gates

### Phase 0 — parser eyeball check

```bash
python precompute.py --candidates /path/to/sample_candidates.json --inspect
```

Dumps `is_eng_title`, `built_real_system`, `product_vs_services`, `honeypot_flag` and raw title/description for all 50 sample candidates. Verify each flag against the raw text before proceeding.

### Phase 1 — calibration

```bash
python calibrate.py --sample /path/to/sample_candidates.json
```

Checks the archetype candidate ranks #1–3, a known stuffer is demoted far below it, and prints disqualifier gate firing rates across the sample.

### Phase 4 — reasoning validation

```bash
python reasoning.py --validate submission.csv
```

Prints the first 5 reasonings for human spot-read and runs a pairwise near-duplicate check across all 100. Exits with code 1 if any pair exceeds the similarity threshold.

---

## Sandbox

**Primary:** HuggingFace Spaces (Streamlit) — link in `submission_metadata.yaml`.

**Backup — Docker:**

```bash
# Build (downloads bge-base-en-v1.5 weights ~440 MB):
docker build -t redrob-ranker .

# Run:
docker run -p 8501:8501 redrob-ranker
# Open http://localhost:8501 and upload a ≤100-candidate JSON/JSONL file.
```

The sandbox accepts a ≤100-candidate sample, re-embeds live (no precomputed matrix), and displays the ranked output.

---

## Compute constraints

| Constraint | Value |
|---|---|
| Reproduce budget | ≤ 5 minutes, ≤ 16 GB RAM |
| Network at rank time | None |
| Model at rank time | None (embeddings precomputed) |
| Pre-computation (one-time) | ~45 min, uncapped |
| Embedding model | `bge-base-en-v1.5` (768-d, ~440 MB) |

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.11+. No GPU required.
