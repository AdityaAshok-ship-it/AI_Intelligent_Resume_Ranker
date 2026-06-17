# Redrob Ranking Challenge

## Reproduce command (single command, ≤5 min, ≤16 GB, no network)

```bash
python rank.py --out submission.csv
```

Requires `./artifacts/` to be populated by the precompute step (run once, offline).

---

## Pipeline overview

```
candidates.jsonl (487 MB)
    │
    ▼ precompute.py (run once, uncapped)
    │   • stream line-by-line (EC-10)
    │   • parse deterministic features → features.parquet
    │   • embed with bge-base-en-v1.5 (L2-norm) → candidate_matrix.npy
    │   • write row-aligned candidate_ids.npy
    │   • embed the fixed JD → jd_vector.npy
    │
    ▼ rank.py (≤5 min, CPU only, no network)
        1. load artifacts + assert alignment
        2. cosine(jd_vector, matrix) → sim[100K]  (~25 ms)
        3. rubric base score (0.65·career + 0.20·cosine + 0.10·skills + 0.05·edu)
        4. disqualifier caps D1–D5 (caps before multiplier)
        5. honeypot exclusion H1–H4 (hard-exclude before multiplier)
        6. availability multiplier (continuous, ghost-floor ~0.15)
        7. distinct-float guarantee
        8. sort ↓ → top-150 audit → take top 100
        9. fact-grounded reasoning (templates, no LLM)
       10. write CSV → validate_submission.py
```

## Setup

```bash
pip install -r requirements.txt

# Download model weights once (requires network):
python precompute.py --download-model

# Run precompute on the full corpus (run once; uncapped time):
python precompute.py --candidates /path/to/candidates.jsonl
```

## Phase 0 exit gate (parser eyeball check)

```bash
python precompute.py --candidates /path/to/sample_candidates.json --inspect
```

Dumps `is_eng_title`, `built_real_system`, `product_vs_services`, `honeypot_flag` and raw title/description for all 50 sample candidates. If any flag disagrees with what a human reads from the raw text, the extraction is wrong — fix before calibrating.

## Phase 1 calibration

```bash
python calibrate.py --sample /path/to/sample_candidates.json
```

Checks archetype `CAND_0000031` ranks #1–3, stuffer `CAND_0000021` is demoted far below it, and prints disqualifier gate firing rates.

## Repo structure

```
precompute.py          offline: embeddings + features + JD vector
rubric.py              base score + disqualifier caps + availability multiplier
detectors.py           H1–H4 honeypots + keyword-stuffer + top-150 audit
reasoning.py           fact-grounded templates + pairwise duplicate check
rank.py                in-budget ranking step → top-100 CSV
calibrate.py           hand-ranking harness + weight tuning on 50-sample
artifacts/
  candidate_matrix.npy  float32 (100K, 768)  ~307 MB
  candidate_ids.npy     row-index → candidate_id
  features.parquet      parsed per-candidate features
  jd_vector.npy         float32 (768,)
sandbox/app.py         Streamlit demo (≤100 sample, embeds live)
Dockerfile             Docker backup for sandbox (downloads model at build time)
decision.md            running decision log (most important file)
edgecases.md → ../docs/edgecases.md
```

## Phase 4 exit gate (reasoning validation)

```bash
python reasoning.py --validate submission.csv
```

Prints first 5 reasonings for human spot-read and runs pairwise near-duplicate check across all 100. Exits with code 1 if any pair exceeds the 0.85 similarity threshold.

## Sandbox (Stage 1 requirement)

**Primary:** HuggingFace Spaces (Streamlit) — link in `submission_metadata.yaml`.

**Backup — Docker run recipe:**

```bash
# Build (one-time; downloads bge-base-en-v1.5 weights ~440 MB):
docker build -t redrob-ranker .

# Run:
docker run -p 8501:8501 redrob-ranker
# Open http://localhost:8501 and upload a ≤100-candidate JSON/JSONL file.
```

The sandbox accepts a ≤100-candidate sample, embeds live with bge-base-en-v1.5 (bundled weights, no network at runtime), and displays the ranked output. It does **not** use the precomputed 100K matrix — it re-embeds from the uploaded file.

## Key decisions

See `decision.md` for the full log. Short answers to the most common Stage-5 questions:

| Question | Short answer |
|---|---|
| Why career history > skills list? | JD explicit thesis + keyword-stuffer trap (65% weight, skills at 10%) |
| Why reject skill-duration rule? | Fires on 13–19%, adds 0 honeypots over H1–H4, deletes the archetype |
| Why gentle multiplier not stack? | Stack flattens 96.6% healthy majority; NDCG@10 is half the composite |
| Why templates not a local LLM? | Determinism (Stage-3) + no hallucination risk vs marginal prose gain |
| Why is top-10 ordering right? | Reasoned, not tuned: written JD-traced tier-5 ordering rules in decision.md |
