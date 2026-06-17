#!/usr/bin/env python3
"""
rank.py — Phase 3: in-budget ranking step

Loads precomputed artifacts, applies the full scoring pipeline, writes a
validated top-100 CSV. No model is loaded here; network-off is structurally
satisfied.

Budget: ~1–2 s load · ~25 ms cosine · <1 s rubric/gates · <0.5 s sort+audit
        · <1 s reasoning+write+validate  →  seconds vs. 5-min cap, ~1 GB RAM.

Pipeline order (EC-45, EC-46 — caps BEFORE multiplier; exclusion BEFORE multiplier):
    1. load artifacts (assert alignment EC-11)
    2. cosine(jd_vector, candidate_matrix) → sim[N]
    3. rubric base score (rubric.py)
    4. disqualifier caps → gated  (rubric.py; most-restrictive wins)
    5. honeypot exclusion → drop H1–H4 rows (detectors.py; hard-exclude)
    6. availability multiplier over surviving rows → final = gated × mult
    7. sort final ↓ → top 150 for audit
    8. distinct-float guarantee: descending offsets on top-150 window (EC-47, EC-48)
    9. top-150 audit guard (detectors.py)
   10. fact-grounded reasoning (reasoning.py)
   11. write CSV → validate_submission.py

Usage:
    python rank.py --out submission.csv          # full run (requires precomputed matrix)
    python rank.py --out submission.csv --no-embed  # dev mode: zero cosine (no matrix needed)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("artifacts")


def load_artifacts(no_embed: bool = False):
    """
    Load precomputed artifacts and assert EC-11 alignment invariant.

    no_embed=True: skip matrix and jd_vector (development/test mode — cosine=0 for all).
    This lets the pipeline run end-to-end without the 307 MB embedding matrix.
    """
    ids_arr = np.load(ARTIFACTS_DIR / "candidate_ids.npy", allow_pickle=True)
    features = pd.read_parquet(ARTIFACTS_DIR / "features.parquet")

    # Alignment assertion (EC-11): every rank depends on this being correct.
    assert list(features["candidate_id"]) == list(ids_arr), (
        "Alignment failure: features.parquet order does not match candidate_ids.npy"
    )

    if no_embed:
        return None, ids_arr, features, None

    matrix_path = ARTIFACTS_DIR / "candidate_matrix.npy"
    jd_path = ARTIFACTS_DIR / "jd_vector.npy"
    if not matrix_path.exists() or not jd_path.exists():
        print(
            "ERROR: candidate_matrix.npy or jd_vector.npy not found in artifacts/.\n"
            "Run: python precompute.py --candidates /path/to/candidates.jsonl\n"
            "Or use --no-embed for a development run without embeddings.",
            file=sys.stderr,
        )
        sys.exit(1)

    matrix = np.load(matrix_path)
    jd_vec = np.load(jd_path)

    assert matrix.shape[0] == len(ids_arr), (
        f"Alignment failure: matrix rows ({matrix.shape[0]}) != ids ({len(ids_arr)})"
    )
    assert matrix.shape[1] == 768, f"Expected 768-d matrix, got {matrix.shape[1]}"
    assert jd_vec.shape == (768,), f"Expected (768,) JD vector, got {jd_vec.shape}"

    return matrix, ids_arr, features, jd_vec


def main() -> None:
    parser = argparse.ArgumentParser(description="In-budget ranking step → top-100 CSV")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument(
        "--validate", action="store_true", default=True,
        help="Run validate_submission.py after writing (default: True)",
    )
    parser.add_argument(
        "--no-embed", action="store_true", default=False,
        help=(
            "Development mode: skip loading candidate_matrix.npy / jd_vector.npy "
            "and use cosine_sim=0 for all candidates. "
            "Produces a valid but embedding-free submission. "
            "Use only for pipeline testing; the full run requires precomputed artifacts."
        ),
    )
    args = parser.parse_args()

    # ── 1. Load artifacts ─────────────────────────────────────────────────────

    print("Loading artifacts ...")
    matrix, ids_arr, features, jd_vec = load_artifacts(no_embed=args.no_embed)
    n = len(ids_arr)
    if args.no_embed:
        print(f"  {n:,} candidates — no-embed mode (cosine=0; 20% component suppressed)")
    else:
        print(f"  {n:,} candidates, matrix {matrix.shape}, jd_vec {jd_vec.shape}")

    # ── 2. Cosine similarity ──────────────────────────────────────────────────

    print("Computing cosine similarity ...")
    if args.no_embed:
        # Development mode: cosine contribution is 0 for all candidates.
        # Career (65%) + skills (10%) + edu (5%) + logistics still rank correctly.
        sim = np.zeros(n, dtype=np.float64)
    else:
        # matrix and jd_vec are L2-normalised at precompute time → cosine = dot product
        sim = matrix @ jd_vec  # shape (N,)

    # ── 3–4. Rubric base + disqualifier caps ─────────────────────────────────

    from rubric import compute_base, apply_disqualifier_caps   # Phase 1 deliverable
    print("Scoring rubric ...")
    base = compute_base(features, sim)
    gated = apply_disqualifier_caps(base, features)

    # ── 5. Honeypot exclusion ─────────────────────────────────────────────────

    from detectors import get_honeypot_mask, get_stuffer_flag  # Phase 2 deliverables
    print("Applying honeypot exclusion ...")
    honeypot_mask = get_honeypot_mask(features)  # True = flagged → exclude
    surviving_mask = ~honeypot_mask
    n_excluded = honeypot_mask.sum()
    print(f"  Excluded {n_excluded} honeypots; {surviving_mask.sum():,} surviving")

    # Stuffer count diagnostic — quote THIS number in the interview, not the doc's 3.6%
    stuffer_mask = get_stuffer_flag(features)
    n_stuffers = int(stuffer_mask.sum())
    print(
        f"  Keyword-stuffer flag: {n_stuffers:,} / {len(features):,} "
        f"({100 * n_stuffers / len(features):.1f}%) — flagged only, "
        f"career-first 65% weight already demotes these"
    )

    # ── 6. Availability multiplier ────────────────────────────────────────────

    from rubric import compute_availability_multiplier
    print("Applying availability multiplier ...")
    multiplier = compute_availability_multiplier(features)
    final = np.where(surviving_mask, gated * multiplier, -np.inf)

    # ── 7. Sort ↓ → top 150 for audit ────────────────────────────────────────────

    top_indices = np.argsort(final)[::-1][:150]   # top 150 for audit

    # ── 8. Distinct-float guarantee (post-sort, descending offsets) ───────────────
    # Rank k (0-indexed, best first) receives offset (N-k)*epsilon.
    # Best candidate gets the LARGEST positive offset → remains ranked first.
    # All 150 scores become strictly distinct while preserving sort order (EC-47, EC-48).
    # Applied post-sort to avoid the pre-sort variant's risk: index-based offsets on
    # an unsorted array can flip pairs closer than (max_index_gap * epsilon) ≈ 1e-4.
    N_window = len(top_indices)
    for k, idx in enumerate(top_indices):
        final[idx] += (N_window - k) * 1e-9

    # ── 9. Top-150 audit guard ────────────────────────────────────────────────

    from detectors import top_150_audit
    top_indices = top_150_audit(top_indices, features, ids_arr)
    top100_indices = top_indices[:100]

    # Score digest for the post-audit top-10 (helps spot "too good to be true" scores)
    print("Top-10 score digest (post-audit):")
    for rank_pos, idx in enumerate(top100_indices[:10], 1):
        cid = str(ids_arr[idx])
        sc = float(final[idx])
        row = features.iloc[idx]
        print(
            f"  #{rank_pos:2d} {cid}  score={sc:.4f}  "
            f"eng={row['is_eng_title']}  built={row['built_real_system']}  "
            f"stale={int(row['staleness_days'])}d"
        )

    # ── 10. Reasoning ─────────────────────────────────────────────────────────

    from reasoning import generate_reasoning
    print("Generating reasoning ...")
    reasonings = generate_reasoning(top100_indices, features, final)

    # ── 11. Write CSV + validate ──────────────────────────────────────────────

    import csv
    out_path = Path(args.out)
    print(f"Writing {out_path} ...")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_pos, idx in enumerate(top100_indices, start=1):
            cid = str(ids_arr[idx])
            score = float(final[idx])
            reason = reasonings[rank_pos - 1]
            writer.writerow([cid, rank_pos, repr(score), reason])  # repr keeps precision (EC-48)

    print(f"Written {out_path}")

    if args.validate:
        print("Running validate_submission.py ...")
        result = subprocess.run(
            [sys.executable, "validate_submission.py", str(out_path)],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            sys.exit(result.returncode)

    print("Done.")


if __name__ == "__main__":
    main()
