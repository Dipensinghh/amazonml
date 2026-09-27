"""One-command fallback submission: score test candidates, write outputs, validate.

  python src/run_fallback.py --candidates <test_candidates.parquet>

Steps: load candidates (s1_id, cand_id) -> rule score with the tuned weights
(src/fallback_params.json) -> ownership.decide(t) -> write_outputs -> checks.py
(strict) -> organizer validator with --check-ids. Exit code 0 only if both pass.
Paths default to the main checkout (--root) so it runs from a worktree too.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from checks import check_outputs
from common import peak_ram_gb
from fallback import PARAMS_PATH, combine, load_features, pair_sims
from io_utils import dedupe_pairs, write_outputs
from ownership import decide

DEFAULT_ROOT = Path(r"C:\Users\sdipe\amazonmlchallenge")


def main() -> int:
    """Run the whole fallback pipeline on test candidates; return the exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, help="test candidate parquet with s1_id, cand_id")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="checkout with dataset/, artifacts/, utils/")
    ap.add_argument("--params", default=str(PARAMS_PATH), help="tuned weights + t (JSON)")
    ap.add_argument("--out", help="output dir (default <root>/output)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    root = Path(a.root)
    art, test_dir = root / "artifacts", root / "dataset" / "test"
    out = Path(a.out) if a.out else root / "output"
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    w, t = tuple(params["weights"]), float(params["t"])

    def log(msg):
        print(f"[{time.perf_counter() - t0:6.0f}s | peak {peak_ram_gb():.1f} GB] {msg}", flush=True)

    log(f"params: weights={w} t={t} (val F0.5 {params.get('val_f05')})")
    cand_path = Path(a.candidates)
    if not cand_path.is_absolute() and not cand_path.exists():
        cand_path = root / cand_path
    cands = pq.read_table(cand_path, columns=["s1_id", "cand_id"])
    cands = dedupe_pairs(cands)
    s1_ids = pq.read_table(art / "test_s1_norm.parquet", columns=["entity_id"])["entity_id"]
    log(f"candidates {cands.num_rows:,} pairs for {len(pc.unique(cands['s1_id'])):,} S1 "
        f"({cands.num_rows / len(s1_ids):.1f} per test S1)")

    s1_feat = load_features([art / "test_s1_norm.parquet"], s1_ids.combine_chunks())
    tgt_feat = load_features([art / "test_s2_norm.parquet", art / "test_s3_norm.parquet"],
                             pc.unique(cands["cand_id"]))
    log("features loaded")
    sims = pair_sims(cands, s1_feat, tgt_feat, log=log)
    del s1_feat, tgt_feat
    score = combine(sims, w)
    del sims
    # decide() drops everything below t anyway; filtering first keeps pandas small
    keep = pa.array(score >= t)
    scored = cands.filter(keep).append_column("score", pa.array(score).filter(keep)).to_pandas()
    matches = decide(scored, t)
    log(f"matches {len(matches):,} ({scored.shape[0] - len(matches):,} dropped by ownership), "
        f"{matches['s1_id'].nunique():,} S1 with >=1 match")

    mpath, cpath = write_outputs(s1_ids, cands, matches, out)
    log(f"wrote {mpath} and {cpath}")
    rep = check_outputs(mpath, cpath, test_dir)
    log("checks.py: " + ("PASS" if rep.ok else "FAIL\n  " + "\n  ".join(rep.lines())))
    r = subprocess.run([sys.executable, str(root / "utils" / "validate_submission.py"),
                        "--matching", str(mpath), "--candidate", str(cpath),
                        "--test-dir", str(test_dir), "--check-ids"], capture_output=True, text=True)
    print(r.stdout + r.stderr, end="")
    log(f"organizer validator exit {r.returncode}")
    return 0 if rep.ok and r.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
