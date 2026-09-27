"""Score full test features, resolve target ownership, write and validate both TSVs.

Requires the integrated A/D modules (`io_utils.py`, `checks.py`, `predict.py`). The score
Parquet preserves the candidate row order and contains every pair, while ownership is reduced
globally over above-threshold rows.

Usage:
  python -u src/score_test.py --model artifacts/lgbm_full.txt --bundle artifacts/bundle.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from itertools import zip_longest
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_bundle(path: Path, threshold: float | None) -> tuple[list[str], float]:
    """Return the ordered model features and selected threshold."""
    bundle = json.loads(path.read_text(encoding="utf-8"))
    features = bundle.get("features")
    if not isinstance(features, list) or not features or len(features) != len(set(features)):
        raise ValueError("bundle.features must be a non-empty ordered list with no duplicates")
    selected = bundle.get("threshold") if threshold is None else threshold
    if not isinstance(selected, (int, float)) or not 0 <= selected <= 1:
        raise ValueError("threshold must be in [0,1]")
    return features, float(selected)


def validator_path(root: Path, value: str | None) -> Path:
    """Resolve the organizer validator from an explicit path or the local resource folder."""
    if value:
        path = Path(value)
        return path if path.is_absolute() else root / path
    choices = [root / "utils" / "validate_submission.py",
               (root / "dataset").resolve().parent / "utils" / "validate_submission.py"]
    found = next((p for p in choices if p.exists()), None)
    if found is None:
        raise FileNotFoundError("validate_submission.py not found; pass --validator")
    return found


def score_batches(feature_path: Path, candidate_path: Path, booster: lgb.Booster,
                  features: list[str], threshold: float, scored_path: Path,
                  batch_size: int):
    """Write all scores in input order and yield above-threshold rows for ownership."""
    fpf, cpf = pq.ParquetFile(feature_path), pq.ParquetFile(candidate_path)
    if fpf.metadata.num_rows != cpf.metadata.num_rows:
        raise ValueError("feature and candidate row counts differ")
    required = ["s1_id", "cand_id", *features]
    missing = set(required) - set(fpf.schema_arrow.names)
    if missing:
        raise ValueError(f"feature parquet missing bundle columns: {sorted(missing)}")
    model_features = booster.feature_name()
    if model_features and model_features != features:
        raise ValueError("model feature order differs from bundle.features")

    scored_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = scored_path.with_suffix(scored_path.suffix + ".tmp")
    writer = pq.ParquetWriter(tmp_path, pa.schema([
        ("s1_id", pa.string()), ("cand_id", pa.string()), ("p", pa.float32())]),
        compression="zstd")
    rows, complete = 0, False
    try:
        fit = fpf.iter_batches(batch_size=batch_size, columns=required)
        cit = cpf.iter_batches(batch_size=batch_size, columns=["s1_id", "cand_id"])
        for fb, cb in zip_longest(fit, cit):
            if fb is None or cb is None or fb.num_rows != cb.num_rows:
                raise ValueError("feature and candidate batch boundaries differ")
            if not fb["s1_id"].equals(cb["s1_id"]) or not fb["cand_id"].equals(cb["cand_id"]):
                raise ValueError(f"feature/candidate pair order differs near row {rows:,}")
            x = pa.Table.from_batches([fb]).select(features).to_pandas()
            p = np.asarray(booster.predict(x), dtype=np.float32)
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError(f"invalid model probabilities near row {rows:,}")
            scored = pa.table({"s1_id": fb["s1_id"], "cand_id": fb["cand_id"], "p": p})
            writer.write_table(scored)
            keep = p >= threshold
            if keep.any():
                yield scored.filter(pa.array(keep)).to_pandas()
            rows += fb.num_rows
            print(f"  scored {rows:,}/{fpf.metadata.num_rows:,}", flush=True)
        complete = True
    finally:
        writer.close()
        if complete:
            os.replace(tmp_path, scored_path)


def main() -> int:
    root_default = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(root_default))
    ap.add_argument("--features", default="artifacts/cand_test_features.parquet")
    ap.add_argument("--candidates", default="artifacts/cand_test.parquet")
    ap.add_argument("--model", default="artifacts/lgbm_full.txt")
    ap.add_argument("--bundle", default="artifacts/bundle.json")
    ap.add_argument("--threshold", type=float, help="override bundle threshold")
    ap.add_argument("--scored", default="artifacts/scored_test.parquet")
    ap.add_argument("--out", default="output")
    ap.add_argument("--validator", help="path to organizer validate_submission.py")
    ap.add_argument("--batch-size", type=int, default=500_000)
    args = ap.parse_args()
    if args.batch_size <= 0:
        ap.error("--batch-size must be positive")

    # These arrive from A and D during integration; keeping imports here lets this C-only branch compile.
    from checks import check_outputs
    from io_utils import write_outputs
    from predict import assign_owners, reduce_owners_streaming

    root = Path(args.root).resolve()
    resolve = lambda p: Path(p) if Path(p).is_absolute() else root / p
    feature_path, candidate_path = resolve(args.features), resolve(args.candidates)
    model_path, bundle_path = resolve(args.model), resolve(args.bundle)
    scored_path, out_dir = resolve(args.scored), resolve(args.out)
    validator = validator_path(root, args.validator)
    features, threshold = load_bundle(bundle_path, args.threshold)
    booster = lgb.Booster(model_file=str(model_path))

    t0 = time.perf_counter()
    winners = reduce_owners_streaming(score_batches(
        feature_path, candidate_path, booster, features, threshold, scored_path, args.batch_size))
    matches = assign_owners(winners, threshold)
    del winners
    candidates = pq.read_table(candidate_path, columns=["s1_id", "cand_id"])
    s1_ids = pq.read_table(root / "artifacts" / "test_s1_norm.parquet",
                           columns=["entity_id"])["entity_id"]
    matching_path, candidate_out = write_outputs(s1_ids, candidates, matches, out_dir)
    print(f"wrote {len(matches):,} matches; scoring + ownership + output "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    report = check_outputs(matching_path, candidate_out, root / "dataset" / "test")
    if not report.ok:
        print("checks.py: FAIL\n  " + "\n  ".join(report.lines()))
        return 1
    print("checks.py: PASS", flush=True)
    result = subprocess.run([
        sys.executable, str(validator), "--matching", str(matching_path),
        "--candidate", str(candidate_out), "--test-dir", str(root / "dataset" / "test"),
        "--check-ids"], text=True, capture_output=True)
    print(result.stdout + result.stderr, end="")
    if result.returncode:
        print(f"organizer validator: FAIL ({result.returncode})")
        return result.returncode
    print("organizer validator: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
