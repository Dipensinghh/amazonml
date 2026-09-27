"""Reproduce the entity-resolution pipeline one stage at a time or end to end."""
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["normalize", "candidates", "features", "train", "score", "all"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=500_000)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    src = root / "src"
    if not src.exists():
        raise SystemExit(f"missing {src}; copy the integrated src/ directory into the package")

    py = sys.executable
    commands = {
        "normalize": [[py, "-u", str(src / "normalize.py"), "--split", split]
                      for split in ("train", "test")],
        "candidates": [[py, "-u", str(src / "blocking.py"), "--produce", name,
                        "--jobs", str(args.jobs)] for name in ("val50k", "fit200k", "test")],
        "features": [[py, "-u", str(src / "features.py"), "--candidates",
                      f"artifacts/cand_{name}.parquet", "--split", split, "--batch",
                      str(args.batch_size), "--out", f"artifacts/cand_{name}_features.parquet",
                      *(["--labels"] if split == "train" else [])]
                     for name, split in (("fit200k", "train"), ("val50k", "train"),
                                         ("test", "test"))],
        "train": [[py, "-u", str(src / "train.py"), "--features",
                   "artifacts/cand_fit200k_features.parquet", "--extra-features",
                   "artifacts/cand_val50k_features.parquet", "--scope-sample", "50000",
                   "--refit-full"]],
        "score": [[py, "-u", str(src / "score_test.py"), "--batch-size", str(args.batch_size)]],
    }
    stages = list(commands) if args.stage == "all" else [args.stage]
    for stage in stages:
        for command in commands[stage]:
            print("+", " ".join(command), flush=True)
            subprocess.run(command, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
