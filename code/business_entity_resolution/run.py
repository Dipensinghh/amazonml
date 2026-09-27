"""Reproduce the entity-resolution pipeline one stage at a time or end to end.

Stages (in order): normalize, candidates, fit_extra, features, competition, apply_competition,
train, report, score.  `all` runs them in that order.

The model used for scoring is one config value, MODEL_DIR (override with --model-dir). The
shipped bundle is models/v3. `train` refuses to overwrite an existing bundle unless
--overwrite is given, so `all` scores with the shipped model unless you retrain into a new
directory, e.g.:  python -u run.py train --model-dir models/v3_retrained
"""
import argparse
import subprocess
import sys
from pathlib import Path

MODEL_DIR = "models/v3"  # the single place that names the scoring model

ART = "artifacts"
FIT_EXTRA_N = 400_000  # v3: fit600k = fit200k + 400k extra fitting S1s
TRAIN_SETS = ["fit200k", "fit400k_extra", "val50k"]  # labelled feature sets (train split)
V3_TRAIN = ["--num-leaves", "255", "--n-estimators", "3000", "--learning-rate", "0.05",
            "--early-stop", "100"]  # 43 features; min_child_samples 50, reg_lambda 1 (train.PARAMS)
STAGES = ["normalize", "candidates", "fit_extra", "features", "competition",
          "apply_competition", "train", "report", "score"]


def commands(a, root: Path) -> dict:
    """Stage name -> list of commands (argv lists, run from the package root)."""
    py, src, md = sys.executable, root / "src", a.model_dir
    feat = lambda n: f"{ART}/cand_{n}_features.parquet"
    full = lambda n: f"{ART}/cand_{n}_features_fullrev.parquet"
    return {
        "normalize": [[py, "-u", str(src / "normalize.py"), "--split", s] for s in ("train", "test")],
        "candidates": [[py, "-u", str(src / "blocking.py"), "--produce", n, "--jobs", str(a.jobs)]
                       for n in ("val50k", "fit200k", "trainall", "test")],
        "fit_extra": [[py, "-u", str(src / "fit_extra.py"), "--n", str(FIT_EXTRA_N)]],
        "features": [[py, "-u", str(src / "features.py"), "--candidates", f"{ART}/cand_{n}.parquet",
                      "--split", "train", "--labels", "--batch", str(a.batch_size), "--out", feat(n)]
                     for n in TRAIN_SETS]
                    + [[py, "-u", str(src / "features.py"), "--candidates", f"{ART}/cand_test.parquet",
                        "--split", "test", "--batch", str(a.batch_size), "--out", feat("test")]],
        "competition": [[py, "-u", str(src / "competition.py"), "--name", n] for n in ("trainall", "test")],
        "apply_competition": [[py, "-u", str(src / "apply_competition.py"), "--features", feat(n),
                               "--competition", f"{ART}/competition_trainall.parquet", "--out", full(n)]
                              for n in TRAIN_SETS]
                             + [[py, "-u", str(src / "apply_competition.py"), "--features", feat("test"),
                                 "--competition", f"{ART}/competition_test.parquet", "--out", full("test")]],
        "train": [[py, "-u", str(src / "train.py"), "--features", full("fit200k"),
                   "--extra-features", full("fit400k_extra"), full("val50k"),
                   "--scope-sample", "50000", "--out-dir", md, "--n-jobs", str(a.threads), *V3_TRAIN]],
        "report": [[py, "-u", str(src / "val_report.py"), "--model", md, "--features-val", full("val50k"),
                    "--cand", f"{ART}/cand_val50k.parquet", "--scope-sample", "50000", "--k", "5",
                    "--out", f"{md}/val_report.json"]],
        "score": [[py, "-u", str(src / "score_test.py"), "--features", full("test"),
                   "--candidates", f"{ART}/cand_test.parquet", "--model", f"{md}/lgbm.txt",
                   "--bundle", f"{md}/bundle.json", "--validator", "utils/validate_submission.py",
                   "--batch-size", str(a.batch_size)]],
    }


def main() -> int:
    """Parse arguments and run the requested stage(s), stopping at the first failure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*STAGES, "all"])
    ap.add_argument("--model-dir", default=MODEL_DIR, help=f"model bundle directory (default {MODEL_DIR})")
    ap.add_argument("--overwrite", action="store_true", help="let `train` replace an existing bundle")
    ap.add_argument("--jobs", type=int, default=8, help="blocking worker processes")
    ap.add_argument("--threads", type=int, default=16, help="LightGBM threads (v3 used 16)")
    ap.add_argument("--batch-size", type=int, default=500_000, help="feature / scoring batch rows")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    if not (root / "src").exists():
        raise SystemExit(f"missing {root / 'src'}")
    cmds = commands(a, root)
    for stage in (STAGES if a.stage == "all" else [a.stage]):
        if stage == "train" and (root / a.model_dir / "bundle.json").exists() and not a.overwrite:
            if a.stage == "all":
                print(f"= skip train: {a.model_dir}/bundle.json exists (shipped model); "
                      "use --model-dir <new dir> or --overwrite to retrain", flush=True)
                continue
            raise SystemExit(f"{a.model_dir}/bundle.json exists; use --model-dir <new dir> or --overwrite")
        for cmd in cmds[stage]:
            print("+", " ".join(str(c) for c in cmd), flush=True)
            if not a.dry_run:
                subprocess.run(cmd, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
