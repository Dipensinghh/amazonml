"""Frozen seed-42 80/20 split of train S1 entities (validation S1s are searched against ALL train S2/S3)."""
import argparse
import gzip
from pathlib import Path

import math

import numpy as np

SEED = 42
VAL_FRAC = 0.2


def make_split(s1_ids, seed: int = SEED, val_frac: float = VAL_FRAC):
    """Return (train_ids, val_ids) as sorted lists; deterministic for a given id set and seed."""
    ids = np.array(sorted(s1_ids))
    perm = np.random.RandomState(seed).permutation(len(ids))
    n_val = math.ceil(len(ids) * val_frac)
    val = ids[perm[:n_val]]
    tr = ids[perm[n_val:]]
    return sorted(tr.tolist()), sorted(val.tolist())


def load_val_ids(path: str = "splits/val_s1_ids.txt.gz") -> set:
    """Load the committed validation S1 id set."""
    with gzip.open(path, "rt") as f:
        return {l.strip() for l in f if l.strip()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", default="student_resource/dataset/train/train_source1.tsv")
    ap.add_argument("--out", default="splits/val_s1_ids.txt.gz")
    a = ap.parse_args()
    with open(a.s1, encoding="utf-8") as f:
        next(f)
        ids = [l.split("\t", 1)[0] for l in f if l.strip()]
    tr, val = make_split(ids)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt") as f:
        f.write("\n".join(val) + "\n")
    print(f"total={len(ids)} train={len(tr)} val={len(val)} -> {a.out}")
