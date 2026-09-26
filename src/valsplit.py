"""Shared validation split and data paths for the D-role modules (train / predict / fallback).

The split is the team's single source of truth: artifacts/split_train.parquet (entity_id, is_val),
written by blocking.make_split (seed 42). If it is missing it is rebuilt with the identical recipe,
so candidates from blocking.py and the model always agree on which S1s are validation.
"""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common import ART, DATA, ROOT, SEED

# Team layout keeps data in <repo>/dataset; on the Mac the organizer folder is <repo>/student_resource/dataset.
DATA_DIR = DATA if DATA.exists() else ROOT / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"


def load_val_ids(split_path: Path = ART / "split_train.parquet") -> set:
    """Validation S1 ids from the shared split file; rebuilt with blocking.py's recipe if the file is absent."""
    if not Path(split_path).exists():
        ids = np.sort(np.array([l.split("\t", 1)[0] for l in open(TRAIN_DIR / "train_source1.tsv", encoding="utf-8")][1:]))
        rng = np.random.default_rng(SEED)
        val = ids[rng.permutation(len(ids))[: int(round(0.2 * len(ids)))]]
        return set(val.tolist())
    t = pq.read_table(split_path).to_pandas()
    return set(t.loc[t["is_val"], "entity_id"])


def sample_val_scope(n: int = 50_000) -> set:
    """The S1s blocking evaluated: blocking.sample_ids on the sorted validation ids (seed 42, n draws).

    Includes S1s that ended up with zero candidates, so macro F0.5 counts them.
    """
    ids = np.sort(np.array(sorted(load_val_ids())))
    if n <= 0 or n >= len(ids):
        return set(ids.tolist())
    return set(np.sort(np.random.default_rng(SEED).choice(ids, size=n, replace=False)).tolist())
