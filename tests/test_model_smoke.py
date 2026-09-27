"""Regression smoke test: train.py bundle -> score_test.py -> TSVs -> checks.py + organizer validator.

Guards the train/score contract (bundle.json features + threshold, model files, feature order
selected by name, global ownership) on ~2,000 synthetic pairs. Runs in a pytest temp folder
only; never touches artifacts/ or output/. Needs the organizer validator: set VALIDATOR=<path>
when utils/ (gitignored) is not in this checkout, otherwise the test is skipped.

Run: python -m pytest tests/test_model_smoke.py -q
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

import features as F  # noqa: E402
import train as T  # noqa: E402
from ownership import decide  # noqa: E402

COLS = F.BASE_FEATURES + F.CORE_RETRIEVAL + F.V3_EXTRA  # the 43 v3 model inputs
VALIDATOR = Path(os.environ.get("VALIDATOR", REPO / "utils" / "validate_submission.py"))


def _feats(rng, y: np.ndarray) -> dict:
    """43 float32 columns; a few carry (noisy) signal about y."""
    out = {c: rng.random(len(y)).astype(np.float32) for c in COLS}
    for c in ("name_core_token_set", "addr_norm_tfidf", "retr_score", "acronym_match"):
        out[c] = np.clip(0.4 * y + 0.6 * rng.random(len(y)), 0, 1).astype(np.float32)
    return out


def _write_source(path: Path, ids) -> None:
    """Minimal source TSV with the real header."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        f.writelines(f"{i}\tn\ta\tUS\n" for i in ids)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train on 200 S1 x 10 candidates (2,000 labelled pairs) and refit; return (root, model_dir, bundle)."""
    assert len(COLS) == 43
    rng = np.random.default_rng(42)
    root = tmp_path_factory.mktemp("smoke_root")
    for d in ("dataset/train", "dataset/test", "artifacts"):
        (root / d).mkdir(parents=True)
    s1 = [f"S1-{1000 + i}" for i in range(200)]
    tr = pd.DataFrame([(s, f"S{2 + j % 2}-{50_000 + i * 10 + j}") for i, s in enumerate(s1) for j in range(10)],
                      columns=["s1_id", "cand_id"])
    tr["label"] = ((tr.index % 10) < 3).astype(np.int8)  # 3 true matches per S1
    gt = tr[tr["label"] == 1].groupby("s1_id")["cand_id"].apply(",".join).reindex(s1, fill_value="")
    gt_path = root / "dataset" / "train" / "train_ground_truth.tsv"
    with open(gt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.writelines(f"{s}\t{m}\n" for s, m in gt.items())
    tr = pd.concat([tr, pd.DataFrame(_feats(rng, tr["label"].to_numpy()))], axis=1)
    tr["source"] = tr["cand_id"].str[:2]
    model_dir = root / "model"
    df = T.attach_labels(tr, str(gt_path))
    T.train(df, set(s1[::5]), str(gt_path), str(model_dir), params={"n_estimators": 50})
    T.refit_full(df, str(model_dir / "bundle.json"), str(model_dir))
    return root, model_dir, json.loads((model_dir / "bundle.json").read_text())


def test_bundle_contract(trained):
    """bundle.json carries the ordered 43 features and a threshold in [0, 1]; both model files exist."""
    _, model_dir, bundle = trained
    assert bundle["features"] == COLS
    assert 0.0 <= bundle["threshold"] <= 1.0
    assert (model_dir / "lgbm.txt").exists() and (model_dir / "lgbm_full.txt").exists()


@pytest.mark.parametrize("model", ["lgbm_full.txt", "lgbm.txt"])
def test_score_test_end_to_end(trained, model):
    """score_test.py on a shuffled-column test parquet: exit 0, both checkers pass, matches == decide()."""
    if not VALIDATOR.exists():
        pytest.skip("organizer validator not found; set VALIDATOR=<path>")
    root, model_dir, bundle = trained
    rng = np.random.default_rng(7)
    ts1 = [f"S1-{9000 + i}" for i in range(150)]
    pool = [f"S{2 + k % 2}-{700_000 + k}" for k in range(600)]  # shared targets -> ownership contest
    te = pd.DataFrame([(s, pool[k]) for s in ts1 for k in sorted(rng.choice(600, 8, replace=False))],
                      columns=["s1_id", "cand_id"])
    _write_source(root / "dataset" / "test" / "test_source1.tsv", ts1)
    _write_source(root / "dataset" / "test" / "test_source2.tsv", [p for p in pool if p.startswith("S2")])
    _write_source(root / "dataset" / "test" / "test_source3.tsv", [p for p in pool if p.startswith("S3")])
    art = root / "artifacts"
    pq.write_table(pa.table({"entity_id": ts1}), art / "test_s1_norm.parquet")
    pq.write_table(pa.Table.from_pandas(te.assign(source=te["cand_id"].str[:2]), preserve_index=False),
                   art / "cand_test.parquet")
    tf = pd.concat([te, pd.DataFrame(_feats(rng, (rng.random(len(te)) < 0.3).astype(float)))], axis=1)
    shuffled = ["s1_id", "cand_id"] + list(rng.permutation(COLS))  # must be selected by name
    pq.write_table(pa.Table.from_pandas(tf[shuffled], preserve_index=False), art / "cand_test_features.parquet")
    out = root / f"output_{model}"

    r = subprocess.run([sys.executable, str(SRC / "score_test.py"), "--root", str(root),
                        "--model", str(model_dir / model), "--bundle", str(model_dir / "bundle.json"),
                        "--scored", str(root / f"scored_{model}.parquet"), "--out", str(out),
                        "--validator", str(VALIDATOR), "--batch-size", "300"],
                       capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "checks.py: PASS" in r.stdout and "organizer validator: PASS" in r.stdout
    assert "WARNING" not in r.stdout, r.stdout

    sc = pq.read_table(root / f"scored_{model}.parquet").to_pandas()
    assert sc[["s1_id", "cand_id"]].equals(te), "scores must keep candidate row order"
    exp = decide(sc.rename(columns={"p": "score"}), bundle["threshold"])
    got = pd.read_csv(out / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    pairs = {(s, c) for s, lst in zip(got.iloc[:, 0], got.iloc[:, 1]) if lst for c in lst.split(",")}
    assert pairs == set(zip(exp["s1_id"], exp["cand_id"]))
    assert len(pairs) > 0 and len(got) == len(ts1)
