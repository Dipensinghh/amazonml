"""Synthetic end-to-end check of evaluate/predict/train (run: python tests/test_model_stack.py from repo root)."""
import sys, tempfile
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from ber.evaluate import _toy_asserts, macro_f05
from ber.predict import assign_owners, macro_f05_scored, sweep_threshold, to_lists
from ber.train import feature_columns, train

_toy_asserts()

# ownership: c1 contested; higher p wins, exact tie -> smallest s1_id
d = pd.DataFrame({"s1_id": ["S1-b", "S1-a", "S1-a", "S1-b"], "cand_id": ["c1", "c1", "c2", "c3"], "p": [0.9, 0.9, 0.8, 0.4]})
a = assign_owners(d, 0.5)
assert dict(zip(a.cand_id, a.s1_id)) == {"c1": "S1-a", "c2": "S1-a"}

# vectorized scorer == exact scorer on random data
rng = np.random.RandomState(0)
s1 = [f"S1-{i}" for i in range(300)]
gold = {s: {f"S2-{rng.randint(0, 900)}" for _ in range(rng.randint(0, 4))} for s in s1}
rows = []
for s in s1:
    for c in gold[s] | {f"S3-{rng.randint(0, 900)}" for _ in range(3)}:
        rows.append((s, c, int(c in gold[s]), float(np.clip(rng.rand() * 0.5 + 0.5 * (c in gold[s]), 0, 1))))
sc = pd.DataFrame(rows, columns=["s1_id", "cand_id", "y", "p"])
sizes = pd.Series({s: len(g) for s, g in gold.items()})
for t in (0.3, 0.6):
    v = macro_f05_scored(sc, sizes, t)
    e = macro_f05(to_lists(assign_owners(sc, t), s1), gold)
    assert abs(v - e) < 1e-9, (v, e)
print("vectorized == exact OK; sweep:", {k: v for k, v in sweep_threshold(sc, sizes).items() if k != "curve"})

# train end-to-end on synthetic features
n = len(sc)
sc["f_a"] = sc["y"] * 0.6 + rng.rand(n) * 0.6
sc["f_b"] = rng.rand(n)
feats = sc[["s1_id", "cand_id", "f_a", "f_b"]]
gt = tempfile.mktemp(suffix=".tsv")
with open(gt, "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s in s1: f.write(f"{s}\t{','.join(sorted(gold[s]))}\n")
from ber.train import attach_labels
df = attach_labels(feats, gt)
assert df["y"].sum() == sc["y"].sum()
try:
    feature_columns(df.assign(country="US")); raise SystemExit("country not rejected")
except ValueError: pass
res = train(df, set(s1[:60]), gt, out_dir=tempfile.mkdtemp())
assert res["macro_f05"] > 0.6, res["macro_f05"]
print("ALL OK")
