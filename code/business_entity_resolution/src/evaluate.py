"""Exact macro F0.5 scoring and candidate-set diagnostics for entity resolution."""
import argparse
from typing import Dict, Iterable, Set

BETA2 = 0.25  # beta**2 for beta = 0.5


def read_id_lists(path: str) -> Dict[str, Set[str]]:
    """Read a 2-column TSV (S1 id, comma-separated id list) into {s1_id: set(ids)}."""
    out: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            k, _, v = line.partition("\t")
            out[k] = {x for x in v.split(",") if x}
    return out


def f05(pred: Set[str], gold: Set[str]) -> float:
    """F0.5 for one S1 entity. Empty prediction on empty gold is 1.0; any mismatch of emptiness is 0.0."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred: Dict[str, Set[str]], gold: Dict[str, Set[str]]) -> float:
    """Mean per-entity F0.5 over every S1 in gold; S1s missing from pred count as empty predictions."""
    if not gold:
        return 0.0
    return sum(f05(pred.get(k, set()), g) for k, g in gold.items()) / len(gold)


def macro_by_group(pred, gold, group: Dict[str, str]) -> Dict[str, float]:
    """Macro F0.5 split by a group label per S1 id (e.g. country)."""
    buckets: Dict[str, list] = {}
    for k, g in gold.items():
        buckets.setdefault(group.get(k, "?"), []).append(f05(pred.get(k, set()), g))
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def candidate_diagnostics(cands: Dict[str, Set[str]], gold: Dict[str, Set[str]]) -> Dict[str, float]:
    """Blocking quality: pair recall, macro recall, oracle F0.5 (keep only true pairs), and candidate-count stats."""
    n_gold = sum(len(g) for g in gold.values())
    hit = sum(len(cands.get(k, set()) & g) for k, g in gold.items())
    recalls = [len(cands.get(k, set()) & g) / len(g) if g else 1.0 for k, g in gold.items()]
    oracle = {k: cands.get(k, set()) & g for k, g in gold.items()}
    sizes = sorted(len(cands.get(k, ())) for k in gold)
    n = max(len(sizes), 1)
    return {
        "pair_recall": hit / n_gold if n_gold else 1.0,
        "macro_recall": sum(recalls) / n,
        "oracle_f05": macro_f05(oracle, gold),
        "total_cands": float(sum(sizes)),
        "mean_cands": sum(sizes) / n,
        "p95_cands": float(sizes[int(0.95 * (n - 1))]) if sizes else 0.0,
    }


def _toy_asserts() -> None:
    """Sanity checks from the task statement's worked example and the singleton rules."""
    assert abs(f05({"a", "b", "c"}, {"a", "c"}) - 0.7142857) < 1e-6
    assert f05(set(), set()) == 1.0
    assert f05({"a"}, set()) == 0.0
    assert f05(set(), {"a"}) == 0.0
    assert f05({"a"}, {"b"}) == 0.0
    assert f05({"a", "b"}, {"a", "b"}) == 1.0
    assert abs(macro_f05({"x": {"a"}}, {"x": {"a"}, "y": set()}) - 1.0) < 1e-9
    assert abs(macro_f05({}, {"x": {"a"}, "y": set()}) - 0.5) < 1e-9
    d = candidate_diagnostics({"x": {"a", "z"}}, {"x": {"a", "b"}})
    assert d["pair_recall"] == 0.5 and d["oracle_f05"] > 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score a matching TSV against ground truth (optionally restricted to a split).")
    ap.add_argument("--pred", help="matching_results.tsv (or candidate_pairs.tsv for diagnostics)")
    ap.add_argument("--gold", help="train_ground_truth.tsv")
    ap.add_argument("--ids", help="optional file of S1 ids (one per line, .gz ok) to restrict scoring to")
    ap.add_argument("--diagnostics", action="store_true", help="treat --pred as a candidate file")
    a = ap.parse_args()
    _toy_asserts()
    if not a.pred:
        print("toy asserts passed")
        raise SystemExit
    gold = read_id_lists(a.gold)
    if a.ids:
        import gzip
        op = gzip.open if a.ids.endswith(".gz") else open
        with op(a.ids, "rt") as f:
            keep = {l.strip() for l in f if l.strip()}
        gold = {k: v for k, v in gold.items() if k in keep}
    pred = read_id_lists(a.pred)
    print(candidate_diagnostics(pred, gold) if a.diagnostics else {"macro_f05": macro_f05(pred, gold)})
