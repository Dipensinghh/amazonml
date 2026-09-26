"""Stage 3 — pair features for candidate (s1_id, cand_id) rows.

Input : a candidate parquet with at least s1_id and cand_id (an optional `label` column is
        carried through untouched), plus the normalized parquets from normalize.py.
Output: one float32 row per candidate pair, keyed by (s1_id, cand_id).

31 features, all float32, all symmetric-free (they never look at the raw ids, the country, or
anything that would not exist at test time):

  name       ratio / token_set / token_sort / partial / Jaro-Winkler
             on name_norm, name_core and name_phon                              (15)
  name extra acronym match, best alternate-name (dba/fka) token_set, name-missing flag (3)
  address    token_set on addr_norm and addr_phon,
             char-3gram TF-IDF cosine on addr_norm and addr_phon                 (4)
  tokens     numeric-token agree / conflict / missing,
             postal-token agree / conflict / missing                             (6)
  context    either-side blank address, candidate was non-Latin, source is S3    (3)

`source` is taken from the S2-/S3- prefix, which is a source marker rather than an identity
feature; no country one-hot and no numeric id is used, per CLAUDE_v2.md.

Missing input (an empty string on either side) always yields 0 similarity plus the matching
missing flag, so the model can tell "different" from "unknown".

Scale: pairs are processed in batches and only the normalized rows actually referenced are
loaded (streamed and filtered), so peak RAM tracks the candidate set, not the 10.3M-row corpus.
String similarity runs through rapidfuzz.process.cpdist(workers=-1), which releases the GIL and
uses every core.

The address TF-IDF vectorizers are fitted once and cached to artifacts/addr_tfidf.joblib so that
train and test features share a vocabulary; delete that file to refit.

Usage:
  python -u src/features.py --candidates artifacts/dev_candidates.parquet --report
"""
import argparse
import sys
import time

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer

from common import ART, SEED, norm_path, peak_ram_gb, timed

REC_COLS = ["entity_id", "name_norm", "name_core", "name_phon", "name_acronym",
            "name_alt_core", "name_alt_phon", "name_non_latin",
            "addr_norm", "addr_phon", "addr_nums", "addr_postal", "addr_blank",
            "addr_non_latin"]

# (feature suffix, scorer) applied to each of name_norm / name_core / name_phon
NAME_SCORERS = (("ratio", fuzz.ratio),
                ("token_set", fuzz.token_set_ratio),
                ("token_sort", fuzz.token_sort_ratio),
                ("partial", fuzz.partial_ratio),
                ("jw", JaroWinkler.normalized_similarity))
NAME_FIELDS = ("name_norm", "name_core", "name_phon")
TFIDF_PARAMS = dict(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True,
                    min_df=2, dtype=np.float32)


# ---------------------------------------------------------------- record loading
def load_records(ids: set, split: str, sources: tuple) -> dict:
    """entity_id -> row dict, for the given ids only.

    Streams each normalized parquet in batches and keeps just the referenced rows, so a
    candidate set that touches 1% of the corpus costs 1% of the memory.
    """
    out = {}
    for src in sources:
        pf = pq.ParquetFile(norm_path(split, src))
        for rb in pf.iter_batches(batch_size=250_000, columns=REC_COLS):
            eids = rb.column("entity_id").to_pylist()
            hit = [i for i, e in enumerate(eids) if e in ids]
            if not hit:
                continue
            cols = {c: rb.column(c).take(pa.array(hit)).to_pylist() for c in REC_COLS}
            for j, i in enumerate(hit):
                out[eids[i]] = {c: cols[c][j] for c in REC_COLS}
    return out


def gather(recs: dict, ids: list, col: str, default=""):
    """Column `col` for each id, with `default` when the id is absent (should not happen)."""
    return [recs[i][col] if i in recs else default for i in ids]


# ---------------------------------------------------------------- similarity helpers
def sim(a: list, b: list, scorer) -> np.ndarray:
    """Element-wise similarity in [0,1]; 0 wherever either side is empty."""
    s = np.asarray(cpdist(a, b, scorer=scorer, workers=-1), dtype=np.float32)
    if s.size and s.max() > 1.0:          # fuzz.* return 0-100, JaroWinkler returns 0-1
        s /= 100.0
    empty = np.fromiter(((not x) or (not y) for x, y in zip(a, b)), bool, len(a))
    s[empty] = 0.0
    return s


def token_stats(a: list, b: list) -> tuple:
    """(agreement, conflict, missing) over two space-joined token strings.

    agreement = Jaccard overlap; conflict = both sides present but disjoint;
    missing   = at least one side empty (agreement and conflict are then 0).
    """
    agree = np.zeros(len(a), dtype=np.float32)
    conflict = np.zeros(len(a), dtype=np.float32)
    missing = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if not sx or not sy:
            missing[i] = 1.0
            continue
        inter = len(sx & sy)
        agree[i] = inter / len(sx | sy)
        conflict[i] = 0.0 if inter else 1.0
    return agree, conflict, missing


def fit_addr_tfidf(texts_norm: list, texts_phon: list, cache=True) -> tuple:
    """Fit (or load) the two char-3gram address vectorizers shared by train and test."""
    path = ART / "addr_tfidf.joblib"
    if cache and path.exists():
        return joblib.load(path)
    vecs = (TfidfVectorizer(**TFIDF_PARAMS).fit([t for t in texts_norm if t]),
            TfidfVectorizer(**TFIDF_PARAMS).fit([t for t in texts_phon if t]))
    if cache:
        ART.mkdir(exist_ok=True)
        joblib.dump(vecs, path)
    return vecs


def cosine_rows(vec, a: list, b: list) -> np.ndarray:
    """Row-wise cosine between paired texts (TfidfVectorizer already L2-normalizes)."""
    X, Y = vec.transform(a), vec.transform(b)
    return np.asarray(X.multiply(Y).sum(axis=1)).ravel().astype(np.float32)


# ---------------------------------------------------------------- feature block
def features_for(s1_ids: list, cand_ids: list, recs: dict, vecs: tuple) -> dict:
    """All 31 features for one batch of pairs, as name -> float32 array."""
    f = {}
    for field in NAME_FIELDS:
        a, b = gather(recs, s1_ids, field), gather(recs, cand_ids, field)
        for tag, scorer in NAME_SCORERS:
            f[f"{field}_{tag}"] = sim(a, b, scorer)

    # acronym: S1's acronym against the candidate's acronym and its whitespace-free core,
    # which catches "IAG" stored as a single-token business name.
    acr_a = gather(recs, s1_ids, "name_acronym")
    acr_b = gather(recs, cand_ids, "name_acronym")
    core_b = [c.replace(" ", "") for c in gather(recs, cand_ids, "name_core")]
    core_a = [c.replace(" ", "") for c in gather(recs, s1_ids, "name_core")]
    f["acronym_match"] = np.fromiter(
        (float((bool(a) and (a == b or a == cb)) or (bool(b) and b == ca))
         for a, b, ca, cb in zip(acr_a, acr_b, core_a, core_b)),
        np.float32, len(s1_ids))

    # best alternate-name (dba / fka) match, in both directions
    alt_b = gather(recs, cand_ids, "name_alt_core")
    alt_a = gather(recs, s1_ids, "name_alt_core")
    core_s1 = gather(recs, s1_ids, "name_core")
    core_cd = gather(recs, cand_ids, "name_core")
    f["name_alt_token_set"] = np.maximum(sim(core_s1, alt_b, fuzz.token_set_ratio),
                                         sim(alt_a, core_cd, fuzz.token_set_ratio))
    f["name_missing"] = np.fromiter(((not x) or (not y) for x, y in zip(core_s1, core_cd)),
                                    np.float32, len(s1_ids))

    an, bn = gather(recs, s1_ids, "addr_norm"), gather(recs, cand_ids, "addr_norm")
    ap, bp = gather(recs, s1_ids, "addr_phon"), gather(recs, cand_ids, "addr_phon")
    f["addr_norm_token_set"] = sim(an, bn, fuzz.token_set_ratio)
    f["addr_phon_token_set"] = sim(ap, bp, fuzz.token_set_ratio)
    f["addr_norm_tfidf"] = cosine_rows(vecs[0], an, bn)
    f["addr_phon_tfidf"] = cosine_rows(vecs[1], ap, bp)

    f["num_agree"], f["num_conflict"], f["num_missing"] = token_stats(
        gather(recs, s1_ids, "addr_nums"), gather(recs, cand_ids, "addr_nums"))
    f["postal_agree"], f["postal_conflict"], f["postal_missing"] = token_stats(
        gather(recs, s1_ids, "addr_postal"), gather(recs, cand_ids, "addr_postal"))

    f["addr_blank"] = np.fromiter(
        (float(x or y) for x, y in zip(gather(recs, s1_ids, "addr_blank", False),
                                       gather(recs, cand_ids, "addr_blank", False))),
        np.float32, len(s1_ids))
    f["was_non_latin"] = np.fromiter(
        (float(x or y) for x, y in zip(gather(recs, cand_ids, "name_non_latin", False),
                                       gather(recs, cand_ids, "addr_non_latin", False))),
        np.float32, len(s1_ids))
    f["is_s3"] = np.fromiter((float(c.startswith("S3-")) for c in cand_ids),
                             np.float32, len(s1_ids))
    return f


FEATURE_NAMES = ([f"{fl}_{tag}" for fl in NAME_FIELDS for tag, _ in NAME_SCORERS]
                 + ["acronym_match", "name_alt_token_set", "name_missing",
                    "addr_norm_token_set", "addr_phon_token_set",
                    "addr_norm_tfidf", "addr_phon_tfidf",
                    "num_agree", "num_conflict", "num_missing",
                    "postal_agree", "postal_conflict", "postal_missing",
                    "addr_blank", "was_non_latin", "is_s3"])


def build(cand_path, split: str = "train", batch: int = 500_000) -> pa.Table:
    """Features for every pair in the candidate parquet, batched."""
    cand = pq.read_table(cand_path)
    n = cand.num_rows
    s1_all, cd_all = cand["s1_id"].to_pylist(), cand["cand_id"].to_pylist()

    with timed(f"load referenced records ({n:,} pairs)"):
        recs = load_records(set(s1_all), split, ("S1",))
        recs.update(load_records(set(cd_all), split, ("S2", "S3")))
    print(f"  {len(recs):,} normalized rows held", file=sys.stderr, flush=True)

    with timed("fit/load address TF-IDF"):
        rng = np.random.default_rng(SEED)
        keys = list(recs)
        pick = [keys[i] for i in rng.permutation(len(keys))[:200_000]]
        vecs = fit_addr_tfidf([recs[k]["addr_norm"] for k in pick],
                              [recs[k]["addr_phon"] for k in pick])

    chunks, t0 = [], time.perf_counter()
    for start in range(0, n, batch):
        end = min(start + batch, n)
        f = features_for(s1_all[start:end], cd_all[start:end], recs, vecs)
        chunks.append(pa.table({k: pa.array(v) for k, v in f.items()}))
        print(f"  features {end:,}/{n:,} ({(end)/(time.perf_counter()-t0):,.0f} pairs/s)",
              file=sys.stderr, flush=True)
    feats = pa.concat_tables(chunks)

    out = {"s1_id": cand["s1_id"], "cand_id": cand["cand_id"]}
    for passthru in ("label", "neg_kind"):     # carried for the model/eval stages
        if passthru in cand.column_names:
            out[passthru] = cand[passthru]
    for name in FEATURE_NAMES:
        out[name] = feats[name]
    return pa.table(out), time.perf_counter() - t0


def report(tbl: pa.Table, elapsed: float) -> None:
    """Feature means split by label, throughput, and the extrapolation to 40M pairs."""
    n = tbl.num_rows
    rate = n / elapsed if elapsed else float("nan")
    print(f"\n{n:,} pairs in {elapsed:,.1f}s  ->  {rate:,.0f} pairs/s"
          f"\n40,000,000 pairs would take ~{40_000_000 / rate / 60:,.1f} min"
          f"   (peak RAM {peak_ram_gb():.1f} GB)\n")
    if "label" not in tbl.column_names:
        return
    lab = np.asarray(tbl["label"])
    pos, neg = lab == 1, lab == 0
    print(f"feature means by label ({int(pos.sum()):,} positive / {int(neg.sum()):,} negative)")
    print(f"{'feature':<26}{'label=1':>10}{'label=0':>10}{'gap':>10}")
    for name in FEATURE_NAMES:
        v = np.asarray(tbl[name], dtype=np.float32)
        mp, mn = float(v[pos].mean()) if pos.any() else 0.0, float(v[neg].mean()) if neg.any() else 0.0
        print(f"{name:<26}{mp:>10.3f}{mn:>10.3f}{mp - mn:>+10.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=str(ART / "dev_candidates.parquet"))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", default=None, help="output parquet (default: <candidates>_feats)")
    ap.add_argument("--batch", type=int, default=500_000)
    ap.add_argument("--report", action="store_true", help="print label means and throughput")
    args = ap.parse_args()

    tbl, elapsed = build(args.candidates, args.split, args.batch)
    out = args.out or str(args.candidates).replace(".parquet", "_features.parquet")
    pq.write_table(tbl, out, compression="zstd")
    print(f"wrote {out}: {tbl.num_rows:,} rows x {len(FEATURE_NAMES)} features")
    if args.report:
        report(tbl, elapsed)


if __name__ == "__main__":
    sys.exit(main())
