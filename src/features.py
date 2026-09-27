"""Stage 3 — pair features for candidate (s1_id, cand_id) rows.

Input : a candidate parquet with at least s1_id and cand_id, plus the normalized parquets from
        normalize.py. If it also carries blocking.py's channel/score/rank columns, the retrieval
        and competition features below are added. `label` and `neg_kind` pass through untouched.
Output: one float32 row per DISTINCT (s1_id, cand_id) pair.

blocking.py emits one row per (pair, channel), so a pair found by three channels appears three
times. build() collapses those first (score = max, rank = min, one flag per channel); anything
consuming a raw blocking parquet without collapsing would double-count pairs.

Column contract: do NOT assume "the numeric columns are the features" — `label` is int8 and would
sail through that test. Use feature_columns(table), which returns exactly the model inputs.

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

Added only when the candidate file carries channel/score/rank (i.e. real blocking output):

  retrieval  one flag per channel (A/P/B/C), channel count, best score, best rank,
             rank as a fraction of the S1's list, candidate-list size                (9)
  competition gap and ratio to the S1's best candidate; the pair's rank among the S1s
             competing for that same target, that target's contender count and score gap (5)

`source` is taken from the S2-/S3- prefix, which is a source marker rather than an identity
feature; no country one-hot and no numeric id is used, per CLAUDE_v2.md.

Missing input (an empty string on either side) always yields 0 similarity plus the matching
missing flag, so the model can tell "different" from "unknown".

Scale: pairs are processed in batches and only the normalized rows actually referenced are
loaded (streamed and filtered), so peak RAM tracks the candidate set, not the 10.3M-row corpus.
String similarity runs through rapidfuzz.process.cpdist(workers=-1), which releases the GIL and
uses every core.

The address TF-IDF vectorizers are fitted on a sample of the FULL normalized corpus (every
split/source parquet present, train and test) and cached to artifacts/addr_tfidf.joblib, so
train and test share a vocabulary. Fitting on the candidate set instead would leave the test-only
France text with no vocabulary at all. The cache records which sources it saw and warns when
more have since appeared; delete the file to refit.

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
def load_records(ids: pa.Array, split: str, sources: tuple) -> pa.Table:
    """The referenced normalized rows, as one Arrow table.

    Arrow-backed rather than a dict of per-row dicts: at ~10M distinct targets a Python dict
    costs a dict object plus boxed str per field per row (tens of GB), while the same data as
    contiguous Arrow string arrays is a few GB. Rows are streamed and filtered so only
    referenced ones are ever materialized.
    """
    parts = []
    for src in sources:
        pf = pq.ParquetFile(norm_path(split, src))
        for rb in pf.iter_batches(batch_size=250_000, columns=REC_COLS):
            sel = rb.filter(pc.is_in(rb.column("entity_id"), value_set=ids))
            if sel.num_rows:
                parts.append(sel)
    if not parts:
        raise ValueError(f"no rows matched in {split}/{sources}")
    return pa.Table.from_batches(parts)


def positions(tbl: pa.Table, ids: pa.Array) -> pa.Array:
    """Row position in `tbl` for each id, computed once for the whole candidate list.

    pc.index_in builds its hash table in C++ and returns an int32 array, so the lookup for 40M
    pairs costs ~160 MB instead of a 40M-entry Python dict.
    """
    pos = pc.index_in(ids, value_set=tbl.column("entity_id"))
    if pos.null_count:
        raise ValueError(f"{pos.null_count:,} candidate ids missing from the normalized sources")
    return pos


def columns_of(tbl: pa.Table, pos: pa.Array) -> dict:
    """Aligned column -> python list for one batch of rows (batch-sized, not corpus-sized)."""
    rows = tbl.take(pos)
    return {c: rows.column(c).to_pylist() for c in REC_COLS}


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


def corpus_sources() -> list:
    """Every normalized (split, source) parquet that currently exists."""
    return [(sp, sr) for sp in ("train", "test") for sr in ("S1", "S2", "S3")
            if norm_path(sp, sr).exists()]


def fit_addr_tfidf(per_source: int = 100_000, cache: bool = True) -> tuple:
    """Fit (or load) the two char-3gram address vectorizers, sampled across the whole corpus.

    Fitted on every normalized parquet present, not on the candidate set: test carries French
    addresses that appear in no training candidate, and a vocabulary fitted on train-only text
    gives those rows an all-zero vector, i.e. a silently dead feature on ~15% of the test set.
    Using the provided test corpus is allowed (it is not external data).

    The cache records the sources it saw, so a vectorizer fitted before the test parquets
    existed is reported rather than reused blindly.
    """
    path = ART / "addr_tfidf.joblib"
    have = corpus_sources()
    if cache and path.exists():
        blob = joblib.load(path)
        if isinstance(blob, dict):
            missing = [s for s in have if s not in blob.get("sources", [])]
            if missing:
                print(f"  WARNING: cached address TF-IDF was fitted without {missing}; "
                      f"delete {path} to refit (France would have no vocabulary)",
                      file=sys.stderr, flush=True)
            return blob["vecs"]
        print(f"  WARNING: {path} predates source tracking; delete it to refit",
              file=sys.stderr, flush=True)
        return blob

    rng = np.random.default_rng(SEED)
    texts = {"addr_norm": [], "addr_phon": []}
    for sp, sr in have:
        for col in texts:
            arr = pq.read_table(norm_path(sp, sr), columns=[col]).column(col)
            n = len(arr)
            take = min(per_source, n)
            idx = np.sort(rng.choice(n, size=take, replace=False)) if take < n else np.arange(n)
            texts[col].extend(v for v in arr.take(pa.array(idx)).to_pylist() if v)
            del arr
    print(f"  fitted address TF-IDF on {len(have)} sources "
          f"({len(texts['addr_norm']):,} sampled addresses)", file=sys.stderr, flush=True)
    vecs = (TfidfVectorizer(**TFIDF_PARAMS).fit(texts["addr_norm"]),
            TfidfVectorizer(**TFIDF_PARAMS).fit(texts["addr_phon"]))
    if cache:
        ART.mkdir(exist_ok=True)
        joblib.dump({"vecs": vecs, "sources": have}, path)
    return vecs


def cosine_rows(vec, a: list, b: list) -> np.ndarray:
    """Row-wise cosine between paired texts (TfidfVectorizer already L2-normalizes)."""
    X, Y = vec.transform(a), vec.transform(b)
    return np.asarray(X.multiply(Y).sum(axis=1)).ravel().astype(np.float32)


# ---------------------------------------------------------------- feature block
def features_for(a: dict, b: dict, vecs: tuple) -> dict:
    """The 31 record-comparison features for one batch.

    `a` and `b` are aligned column -> list mappings for the S1 side and the candidate side
    (see columns_of); row i of `a` is compared with row i of `b`.
    """
    f = {}
    n = len(a["entity_id"])
    for field in NAME_FIELDS:
        for tag, scorer in NAME_SCORERS:
            f[f"{field}_{tag}"] = sim(a[field], b[field], scorer)

    # acronym: S1's acronym against the candidate's acronym and its whitespace-free core,
    # which catches "IAG" stored as a single-token business name, and the reverse direction.
    acr_a, acr_b = a["name_acronym"], b["name_acronym"]
    core_a, core_b = a["name_core"], b["name_core"]
    ca = [c.replace(" ", "") for c in core_a]
    cb = [c.replace(" ", "") for c in core_b]
    f["acronym_match"] = np.fromiter(
        (float((bool(x) and (x == y or x == v)) or (bool(y) and y == u))
         for x, y, u, v in zip(acr_a, acr_b, ca, cb)), np.float32, n)

    # best alternate-name (dba / fka) match, in both directions
    f["name_alt_token_set"] = np.maximum(
        sim(core_a, b["name_alt_core"], fuzz.token_set_ratio),
        sim(a["name_alt_core"], core_b, fuzz.token_set_ratio))
    f["name_missing"] = np.fromiter(((not x) or (not y) for x, y in zip(core_a, core_b)),
                                    np.float32, n)

    an, bn = a["addr_norm"], b["addr_norm"]
    ap, bp = a["addr_phon"], b["addr_phon"]
    f["addr_norm_token_set"] = sim(an, bn, fuzz.token_set_ratio)
    f["addr_phon_token_set"] = sim(ap, bp, fuzz.token_set_ratio)
    f["addr_norm_tfidf"] = cosine_rows(vecs[0], an, bn)
    f["addr_phon_tfidf"] = cosine_rows(vecs[1], ap, bp)

    f["num_agree"], f["num_conflict"], f["num_missing"] = token_stats(a["addr_nums"],
                                                                      b["addr_nums"])
    f["postal_agree"], f["postal_conflict"], f["postal_missing"] = token_stats(a["addr_postal"],
                                                                               b["addr_postal"])

    f["addr_blank"] = np.fromiter((float(x or y) for x, y in zip(a["addr_blank"],
                                                                 b["addr_blank"])), np.float32, n)
    f["was_non_latin"] = np.fromiter((float(x or y) for x, y in zip(b["name_non_latin"],
                                                                    b["addr_non_latin"])),
                                     np.float32, n)
    f["is_s3"] = np.fromiter((float(c.startswith("S3-")) for c in b["entity_id"]), np.float32, n)
    return f


BASE_FEATURES = ([f"{fl}_{tag}" for fl in NAME_FIELDS for tag, _ in NAME_SCORERS]
                 + ["acronym_match", "name_alt_token_set", "name_missing",
                    "addr_norm_token_set", "addr_phon_token_set",
                    "addr_norm_tfidf", "addr_phon_tfidf",
                    "num_agree", "num_conflict", "num_missing",
                    "postal_agree", "postal_conflict", "postal_missing",
                    "addr_blank", "was_non_latin", "is_s3"])
CHANNELS = ("A", "P", "B", "C")
# Ranking/competition signals derived for any candidate file that carries retrieval evidence.
CORE_RETRIEVAL = ["retr_score", "retr_rank", "rank_frac", "cand_count",
                  "score_gap_best", "score_ratio_best",
                  "rev_rank", "rev_count", "rev_score_gap"]
# blocking.py v3 (frozen) emits tok_score / name_sim / final_rank and has no A/P/B/C channels.
V3_EXTRA = ["retr_tok_score", "retr_name_sim", "from_blank_index"]
# legacy channel-union schema (blocking v1/v2), kept so older parquets still work
CHANNEL_EXTRA = [f"ch_{c}" for c in CHANNELS] + ["n_channels"]
RETRIEVAL_FEATURES = CORE_RETRIEVAL + V3_EXTRA + CHANNEL_EXTRA
FEATURE_NAMES = BASE_FEATURES          # retained for callers that predate the retrieval block
PASSTHROUGH = ("s1_id", "cand_id", "label", "neg_kind", "source")


def rerank_alpha() -> float:
    """The frozen rerank weight from blocking.py, so these features track A's config."""
    try:
        from blocking import FROZEN
        return float(FROZEN["alpha"])
    except Exception:
        return 6.0


def feature_columns(tbl: pa.Table) -> list:
    """The model-input columns of a feature table, in order.

    The contract for train/predict. Selecting "numeric columns" instead would silently keep
    `label` (int8) and train the model on its own answer.
    """
    return [c for c in tbl.column_names if c not in PASSTHROUGH]


def collapse(cand: pa.Table) -> "pd.DataFrame":
    """One row per distinct (s1_id, cand_id), with retrieval/competition columns when available.

    Three input schemas are recognised:

      v3 (blocking.py, frozen)  s1_id, cand_id, source, blank_addr_index,
                                tok_score, name_sim, final_rank
      channel union (v1/v2)     s1_id, cand_id, channel, score, rank
      plain                     s1_id, cand_id [, label, neg_kind]  -> dedupe only

    v3 ranks by `tok_score + alpha * name_sim/100` but stores only the two parts, so the
    composite is reconstructed with blocking's own FROZEN alpha. v3 has no A/P/B/C channels;
    `blank_addr_index` (the name-only index used for blank-address targets) is the one
    provenance flag it does carry.

    Whatever the schema, the derived signals are the same: how this candidate compares with
    the S1's other candidates (rank_frac, score_gap_best, score_ratio_best), and how it fares
    against the other S1s competing for the same target (rev_rank, rev_count, rev_score_gap).
    Since each S2/S3 id belongs to at most one S1, that competition view is what lets the model
    reason about the contest rather than scoring each pair alone.
    """
    import pandas as pd
    df = cand.to_pandas()
    cols = set(df.columns)
    keys = ["s1_id", "cand_id"]
    rank_col = next((c for c in ("final_rank", "tok_rank", "rank") if c in cols), None)
    v3 = "tok_score" in cols
    legacy = "channel" in cols and "score" in cols
    if not (v3 or legacy) or rank_col is None:
        return df.drop_duplicates(keys).reset_index(drop=True)

    if v3:
        alpha = rerank_alpha()
        df = df.assign(_score=df["tok_score"].to_numpy(np.float32)
                       + alpha * df["name_sim"].to_numpy(np.float32) / 100.0)
        spec = dict(retr_score=("_score", "max"), retr_rank=(rank_col, "min"),
                    retr_tok_score=("tok_score", "max"), retr_name_sim=("name_sim", "max"))
        if "blank_addr_index" in cols:
            spec["from_blank_index"] = ("blank_addr_index", "max")
    else:
        df = df.assign(_score=df["score"].to_numpy(np.float32))
        spec = dict(retr_score=("_score", "max"), retr_rank=(rank_col, "min"))

    agg = df.groupby(keys, sort=False).agg(**spec).reset_index()

    if legacy:
        ch = df[[*keys, "channel"]].copy()
        ch["channel"] = ch["channel"].astype(str)
        ch = ch.drop_duplicates()
        bit = {c: 1 << i for i, c in enumerate(CHANNELS)}
        ch["_b"] = ch["channel"].map(bit).fillna(0).astype(np.int16)
        # each channel appears at most once per pair after the dedupe, so sum == bitwise or
        bits = ch.groupby(keys, sort=False)["_b"].sum().reset_index()
        agg = agg.merge(bits, on=keys, how="left")
        b = agg["_b"].fillna(0).astype(np.int16).to_numpy()   # pandas Series has no >>; numpy does
        for i, c in enumerate(CHANNELS):
            agg[f"ch_{c}"] = (np.right_shift(b, i) & 1).astype(np.float32)
        agg["n_channels"] = agg[[f"ch_{c}" for c in CHANNELS]].sum(axis=1).astype(np.float32)
        agg = agg.drop(columns="_b")

    # within the S1's own candidate list
    best = agg.groupby("s1_id")["retr_score"].transform("max")
    agg["cand_count"] = agg.groupby("s1_id")["s1_id"].transform("size").astype(np.float32)
    agg["score_gap_best"] = (best - agg["retr_score"]).astype(np.float32)
    agg["score_ratio_best"] = np.where(best > 0, agg["retr_score"] / best, 0.0).astype(np.float32)
    agg["rank_frac"] = (agg["retr_rank"] / agg["cand_count"].clip(lower=1)).astype(np.float32)

    # competition: the same target is reachable from several S1s but belongs to at most one
    gc = agg.groupby("cand_id")["retr_score"]
    agg["rev_rank"] = (gc.rank(ascending=False, method="min") - 1).astype(np.float32)
    agg["rev_count"] = agg.groupby("cand_id")["cand_id"].transform("size").astype(np.float32)
    agg["rev_score_gap"] = (gc.transform("max") - agg["retr_score"]).astype(np.float32)

    for c in ("retr_score", "retr_rank", "retr_tok_score", "retr_name_sim", "from_blank_index"):
        if c in agg.columns:
            agg[c] = agg[c].astype(np.float32)
    if "retr_name_sim" in agg.columns:
        agg["retr_name_sim"] /= 100.0            # rapidfuzz 0-100 -> 0-1, like every other sim

    extra = [c for c in cand.column_names if c in ("label", "neg_kind")]
    if extra:
        agg = agg.merge(df[[*keys, *extra]].drop_duplicates(keys), on=keys, how="left")
    return agg


def attach_labels(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add an int8 `label` from train_ground_truth: 1 if the pair is a true match."""
    from blocking import gt_pairs
    gt = gt_pairs()
    truth = set(zip(gt["s1_id"].to_numpy(), gt["cand_id"].to_numpy()))
    df["label"] = np.fromiter((1 if k in truth else 0
                               for k in zip(df["s1_id"].to_numpy(), df["cand_id"].to_numpy())),
                              np.int8, len(df))
    return df


def constant_columns(tbl: pa.Table) -> list:
    """Feature columns with a single distinct value in THIS table.

    Reported, never dropped. Dropping per file would give the fitting and validation parquets
    different column sets whenever a feature is constant in one and not the other, and the
    model would then be trained and scored on different schemas.
    """
    out = []
    for c in feature_columns(tbl):
        v = np.asarray(tbl[c], dtype=np.float32)
        if v.size and float(v.min()) == float(v.max()):
            out.append(c)
    return out


def build(cand_path, split: str = "train", batch: int = 500_000,
          labels: bool = False) -> tuple:
    """Features for every distinct pair in the candidate parquet, batched."""
    import pandas as pd
    raw = pq.read_table(cand_path)
    with timed(f"collapse candidates ({raw.num_rows:,} rows)"):
        df = collapse(raw)
    if labels and "label" not in df.columns:
        with timed("attach ground-truth labels"):
            df = attach_labels(df)
    retr = [c for c in RETRIEVAL_FEATURES if c in df.columns]
    n = len(df)
    print(f"  {n:,} distinct pairs; retrieval features: "
          f"{'yes' if retr else 'no (candidate file has no score/rank)'}",
          file=sys.stderr, flush=True)

    s1_ids = pa.array(df["s1_id"].to_numpy(), type=pa.string())
    cd_ids = pa.array(df["cand_id"].to_numpy(), type=pa.string())

    with timed("load referenced records"):
        s1_tbl = load_records(pa.array(pd.unique(df["s1_id"]), type=pa.string()), split, ("S1",))
        cd_tbl = load_records(pa.array(pd.unique(df["cand_id"]), type=pa.string()),
                              split, ("S2", "S3"))
        s1_pos, cd_pos = positions(s1_tbl, s1_ids), positions(cd_tbl, cd_ids)
    print(f"  {s1_tbl.num_rows:,} S1 + {cd_tbl.num_rows:,} candidate rows held "
          f"(peak RAM {peak_ram_gb():.1f} GB)", file=sys.stderr, flush=True)

    with timed("fit/load address TF-IDF"):
        vecs = fit_addr_tfidf()

    chunks, t0 = [], time.perf_counter()
    for start in range(0, n, batch):
        end = min(start + batch, n)
        sl = slice(start, end)
        f = features_for(columns_of(s1_tbl, s1_pos[sl]), columns_of(cd_tbl, cd_pos[sl]), vecs)
        chunks.append(pa.table({k: pa.array(v) for k, v in f.items()}))
        print(f"  features {end:,}/{n:,} ({end / (time.perf_counter() - t0):,.0f} pairs/s)",
              file=sys.stderr, flush=True)
    feats = pa.concat_tables(chunks)

    out = {"s1_id": s1_ids, "cand_id": cd_ids}
    for col in ("label", "neg_kind"):
        if col in df.columns:
            out[col] = pa.array(df[col].to_numpy())
    for name in BASE_FEATURES:
        out[name] = feats[name]
    for name in retr:
        out[name] = pa.array(df[name].to_numpy(np.float32))
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
    names = feature_columns(tbl)
    lab = np.asarray(tbl["label"])
    pos, neg = lab == 1, lab == 0
    print(f"feature means by label ({int(pos.sum()):,} positive / {int(neg.sum()):,} negative)")
    print(f"{'feature':<26}{'label=1':>10}{'label=0':>10}{'gap':>10}")
    for name in names:
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
    ap.add_argument("--labels", action="store_true",
                    help="attach label from train_ground_truth (train split only)")
    ap.add_argument("--head", type=int, default=0, help="smoke-test on the first N pairs only")
    args = ap.parse_args()

    src_path = args.candidates
    if args.head:
        sub = pq.read_table(src_path).slice(0, args.head)
        src_path = str(ART / "_head_sample.parquet")
        pq.write_table(sub, src_path)
        print(f"smoke test on the first {args.head:,} rows", file=sys.stderr, flush=True)

    tbl, elapsed = build(src_path, args.split, args.batch, args.labels)
    out = args.out or str(args.candidates).replace(".parquet", "_features.parquet")
    pq.write_table(tbl, out, compression="zstd")
    names = feature_columns(tbl)
    print(f"wrote {out}: {tbl.num_rows:,} rows x {len(names)} features")
    const = constant_columns(tbl)
    if const:
        print(f"  constant in this file (kept, so fit/val schemas stay identical): "
              f"{', '.join(const)}")
    if args.report:
        report(tbl, elapsed)


if __name__ == "__main__":
    sys.exit(main())
