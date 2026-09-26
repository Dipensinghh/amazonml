"""Stage 2b (dev) — a small labelled candidate set so feature and model work can start
before the real blocking output exists.

Takes N validation S1s (the frozen 20% split from blocking.make_split, seed 42) and pairs each
with every true S2/S3 target from the ground truth (label 1) plus ~K hard negatives from the
SAME country that share at least one name_core token, name_phon token or addr_postal token
(label 0).

Negatives come from an inverted index keyed by (country, token). The index is restricted to the
tokens the queries actually use, so it stays small regardless of corpus size; targets are streamed
in record batches so peak RAM does not scale with the 10.3M target rows. Tokens appearing in more
than --max-df target rows are skipped as uninformative (they would only yield easy negatives), and
each token keeps at most --cap postings so one generic word cannot dominate the sample.

Negatives are ranked by how many distinct tokens they share with the query, so the hardest
(most overlapping) rows are kept.

A `neg_kind` column marks how each row was produced: "pos" (ground truth), "overlap" (a genuine
hard negative sharing a token) or "pool" (a random same-country filler). Pool rows are easy and
inflate any label-0 vs label-1 comparison, so report on overlap negatives when the number needs
to mean something.

Output: artifacts/dev_candidates.parquet  (s1_id, cand_id, label, neg_kind)

Usage: python -u src/dev_candidates.py [--n-s1 5000] [--negatives 20]
"""
import argparse
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from blocking import gt_pairs, make_split, sample_ids  # reuse A's frozen split, do not redefine
from common import ART, SEED, norm_path, peak_ram_gb, timed

TOK_COLS = ("name_core", "name_phon", "addr_postal")
IDX_COLS = ["entity_id", "country", *TOK_COLS]


def query_rows(n_s1: int) -> pa.Table:
    """The n_s1 sampled validation S1 rows, with the columns used for token overlap."""
    t = pq.read_table(norm_path("train", "S1"), columns=IDX_COLS)
    is_val = make_split(t["entity_id"].to_pandas())
    val = t["entity_id"].filter(pa.array(is_val)).to_numpy(zero_copy_only=False)
    keep = sample_ids(val, n_s1)
    return t.filter(pc.is_in(t["entity_id"], value_set=pa.array(keep)))


def wanted_keys(q: pa.Table) -> set:
    """(country, token) keys the queries can match on — the only keys worth indexing."""
    out = set()
    countries = q["country"].to_pylist()
    for col in TOK_COLS:
        for c, s in zip(countries, q[col].to_pylist()):
            for tok in s.split():
                out.add((c, tok))
    return out


def build_index(wanted: set, cap: int, max_df: int, batch: int, pool_cap: int) -> tuple:
    """Inverted index (country, token) -> sampled target entity_ids, plus a per-country pool.

    One streaming pass over the normalized S2/S3 parquets, keeping only keys in `wanted`.

    Postings use reservoir sampling rather than "first `cap` wins": S2 is scanned before S3, so
    a first-wins cap would fill every common token with S2 rows and silently starve S3 of
    negatives. Reservoir sampling makes each posting list a uniform sample of all rows carrying
    that token, across both sources.

    `max_df` only drops tokens so common they carry no signal; the reservoir already bounds
    memory, so this is deliberately generous. The per-country pool supplies fallback negatives
    for queries whose tokens are all rare or all dropped.
    """
    rng = random.Random(SEED)
    post, df = defaultdict(list), Counter()
    pool, pool_n = defaultdict(list), Counter()
    for src in ("S2", "S3"):
        pf = pq.ParquetFile(norm_path("train", src))
        seen = 0
        for rb in pf.iter_batches(batch_size=batch, columns=IDX_COLS):
            ids = rb.column("entity_id").to_pylist()
            countries = rb.column("country").to_pylist()
            cols = [rb.column(c).to_pylist() for c in TOK_COLS]
            for i, (eid, country) in enumerate(zip(ids, countries)):
                pool_n[country] += 1
                pl = pool[country]
                if len(pl) < pool_cap:
                    pl.append(eid)
                else:
                    j = rng.randrange(pool_n[country])
                    if j < pool_cap:
                        pl[j] = eid
                toks = set()
                for col in cols:
                    toks.update(col[i].split())
                for tok in toks:
                    key = (country, tok)
                    if key not in wanted:
                        continue
                    df[key] += 1
                    lst = post[key]
                    if len(lst) < cap:
                        lst.append(eid)
                    else:
                        j = rng.randrange(df[key])
                        if j < cap:
                            lst[j] = eid
            seen += len(ids)
        print(f"  {src}: indexed {seen:,} rows, {len(post):,} live keys "
              f"(peak RAM {peak_ram_gb():.1f} GB)", file=sys.stderr, flush=True)
    dropped = [k for k, n in df.items() if n > max_df]
    for k in dropped:
        del post[k]
    print(f"  dropped {len(dropped):,} keys with df > {max_df:,}; {len(post):,} keys retained; "
          f"fallback pool {{{', '.join(f'{c}: {len(v):,}' for c, v in pool.items())}}}",
          file=sys.stderr, flush=True)
    return post, pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-s1", type=int, default=5000, help="validation S1s to sample")
    ap.add_argument("--negatives", type=int, default=20, help="hard negatives per S1")
    ap.add_argument("--cap", type=int, default=400, help="max postings kept per token")
    ap.add_argument("--max-df", type=int, default=250_000,
                    help="skip tokens more frequent than this (reservoir already bounds memory)")
    ap.add_argument("--pool-cap", type=int, default=50_000,
                    help="per-country reservoir used for fallback negatives")
    ap.add_argument("--batch", type=int, default=250_000, help="rows per streaming batch")
    args = ap.parse_args()

    with timed("load queries"):
        q = query_rows(args.n_s1)
    print(f"queries: {q.num_rows:,} val S1", file=sys.stderr, flush=True)

    with timed("ground-truth positives"):
        gt = gt_pairs()
        qids = set(q["entity_id"].to_pylist())
        pos = gt[gt["s1_id"].isin(qids)]
        pos_by_s1 = pos.groupby("s1_id")["cand_id"].apply(set).to_dict()

    with timed("build inverted index"):
        wanted = wanted_keys(q)
        print(f"  {len(wanted):,} (country, token) keys wanted", file=sys.stderr, flush=True)
        post, pool = build_index(wanted, args.cap, args.max_df, args.batch, args.pool_cap)

    with timed("select hard negatives"):
        rng = np.random.default_rng(SEED)
        rows_s1, rows_cand, rows_lab, rows_kind = [], [], [], []
        countries = q["country"].to_pylist()
        qid_list = q["entity_id"].to_pylist()
        qcols = {c: q[c].to_pylist() for c in TOK_COLS}
        n_no_neg = 0
        for i, (s1_id, country) in enumerate(zip(qid_list, countries)):
            truth = pos_by_s1.get(s1_id, set())
            shared = Counter()
            for c in TOK_COLS:
                for tok in set(qcols[c][i].split()):
                    for eid in post.get((country, tok), ()):
                        shared[eid] += 1
            for eid in truth:                       # never sample a true match as a negative
                shared.pop(eid, None)
            # hardest first (most shared tokens); random tie-break keeps it unbiased
            fallback = set()
            cands = np.array(list(shared.keys()))
            if len(cands):
                counts = np.array([shared[c] for c in cands])
                order = np.lexsort((rng.random(len(cands)), -counts))
                cands = cands[order][: args.negatives]
            # Fall back to random same-country targets so every S1 carries negatives; without
            # this, S1s with only rare tokens would contribute positives only and skew training.
            if len(cands) < args.negatives:
                n_no_neg += len(cands) == 0
                have = set(cands.tolist()) | truth
                src_pool = pool.get(country, [])
                if src_pool:
                    extra, tries = [], 0
                    need = args.negatives - len(cands)
                    while len(extra) < need and tries < need * 20:
                        eid = src_pool[rng.integers(len(src_pool))]
                        if eid not in have:
                            have.add(eid); extra.append(eid)
                        tries += 1
                    fallback = set(extra)
                    cands = np.concatenate([cands, np.array(extra)]) if len(cands) else np.array(extra)
            for eid in truth:
                rows_s1.append(s1_id); rows_cand.append(eid)
                rows_lab.append(1); rows_kind.append("pos")
            for eid in cands:
                rows_s1.append(s1_id); rows_cand.append(str(eid))
                rows_lab.append(0); rows_kind.append("pool" if eid in fallback else "overlap")

    out = pa.table({"s1_id": pa.array(rows_s1, type=pa.string()),
                    "cand_id": pa.array(rows_cand, type=pa.string()),
                    "label": pa.array(rows_lab, type=pa.int8()),
                    "neg_kind": pa.array(rows_kind, type=pa.string())})
    ART.mkdir(exist_ok=True)
    pq.write_table(out, ART / "dev_candidates.parquet", compression="zstd")

    n_pos = sum(rows_lab)
    n_pool = rows_kind.count("pool")
    print(f"\nwrote {ART / 'dev_candidates.parquet'}: {out.num_rows:,} pairs "
          f"({n_pos:,} positive, {out.num_rows - n_pos:,} negative "
          f"= {out.num_rows - n_pos - n_pool:,} overlap + {n_pool:,} pool) over {q.num_rows:,} S1"
          f"\n  {n_no_neg:,} S1 had no token-overlap negative (filled from the country pool)"
          f"\n  peak RAM {peak_ram_gb():.1f} GB")


if __name__ == "__main__":
    sys.exit(main())
