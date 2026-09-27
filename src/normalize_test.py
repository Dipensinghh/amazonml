"""Normalize the TEST sources with normalize.py (imported unchanged) into parquet.

Same column recipe as normalize.normalize_source, but with explicit input/output
folders so it can read the main checkout's dataset/ from a worktree, and it never
overwrites an existing file.

Usage:
  python src/normalize_test.py --root .
Writes <root>/artifacts/test_s{1,2,3}_norm.parquet (same names as common.norm_path).
"""
import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import SOURCES, peak_ram_gb, read_tsv, utf8_stdout
from normalize import normalize_addresses, normalize_names


def normalize_file(src_path: Path, out_path: Path) -> pa.Table:
    """Normalize one raw source TSV and write it to out_path (must not exist)."""
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; refusing to overwrite")
    raw = read_tsv(src_path)
    out = pa.table({"entity_id": raw["entity_id"],
                    "country": pc.utf8_trim_whitespace(raw["country"]),
                    **normalize_names(raw["business_name"]),
                    **normalize_addresses(raw["business_address"])})
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(out, tmp, compression="zstd")
    tmp.rename(out_path)
    return out


def main() -> int:
    """Normalize every test source and print runtime, peak RAM, per-country counts."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="checkout holding dataset/test and artifacts/")
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    a = ap.parse_args()
    utf8_stdout()
    root = Path(a.root)
    outs = {s: root / "artifacts" / f"test_{s.lower()}_norm.parquet" for s in a.sources}
    clash = [str(p) for p in outs.values() if p.exists()]
    if clash:  # check all up front so we never leave a partial set
        print(f"refusing to overwrite: {clash}")
        return 1
    t_all = time.perf_counter()
    for s in a.sources:
        t0 = time.perf_counter()
        tab = normalize_file(root / "dataset" / "test" / f"test_source{s[-1]}.tsv", outs[s])
        counts = tab.group_by("country").aggregate([("entity_id", "count")]).to_pylist()
        counts = ", ".join(f"{c['country']} {c['entity_id_count']:,}" for c in
                           sorted(counts, key=lambda c: -c["entity_id_count"]))
        print(f"{s}: {tab.num_rows:,} rows in {time.perf_counter() - t0:.0f}s "
              f"(peak RAM {peak_ram_gb():.1f} GB) | {counts}", flush=True)
        del tab
    print(f"total {time.perf_counter() - t_all:.0f}s, peak RAM {peak_ram_gb():.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
