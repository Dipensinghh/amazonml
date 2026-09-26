"""Strict, streaming checker for the two submission files.

Stricter than utils/validate_submission.py: in addition to the organizer rules it
enforces our own format contract (LF only, no BOM, no quotes, sorted rows and
ids, S1 coverage exactly equal to test S1), requires every id to exist in test
S2/S3, requires matches ⊆ candidates, and (for matches) that each S2/S3 id is
assigned to at most one S1.

Memory: ids are encoded as int64 (S2-123 -> 2*10**12 + 123) and kept in sorted
numpy arrays, so the full test set (1.7M S1, 10M S2/S3, tens of millions of
candidate pairs) stays well under 4 GB. Files are read line by line.

Usage:
    python src/checks.py --out-dir output --test-dir dataset/test
Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

MATCHING_HEADER = b"source1_entity_id\tmatched_entity_ids"
CANDIDATE_HEADER = b"source1_entity_id\tcandidate_entity_ids"
_ID_RE = re.compile(r"S([123])-([1-9][0-9]{0,11})\Z")
_BASE = 10**12
_CHUNK = 200_000  # rows buffered before a vectorized membership check
MAX_EXAMPLES = 5


def encode_id(s: str) -> int:
    """Map a canonical id 'S<k>-<digits>' to k*10**12 + digits; -1 if malformed."""
    m = _ID_RE.match(s)
    return int(m.group(1)) * _BASE + int(m.group(2)) if m else -1


def _fmt(v) -> str:
    """Inverse of encode_id for error messages."""
    return f"S{int(v) // _BASE}-{int(v) % _BASE}"


def load_source_ids(path: Path) -> np.ndarray:
    """Sorted unique int64 ids from the first column of a source TSV (header skipped)."""
    out, bad = [], []
    with open(path, encoding="utf-8", newline="") as f:
        next(f, None)
        for line in f:
            s = line.split("\t", 1)[0].strip()
            if s:
                v = encode_id(s)
                (bad if v < 0 else out).append(s if v < 0 else v)
    if bad:  # our int encoding relies on canonical ids; refuse silently-lossy input
        raise ValueError(f"{path}: non-canonical ids, e.g. {bad[:3]}")
    return np.unique(np.asarray(out, dtype=np.int64))


def _isin_sorted(values: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    """Vectorized membership test against a sorted reference array."""
    if len(sorted_ref) == 0:
        return np.zeros(len(values), dtype=bool)
    pos = np.searchsorted(sorted_ref, values)
    pos[pos == len(sorted_ref)] = 0
    return sorted_ref[pos] == values


class Report:
    """Collects errors per category with a count and a few example offenders."""

    def __init__(self):
        self.cats: dict[str, list] = {}

    def add(self, category: str, example) -> None:
        """Record one offence under category."""
        c = self.cats.setdefault(category, [0, []])
        c[0] += 1
        if len(c[1]) < MAX_EXAMPLES:
            c[1].append(example)

    def add_bulk(self, category: str, examples: list, n: int) -> None:
        """Record n offences under category, keeping the first few examples."""
        if n:
            c = self.cats.setdefault(category, [0, []])
            c[0] += n
            c[1].extend(examples[:MAX_EXAMPLES - len(c[1])])

    @property
    def ok(self) -> bool:
        """True when no errors were recorded."""
        return not self.cats

    def lines(self) -> list[str]:
        """Human-readable error lines."""
        return [f"{cat}: {n} (e.g. {ex})" for cat, (n, ex) in self.cats.items()]


def _scan_file(path: Path, header: bytes, tag: str, s1_ref: np.ndarray, tgt_ref: np.ndarray,
               rep: Report) -> np.ndarray:
    """Stream one output file, recording errors in rep.

    Returns int64 pair keys s1_idx * len(tgt_ref) + tgt_idx for every valid
    (S1, target) pair, used for the matches ⊆ candidates check.
    """
    n_tgt = len(tgt_ref)
    seen_s1 = np.zeros(len(s1_ref), dtype=bool)
    keys, buf_s1, buf_tgt = [], [], []

    def flush():
        """Membership-check buffered targets and turn them into pair keys."""
        if not buf_tgt:
            return
        t = np.asarray(buf_tgt, dtype=np.int64)
        s = np.asarray(buf_s1, dtype=np.int64)
        ok = _isin_sorted(t, tgt_ref)
        rep.add_bulk(f"{tag}: id not in test S2/S3", [_fmt(v) for v in t[~ok][:MAX_EXAMPLES]],
                     int((~ok).sum()))
        keys.append(s[ok] * n_tgt + np.searchsorted(tgt_ref, t[ok]))
        buf_s1.clear()
        buf_tgt.clear()

    with open(path, "rb") as f:
        raw = f.readline()
        if raw.startswith(b"\xef\xbb\xbf"):
            rep.add(f"{tag}: BOM at start of file", "")
            raw = raw[3:]
        if raw.rstrip(b"\r\n") != header:
            rep.add(f"{tag}: bad header", raw[:80])
        if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
            rep.add(f"{tag}: header line not LF-terminated", raw[-4:])
        prev = None
        for ln, raw in enumerate(f, start=2):
            if raw.endswith(b"\r\n"):
                rep.add(f"{tag}: CRLF line ending", ln)
            elif not raw.endswith(b"\n"):
                rep.add(f"{tag}: last line not LF-terminated", ln)
            if b'"' in raw:
                rep.add(f"{tag}: quote character", ln)
            try:
                line = raw.rstrip(b"\r\n").decode("utf-8")
            except UnicodeDecodeError:
                rep.add(f"{tag}: invalid UTF-8", ln)
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                rep.add(f"{tag}: row does not have exactly 2 columns", ln)
                continue
            s1, rest = parts
            if prev is not None and s1 <= prev:
                rep.add(f"{tag}: rows not strictly sorted by S1 id", s1)
            prev = s1
            v1 = encode_id(s1)
            i = int(np.searchsorted(s1_ref, v1))
            if v1 < 0 or i >= len(s1_ref) or s1_ref[i] != v1:
                rep.add(f"{tag}: S1 id not in test S1", s1)
                continue
            if seen_s1[i]:
                rep.add(f"{tag}: duplicate S1 row", s1)
                continue
            seen_s1[i] = True
            if not rest:
                continue
            ids = rest.split(",")
            if ids != sorted(ids):
                rep.add(f"{tag}: ids not sorted", s1)
            if len(set(ids)) != len(ids):
                rep.add(f"{tag}: duplicate id in list", s1)
            for x in ids:
                v = encode_id(x)
                if v < 0 or v // _BASE == 1:
                    rep.add(f"{tag}: malformed / non S2-S3 id", f"{s1}:{x!r}")
                else:
                    buf_s1.append(i)
                    buf_tgt.append(v)
            if len(buf_tgt) >= _CHUNK:
                flush()
        flush()
    missing = np.flatnonzero(~seen_s1)
    rep.add_bulk(f"{tag}: test S1 missing a row", [_fmt(v) for v in s1_ref[missing[:MAX_EXAMPLES]]],
                 len(missing))
    return np.concatenate(keys) if keys else np.empty(0, dtype=np.int64)


def check_outputs(matching_path, candidate_path, test_dir) -> Report:
    """Run every check on the two output files against test_dir; return a Report."""
    test_dir = Path(test_dir)
    s1_ref = load_source_ids(test_dir / "test_source1.tsv")
    tgt_ref = np.union1d(load_source_ids(test_dir / "test_source2.tsv"),
                         load_source_ids(test_dir / "test_source3.tsv"))
    if len(s1_ref) * max(len(tgt_ref), 1) >= 2**62:
        raise OverflowError("pair keys would overflow int64")
    rep = Report()
    mkeys = _scan_file(Path(matching_path), MATCHING_HEADER, "matching", s1_ref, tgt_ref, rep)
    tgt_idx = mkeys % max(len(tgt_ref), 1)
    uniq, cnt = np.unique(tgt_idx, return_counts=True)
    multi = tgt_ref[uniq[cnt > 1]]
    rep.add_bulk("matching: S2/S3 id matched to more than one S1",
                 [_fmt(v) for v in multi[:MAX_EXAMPLES]], len(multi))
    ckeys = _scan_file(Path(candidate_path), CANDIDATE_HEADER, "candidate", s1_ref, tgt_ref, rep)
    notin = ~np.isin(mkeys, ckeys)
    del ckeys
    ex = [f"{_fmt(s1_ref[k // len(tgt_ref)])}:{_fmt(tgt_ref[k % len(tgt_ref)])}"
          for k in mkeys[notin][:MAX_EXAMPLES]]
    rep.add_bulk("match not in candidates", ex, int(notin.sum()))
    return rep


def main() -> int:
    """CLI entry point; prints PASS/FAIL and returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--test-dir", default="dataset/test")
    a = ap.parse_args()
    out = Path(a.out_dir)
    rep = check_outputs(out / "matching_results.tsv", out / "candidate_pairs.tsv", a.test_dir)
    try:
        import psutil
        mi = psutil.Process().memory_info()
        print(f"peak RAM {getattr(mi, 'peak_wset', mi.rss) / 2**30:.2f} GB")
    except ImportError:
        pass
    if rep.ok:
        print("PASS")
        return 0
    print("FAIL")
    for line in rep.lines():
        print("  " + line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
