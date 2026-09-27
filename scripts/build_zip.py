"""Build <team>_submission.zip from the submitted TSVs and the committed package.

Layout (organizer spec):
  output/matching_results.tsv
  output/candidate_pairs.tsv
  code/business_entity_resolution/...     (src/, run.py, README.md, requirements.txt, models/, artifacts/)
  Documentation_template.md               (= Documentation_filled.md)

Safety:
  * both TSVs must match the SHA-256 listed in <submission-dir>/SHA256SUMS.txt
    (sha256sum format: "<hex>  <name>" or "<hex> *<name>"), so the zip carries exactly the files
    that were uploaded to the leaderboard;
  * the code and documentation come from a git ref via `git archive` (default origin/main), so
    only committed files are packed: no caches, no local artifacts, no uncommitted edits;
  * an existing zip is never overwritten (pass --overwrite), and the zip is written to a temp
    file and renamed only when complete.

Usage:
  python scripts/build_zip.py --submission-dir submissions/03_v3 --team-name "Commit&Pray"
"""
import argparse
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TSVS = ("matching_results.tsv", "candidate_pairs.tsv")
HEADERS = {"matching_results.tsv": b"source1_entity_id\tmatched_entity_ids",
           "candidate_pairs.tsv": b"source1_entity_id\tcandidate_entity_ids"}
PACKAGE = "code/business_entity_resolution"
DOC_SRC, DOC_DST = "Documentation_filled.md", "Documentation_template.md"
EXCLUDE = re.compile(r"(^|/)(__pycache__|\.pytest_cache|\.ipynb_checkpoints)(/|$)|\.py[co]$|(^|/)\.DS_Store$")


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_sums(path: Path) -> dict:
    """{file name: hex digest} from a sha256sum-style file."""
    sums = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.fullmatch(r"\s*([0-9a-fA-F]{64})\s+\*?(.+?)\s*", line)
        if not m:
            raise SystemExit(f"{path}:{n}: not a '<sha256>  <name>' line: {line!r}")
        sums[Path(m.group(2)).name] = m.group(1).lower()
    return sums


def verify_submission(sub: Path) -> dict:
    """Check both TSVs exist, start with the right header and match SHA256SUMS.txt; return their paths."""
    sums_path = sub / "SHA256SUMS.txt"
    if not sums_path.is_file():
        raise SystemExit(f"missing {sums_path}")
    sums = read_sums(sums_path)
    files = {}
    for name in TSVS:
        p = sub / name
        if not p.is_file():
            raise SystemExit(f"missing {p}")
        if name not in sums:
            raise SystemExit(f"{sums_path} has no entry for {name}")
        with open(p, "rb") as f:
            head = f.readline().rstrip(b"\r\n")
        if head != HEADERS[name]:
            raise SystemExit(f"{p}: unexpected header {head[:80]!r}")
        got = sha256(p)
        if got != sums[name]:
            raise SystemExit(f"SHA-256 mismatch for {p}:\n  expected {sums[name]}\n  actual   {got}")
        print(f"ok  {name}  {got}  ({p.stat().st_size:,} bytes)")
        files[name] = p
    return files


def git_files(ref: str, paths: list) -> list:
    """[(path, bytes, mode)] for the committed files under paths at ref (via git archive)."""
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", f"{ref}^{{commit}}"],
                            capture_output=True, text=True)
    if commit.returncode:
        raise SystemExit(f"unknown git ref {ref!r}: {commit.stderr.strip()}")
    print(f"code + docs from {ref} = {commit.stdout.strip()}")
    tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", ref, "--", *paths],
                         capture_output=True)
    if tar.returncode:
        raise SystemExit(f"git archive failed: {tar.stderr.decode(errors='replace').strip()}")
    out = []
    with tarfile.open(fileobj=io.BytesIO(tar.stdout)) as t:
        for m in t.getmembers():
            if m.isfile() and not EXCLUDE.search(m.name):
                out.append((m.name, t.extractfile(m).read(), m.mode))
    return out


def safe_team(name: str) -> str:
    """Team name usable in a file name (keeps letters, digits, & . _ -)."""
    s = re.sub(r"[^A-Za-z0-9&._-]+", "_", name.strip()).strip("_")
    if not s:
        raise SystemExit("--team-name is empty")
    return s


def build(sub: Path, team: str, ref: str, out_dir: Path, overwrite: bool) -> Path:
    """Verify the submission and write the zip; return its path."""
    files = verify_submission(sub)
    code = git_files(ref, [PACKAGE, DOC_SRC])
    pkg = [(n, b, m) for n, b, m in code if n.startswith(PACKAGE + "/")]
    doc = [b for n, b, m in code if n == DOC_SRC]
    if not doc:
        raise SystemExit(f"{DOC_SRC} not found at {ref}")
    for need in ("README.md", "requirements.txt", "run.py", "src/"):
        if not any(n.startswith(f"{PACKAGE}/{need}") for n, _, _ in pkg):
            raise SystemExit(f"{PACKAGE}/{need} not found at {ref}")

    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / f"{safe_team(team)}_submission.zip"
    if zpath.exists() and not overwrite:
        raise SystemExit(f"{zpath} exists; pass --overwrite to replace it")
    tmp = zpath.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name, p in files.items():
                z.write(p, f"output/{name}")
            for name, data, mode in sorted(pkg):
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 27, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (mode & 0o777 | 0o100000) << 16
                z.writestr(info, data)
            info = zipfile.ZipInfo(DOC_DST, date_time=(2026, 9, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, doc[0])
        with zipfile.ZipFile(tmp) as z:  # re-read: CRC check of every member
            bad = z.testzip()
            if bad:
                raise SystemExit(f"corrupt zip member {bad}")
        os.replace(tmp, zpath)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return zpath


def list_zip(zpath: Path) -> None:
    """Print every member with its size, then the zip's own size."""
    with zipfile.ZipFile(zpath) as z:
        infos = z.infolist()
        for i in infos:
            print(f"{i.file_size:>13,}  {i.compress_size:>13,}  {i.filename}")
        total = sum(i.file_size for i in infos)
    print(f"{len(infos)} files, {total:,} bytes uncompressed; {zpath} = {zpath.stat().st_size:,} bytes "
          f"({zpath.stat().st_size / 2**20:.1f} MiB)")


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission-dir", required=True, help="folder with both TSVs and SHA256SUMS.txt")
    ap.add_argument("--team-name", required=True)
    ap.add_argument("--ref", default="origin/main", help="git ref for code + documentation (default origin/main)")
    ap.add_argument("--out-dir", default=".", help="where to write <team>_submission.zip")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    zpath = build(Path(a.submission_dir), a.team_name, a.ref, Path(a.out_dir), a.overwrite)
    print()
    list_zip(zpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
