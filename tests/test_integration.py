"""Tests for io_utils, checks and ownership on a tiny synthetic fixture.

Run: python -m pytest tests -q
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from checks import check_outputs  # noqa: E402
from io_utils import write_outputs  # noqa: E402
from ownership import decide  # noqa: E402

S1 = ["S1-30", "S1-10", "S1-20", "S1-40"]
S2 = ["S2-100", "S2-101", "S2-102"]
S3 = ["S3-200", "S3-201", "S3-9"]


def _write_source(path: Path, ids) -> None:
    """Minimal source TSV with the real header."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for i in ids:
            f.write(f"{i}\tName {i}\t1 Main St, Town\tUS\n")


@pytest.fixture
def fixture(tmp_path):
    """test_dir with 3 sources plus candidate / score frames."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for n, ids in (("1", S1), ("2", S2), ("3", S3)):
        _write_source(test_dir / f"test_source{n}.tsv", ids)
    scores = pd.DataFrame([
        ("S1-10", "S2-101", 0.9), ("S1-10", "S3-9", 0.8), ("S1-10", "S2-100", 0.4),
        ("S1-20", "S2-101", 0.7), ("S1-20", "S3-200", 0.95), ("S1-20", "S3-201", 0.6),
        ("S1-30", "S3-201", 0.6), ("S1-30", "S2-102", 0.1),
        ("S1-10", "S2-101", 0.9),  # duplicate pair
    ], columns=["s1_id", "cand_id", "score"])
    return test_dir, scores


# ---------- ownership ----------

def test_ownership_toy():
    """Toy asserts for the ownership rule (argmax per target, tie -> smaller S1, >= t)."""
    df = pd.DataFrame({"s1_id": ["S1-2", "S1-1", "S1-3", "S1-3", "S1-1"],
                       "cand_id": ["S2-9", "S2-9", "S2-9", "S3-5", "S3-7"],
                       "score": [0.8, 0.8, 0.5, 0.6, 0.4]})
    got = decide(df, 0.5)
    pairs = set(zip(got.s1_id, got.cand_id))
    assert pairs == {("S1-1", "S2-9"), ("S1-3", "S3-5")}           # tie -> S1-1; S3-7 below t
    assert got.cand_id.is_unique                                     # one owner per target
    assert set(zip(decide(df, 0.6).s1_id, decide(df, 0.6).cand_id)) == pairs  # score == t kept
    assert len(decide(df, 0.81)) == 0
    # The loser is dropped even if its score clears t: S1-3 (0.5) never gets S2-9.
    assert ("S1-3", "S2-9") not in set(zip(decide(df, 0.0).s1_id, decide(df, 0.0).cand_id))
    # Lexicographic, not numeric, tie-break: "S1-10" < "S1-9".
    tie = pd.DataFrame({"s1_id": ["S1-9", "S1-10"], "cand_id": ["S2-1", "S2-1"], "score": [1.0, 1.0]})
    assert decide(tie, 0.5).s1_id.tolist() == ["S1-10"]


def test_ownership_nan_rejected():
    """NaN scores are an upstream bug; refuse them."""
    df = pd.DataFrame({"s1_id": ["S1-1"], "cand_id": ["S2-1"], "score": [float("nan")]})
    with pytest.raises(ValueError):
        decide(df, 0.5)


# ---------- io_utils + checks ----------

def test_write_and_check(fixture, tmp_path):
    """End to end: decide -> write -> strict check passes, exact bytes as expected."""
    test_dir, scores = fixture
    matches = decide(scores, 0.5)
    m, c = write_outputs(S1, scores, matches, tmp_path / "out")
    assert m.read_bytes() == (
        b"source1_entity_id\tmatched_entity_ids\n"
        b"S1-10\tS2-101,S3-9\n"
        b"S1-20\tS3-200,S3-201\n"  # S3-201: 0.6 tie with S1-30 -> smaller id
        b"S1-30\t\n"
        b"S1-40\t\n"
    )
    assert c.read_bytes() == (
        b"source1_entity_id\tcandidate_entity_ids\n"
        b"S1-10\tS2-100,S2-101,S3-9\n"
        b"S1-20\tS2-101,S3-200,S3-201\n"
        b"S1-30\tS2-102,S3-201\n"
        b"S1-40\t\n"
    )
    rep = check_outputs(m, c, test_dir)
    assert rep.ok, rep.lines()


def test_writer_rejects_bad_input(fixture, tmp_path):
    """Writer refuses matches outside candidates, unknown S1, and S1 targets."""
    _, scores = fixture
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="not in candidates"):
        write_outputs(S1, scores, pd.DataFrame({"s1_id": ["S1-40"], "cand_id": ["S2-100"]}), out)
    with pytest.raises(ValueError, match="not in test S1"):
        write_outputs(S1, pd.DataFrame({"s1_id": ["S1-99"], "cand_id": ["S2-100"]}), scores.iloc[:0], out)
    with pytest.raises(ValueError, match="S2-/S3-"):
        write_outputs(S1, pd.DataFrame({"s1_id": ["S1-10"], "cand_id": ["S1-20"]}), scores.iloc[:0], out)


def _corrupt(fixture, tmp_path, which, old: bytes, new: bytes):
    """Write good outputs, apply a byte-level edit to one file, return the Report."""
    test_dir, scores = fixture
    m, c = write_outputs(S1, scores, decide(scores, 0.5), tmp_path / "out")
    p = m if which == "m" else c
    data = p.read_bytes()
    assert old in data
    p.write_bytes(data.replace(old, new, 1))
    return check_outputs(m, c, test_dir)


@pytest.mark.parametrize("which,old,new,expect", [
    ("m", b"S1-40\t\n", b"", "test S1 missing a row"),
    ("m", b"S1-40\t\n", b"S1-40\t\nS1-40\t\n", "duplicate S1 row"),
    ("m", b"S1-40\t\n", b"S1-40\t\nS1-99\t\n", "S1 id not in test S1"),
    ("m", b"S1-40\t\n", b"S1-40\tS2-555\n", "id not in test S2/S3"),
    ("m", b"S2-101,S3-9", b"S2-101,S2-101,S3-9", "duplicate id in list"),
    ("m", b"S2-101,S3-9", b"S3-9,S2-101", "ids not sorted"),
    ("m", b"S1-40\t\n", b"S1-40\tS1-10\n", "non S2-S3 id"),
    ("m", b"S1-30\t\n", b"S1-30\tS3-201\n", "more than one S1"),
    ("m", b"S1-40\t\n", b"S1-40\tS2-102\n", "match not in candidates"),
    ("m", b"S1-40\t\n", b"S1-40\t\r\n", "CRLF"),
    ("m", b"S1-40\t\n", b"S1-40\t", "not LF-terminated"),
    ("m", b"S1-10\t", b'"S1-10"\t', "quote"),
    ("m", b"source1_entity_id", b"\xef\xbb\xbfsource1_entity_id", "BOM"),
    ("m", b"matched_entity_ids", b"matches", "bad header"),
    ("c", b"S1-10\tS2-100,", b"S1-10\t\xff", "invalid UTF-8"),
    ("c", b"S1-30\tS2-102,S3-201\n", b"", "missing a row"),
])
def test_checker_catches(fixture, tmp_path, which, old, new, expect):
    """Each corruption is reported under the expected category."""
    rep = _corrupt(fixture, tmp_path, which, old, new)
    assert not rep.ok
    assert any(expect in line for line in rep.lines()), rep.lines()


def test_checker_row_order(fixture, tmp_path):
    """Rows out of S1 order are rejected."""
    rep = _corrupt(fixture, tmp_path, "m", b"S1-10\tS2-101,S3-9\nS1-20\tS3-200,S3-201\n",
                   b"S1-20\tS3-200,S3-201\nS1-10\tS2-101,S3-9\n")
    assert any("not strictly sorted" in line for line in rep.lines()), rep.lines()


def test_organizer_validator(fixture, tmp_path):
    """utils/validate_submission.py --check-ids passes on our fixture output.

    utils/ is gitignored, so set VALIDATOR=<path> when it is not in this checkout.
    """
    validator = Path(os.environ.get("VALIDATOR", ROOT / "utils" / "validate_submission.py"))
    if not validator.exists():
        pytest.skip("organizer validator not in this checkout (utils/ is gitignored)")
    test_dir, scores = fixture
    m, c = write_outputs(S1, scores, decide(scores, 0.5), tmp_path / "out")
    r = subprocess.run([sys.executable, str(validator), "--matching", str(m), "--candidate", str(c),
                        "--test-dir", str(test_dir), "--check-ids"], capture_output=True, text=True)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr
    assert "WARNING" not in r.stdout, r.stdout
