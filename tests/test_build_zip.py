"""scripts/build_zip.py on a tiny fake submission folder (temp dir only; packs the committed HEAD)."""
import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_zip as B  # noqa: E402

MATCH = b"source1_entity_id\tmatched_entity_ids\nS1-1\tS2-5\nS1-2\t\n"
CAND = b"source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-5,S3-7\nS1-2\tS2-9\n"


def _sub(tmp_path, sums_text=None):
    """Fake submission folder with both TSVs and a SHA256SUMS.txt (correct unless overridden)."""
    d = tmp_path / "sub"
    d.mkdir()
    (d / "matching_results.tsv").write_bytes(MATCH)
    (d / "candidate_pairs.tsv").write_bytes(CAND)
    if sums_text is None:
        sums_text = (f"{hashlib.sha256(MATCH).hexdigest()}  matching_results.tsv\n"
                     f"{hashlib.sha256(CAND).hexdigest()} *candidate_pairs.tsv\n")
    if sums_text is not False:
        (d / "SHA256SUMS.txt").write_text(sums_text, encoding="utf-8")
    return d


def _git_show(path: str) -> bytes:
    """A file's committed bytes at HEAD."""
    return subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{path}"], capture_output=True, check=True).stdout


def _model_dir() -> str:
    """MODEL_DIR of the committed package run.py (the shipped model folder)."""
    m = re.search(r'^MODEL_DIR = "([^"]+)"', _git_show("code/business_entity_resolution/run.py").decode(), re.M)
    assert m, "run.py must define MODEL_DIR"
    return m.group(1)


def test_build_zip_layout(tmp_path, capsys):
    """Zip has output/ TSVs, the committed package (no caches) and the documentation renamed."""
    md = _model_dir()
    z = B.build(_sub(tmp_path), "Commit&Pray", "HEAD", tmp_path / "out", overwrite=False)
    assert z.name == "Commit&Pray_submission.zip"
    B.list_zip(z)
    assert "Commit&Pray_submission.zip =" in capsys.readouterr().out
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert zf.read("output/matching_results.tsv") == MATCH
        assert zf.read("output/candidate_pairs.tsv") == CAND
        assert zf.read("Documentation_template.md") == _git_show("Documentation_filled.md")
        pkg = [n for n in names if n.startswith("code/business_entity_resolution/")]
        for need in ("README.md", "requirements.txt", "run.py", "src/score_test.py", "src/features.py",
                     f"{md}/lgbm.txt", f"{md}/bundle.json", "artifacts/addr_tfidf.joblib"):
            assert f"code/business_entity_resolution/{need}" in names, need
        assert not [n for n in names if "__pycache__" in n or n.endswith((".pyc", ".pyo"))]
        assert set(names) == {"output/matching_results.tsv", "output/candidate_pairs.tsv",
                              "Documentation_template.md", *pkg}
        for n in ("code/business_entity_resolution/run.py", f"code/business_entity_resolution/{md}/bundle.json"):
            assert zf.read(n) == _git_show(n)


def test_build_zip_refuses_bad_hash(tmp_path):
    """A digest mismatch stops the build and leaves no zip."""
    sums = f"{'0' * 64}  matching_results.tsv\n{hashlib.sha256(CAND).hexdigest()}  candidate_pairs.tsv\n"
    with pytest.raises(SystemExit, match="mismatch"):
        B.build(_sub(tmp_path, sums), "T", "HEAD", tmp_path / "out", overwrite=False)
    assert not list((tmp_path / "out").glob("*.zip*")) if (tmp_path / "out").exists() else True


def test_build_zip_refuses_missing_sums_and_bad_header(tmp_path):
    """No SHA256SUMS.txt, or a TSV with the wrong header, is refused."""
    with pytest.raises(SystemExit, match="SHA256SUMS"):
        B.build(_sub(tmp_path, False), "T", "HEAD", tmp_path / "out", overwrite=False)
    d = tmp_path / "bad"
    d.mkdir()
    (d / "matching_results.tsv").write_bytes(b"a,b\n")
    (d / "candidate_pairs.tsv").write_bytes(CAND)
    (d / "SHA256SUMS.txt").write_text(f"{hashlib.sha256(b'a,b').hexdigest()}  matching_results.tsv\n"
                                      f"{hashlib.sha256(CAND).hexdigest()}  candidate_pairs.tsv\n")
    with pytest.raises(SystemExit, match="header"):
        B.build(d, "T", "HEAD", tmp_path / "out", overwrite=False)


def test_build_zip_no_overwrite(tmp_path):
    """An existing zip is kept unless --overwrite."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "T_submission.zip").write_bytes(b"keep")
    with pytest.raises(SystemExit, match="exists"):
        B.build(_sub(tmp_path), "T", "HEAD", out, overwrite=False)
    assert (out / "T_submission.zip").read_bytes() == b"keep"
