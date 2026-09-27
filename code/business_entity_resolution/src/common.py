"""Shared paths, IO and small utilities used by every pipeline stage."""
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
ART = ROOT / "artifacts"
SEED = 42
SOURCES = ("S1", "S2", "S3")


def read_tsv(path: Path) -> pa.Table:
    """Read a TSV as an Arrow table with every column typed as string.

    Quotes are treated literally (quote_char=False) and empty fields stay ''.
    """
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
    return pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in header},
                                             strings_can_be_null=False),
    )


def source_path(split: str, src: str) -> Path:
    """Raw TSV path for split ('train'/'test') and source ('S1'/'S2'/'S3')."""
    return DATA / split / f"{split}_source{src[-1]}.tsv"


def norm_path(split: str, src: str) -> Path:
    """Cached normalized parquet for split/source."""
    return ART / f"{split}_{src.lower()}_norm.parquet"


def peak_ram_gb() -> float:
    """Peak resident memory of this process in GB (Windows peak working set, else ru_maxrss)."""
    import psutil
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)
    if peak is None:  # Linux/macOS
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != "darwin" else 1)
    return peak / 2**30


@contextmanager
def timed(label: str, sink: dict | None = None):
    """Print (to stderr) and optionally record the wall time of a block."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[{dt:7.1f}s] {label}  (peak RAM {peak_ram_gb():.1f} GB)", file=sys.stderr, flush=True)
    if sink is not None:
        sink[label] = dt


def utf8_stdout() -> None:
    """Make stdout UTF-8 so non-Latin text prints on Windows consoles."""
    sys.stdout.reconfigure(encoding="utf-8")
