import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

import crawfish
from crawfish import result_changes
from crawfish.result_changes import ResultChange
from crawfish.utils import caching
from crawfish.utils.caching import validate_cache_dir, read_cache_stamp, CACHE_STAMP_FILENAME
from crawfish.utils.testing import EXAMPLE_CALC_DIRS_DIR


@pytest.fixture
def calc(tmp_path):
    """A fake calculation directory with two source files, one missing optional file and a populated cache."""
    (tmp_path / "a").write_text("aaaa")
    (tmp_path / "b").write_text("bbbb")
    cache_dir = tmp_path / ".crawfish_cache"
    cache_dir.mkdir()
    sources = {"a": tmp_path / "a", "b": tmp_path / "b", "opt": None}
    return tmp_path, cache_dir, sources


def populate(cache_dir: Path) -> dict[str, Path]:
    artifacts = {
        "parse": cache_dir / "proj_tju.npy",
        "matrices": cache_dir / "settings" / "h_uu.npy",
        "spectra": cache_dir / "settings" / "erange_0" / "pdos_0" / "spec.npy",
    }
    for path in artifacts.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cached")
    return artifacts


def shift_mtime(path: Path, seconds: float):
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + seconds))


def test_fresh_cache_is_stamped(calc, capsys):
    _, cache_dir, sources = calc
    validate_cache_dir(cache_dir, sources)
    assert capsys.readouterr().out == ""
    stamp = read_cache_stamp(cache_dir)
    assert stamp["code"]["version"] is not None
    assert set(stamp["source_files"]) == {"a", "b", "opt"}
    assert stamp["source_files"]["opt"] is None
    assert stamp["source_files"]["a"]["size"] == 4


def test_unstamped_cache_is_cleared(calc, capsys):
    _, cache_dir, sources = calc
    artifacts = populate(cache_dir)
    validate_cache_dir(cache_dir, sources)
    assert "Clearing" in capsys.readouterr().out
    assert not any(path.exists() for path in artifacts.values())
    assert read_cache_stamp(cache_dir) is not None


def test_corrupt_stamp_is_cleared(calc):
    _, cache_dir, sources = calc
    validate_cache_dir(cache_dir, sources)
    artifacts = populate(cache_dir)
    (cache_dir / CACHE_STAMP_FILENAME).write_text("{not json")
    validate_cache_dir(cache_dir, sources)
    assert not any(path.exists() for path in artifacts.values())


def test_unchanged_cache_is_kept(calc, capsys):
    _, cache_dir, sources = calc
    validate_cache_dir(cache_dir, sources)
    artifacts = populate(cache_dir)
    validate_cache_dir(cache_dir, sources)
    assert capsys.readouterr().out == ""
    assert all(path.exists() for path in artifacts.values())


@pytest.mark.parametrize("new_content", ["cccc", "aaaaa"], ids=["same_size", "new_size"])
def test_changed_source_file_clears_cache(calc, new_content):
    tmp_path, cache_dir, sources = calc
    validate_cache_dir(cache_dir, sources)
    artifacts = populate(cache_dir)
    (tmp_path / "a").write_text(new_content)
    shift_mtime(tmp_path / "a", 100)
    validate_cache_dir(cache_dir, sources)
    assert not any(path.exists() for path in artifacts.values())


def test_optional_file_appearing_clears_cache(calc):
    tmp_path, cache_dir, sources = calc
    validate_cache_dir(cache_dir, sources)
    artifacts = populate(cache_dir)
    (tmp_path / "opt").write_text("new")
    validate_cache_dir(cache_dir, {**sources, "opt": tmp_path / "opt"})
    assert not any(path.exists() for path in artifacts.values())


@pytest.mark.parametrize("seconds", [-5 * 3600, 3600, 86400 * 30], ids=["tz_shift_back", "tz_shift_fwd", "recopied"])
def test_moved_mtime_with_same_content_keeps_cache(calc, seconds):
    # e.g. `scp` without -p, or a zip archive extracted in a different time zone
    tmp_path, cache_dir, sources = calc
    validate_cache_dir(cache_dir, sources)
    artifacts = populate(cache_dir)
    shift_mtime(tmp_path / "a", seconds)
    validate_cache_dir(cache_dir, sources)
    assert all(path.exists() for path in artifacts.values())
    # The recorded mtime follows the file so the next check doesn't need to hash it again
    assert read_cache_stamp(cache_dir)["source_files"]["a"]["mtime"] == (tmp_path / "a").stat().st_mtime


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset is unavailable on this platform")
def test_reader_time_zone_does_not_matter(calc, monkeypatch):
    _, cache_dir, sources = calc
    monkeypatch.setenv("TZ", "America/Denver")
    time.tzset()
    try:
        validate_cache_dir(cache_dir, sources)
        artifacts = populate(cache_dir)
        monkeypatch.setenv("TZ", "Asia/Tokyo")
        time.tzset()

        def fail_hash(*args, **kwargs):
            raise AssertionError("an unchanged file should not need hashing")

        monkeypatch.setattr(caching, "get_file_hash", fail_hash)
        validate_cache_dir(cache_dir, sources)
    finally:
        monkeypatch.undo()
        time.tzset()
    assert all(path.exists() for path in artifacts.values())


@pytest.mark.parametrize(
    "stage, cleared",
    [("parse", {"parse", "matrices", "spectra"}), ("matrices", {"matrices", "spectra"}), ("spectra", {"spectra"})],
)
def test_result_change_clears_stage_and_downstream(calc, monkeypatch, capsys, stage, cleared):
    _, cache_dir, sources = calc
    # Written by code that doesn't know about the change yet
    monkeypatch.setattr(result_changes, "RESULT_CHANGES", [])
    validate_cache_dir(cache_dir, sources)
    artifacts = populate(cache_dir)
    change = ResultChange(commit="0123abcd", version=crawfish.__version__, stage=stage, description="test change")
    monkeypatch.setattr(result_changes, "RESULT_CHANGES", [change])
    validate_cache_dir(cache_dir, sources)
    assert "test change" in capsys.readouterr().out
    for artifact_stage, path in artifacts.items():
        assert path.exists() == (artifact_stage not in cleared)
    # The cache is now stamped with the running code, so it isn't cleared again
    artifacts = populate(cache_dir)
    validate_cache_dir(cache_dir, sources)
    assert all(path.exists() for path in artifacts.values())


def test_elecdata_cache_validation(tmp_path, capsys):
    from crawfish.core.elecdata import ElecData

    calc_dir = tmp_path / "N2_bare_min"
    shutil.copytree(Path(EXAMPLE_CALC_DIRS_DIR) / "N2_bare_min", calc_dir, ignore=shutil.ignore_patterns(".crawfish_cache"))
    edata = ElecData(calc_dir)
    _ = edata.h_uu
    stamp = json.loads((edata.cache_dir / CACHE_STAMP_FILENAME).read_text())
    assert set(stamp["source_files"]) == {"bandProjections", "eigenvals", "out", "kPts", "fillings"}
    assert edata.proj_tju_filepath.exists()
    capsys.readouterr()
    # Touching a file without changing it keeps the cache
    shift_mtime(calc_dir / "eigenvals", 3600)
    ElecData(calc_dir)
    assert "Clearing" not in capsys.readouterr().out
    # Changing a file clears it
    with open(calc_dir / "out", "a") as f:
        f.write("\n")
    edata = ElecData(calc_dir)
    assert "Clearing" in capsys.readouterr().out
    assert np.all(np.isfinite(edata.h_uu))
