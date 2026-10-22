"""Identify the running crawfish code and compare cached data against result-changing updates.

Cached data is stamped with the commits of the `crawfish.result_changes.RESULT_CHANGES` entries the
writing code knew about. Any current entry missing from that list was not applied when the cache was
written. This needs no git and no version ordering, so it behaves the same on a git checkout and on a
`pip install` on a remote cluster.

The version and commit of the writing code are recorded too, for diagnostics. The commit is read from
git when crawfish runs from a git checkout (editable install), and from pip's `direct_url.json` when it
was installed with `pip install git+https://...`. Plain `pip install .` records no commit.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import crawfish
from crawfish import result_changes
from crawfish.result_changes import ResultChange

PACKAGE_DIR = Path(crawfish.__file__).resolve().parent


def get_crawfish_version() -> str:
    """Return the version of the running crawfish code."""
    return crawfish.__version__


@lru_cache(maxsize=None)
def get_crawfish_commit() -> str | None:
    """Return the commit hash of the running crawfish code, or None if it can't be determined."""
    repo_dir = _get_crawfish_repo_dir()
    if repo_dir is not None:
        return _run_git(["rev-parse", "HEAD"], repo_dir)
    try:
        direct_url = metadata.distribution("crawfish").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if direct_url is None:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def get_code_stamp() -> dict:
    """Return the stamp of the running code recorded alongside cached files."""
    return {
        "version": get_crawfish_version(),
        "commit": get_crawfish_commit(),
        "known_changes": [change.commit for change in result_changes.RESULT_CHANGES],
    }


def get_postdating_changes(
    code_stamp: dict | None, changes: list[ResultChange] | None = None
) -> list[ResultChange]:
    """Return the result-changing updates that cached data stamped with `code_stamp` predates."""
    if changes is None:
        changes = result_changes.RESULT_CHANGES
    return [change for change in changes if stamp_predates_change(code_stamp, change)]


def stamp_predates_change(code_stamp: dict | None, change: ResultChange) -> bool:
    """Return True if the code that wrote `code_stamp` did not include `change`.

    Missing or unreadable stamps are treated as predating everything.
    """
    if not code_stamp or not isinstance(code_stamp.get("known_changes"), list):
        return True
    return change.commit not in code_stamp["known_changes"]


@lru_cache(maxsize=None)
def _get_crawfish_repo_dir() -> Path | None:
    """Return the crawfish git repository root if crawfish is running from a checkout of it."""
    toplevel = _run_git(["rev-parse", "--show-toplevel"], PACKAGE_DIR)
    if toplevel is None:
        return None
    repo_dir = Path(toplevel).resolve()
    # Guard against a non-editable install that happens to sit inside some other git repo
    if repo_dir / "src" / "crawfish" != PACKAGE_DIR:
        return None
    return repo_dir


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
