import pytest

import crawfish
from crawfish import result_changes
from crawfish.result_changes import RESULT_CHANGES, CACHE_STAGES, ResultChange
from crawfish.utils.versioning import stamp_predates_change, get_code_stamp, get_postdating_changes


def test_registry_entries_are_valid():
    commits = [change.commit for change in RESULT_CHANGES]
    # Commits are the entries' IDs
    assert len(commits) == len(set(commits))
    for change in RESULT_CHANGES:
        assert change.commit
        assert change.stage in CACHE_STAGES


def test_result_change_validation():
    with pytest.raises(ValueError):
        ResultChange(commit="abc", version="1.0.0", stage="not_a_stage", description="")


def test_stamp_predates_change():
    change = ResultChange(commit="0123abcd", version="1.0.0", stage="matrices", description="")
    assert not stamp_predates_change({"version": "1.0.0", "commit": None, "known_changes": ["0123abcd"]}, change)
    assert stamp_predates_change({"version": "1.0.0", "commit": None, "known_changes": ["4567cdef"]}, change)
    assert stamp_predates_change({"version": "1.0.0", "commit": None, "known_changes": []}, change)
    # Missing/unreadable stamps predate everything
    assert stamp_predates_change(None, change)
    assert stamp_predates_change({"version": "1.0.0", "commit": None}, change)
    assert stamp_predates_change({"version": "1.0.0", "commit": None, "known_changes": "0123abcd"}, change)


def test_code_stamp_knows_current_changes(monkeypatch):
    changes = [
        ResultChange(commit="0123abcd", version="1.0.0", stage="parse", description=""),
        ResultChange(commit="4567cdef", version="1.0.0", stage="spectra", description=""),
    ]
    monkeypatch.setattr(result_changes, "RESULT_CHANGES", changes)
    stamp = get_code_stamp()
    assert stamp["version"] == crawfish.__version__
    assert stamp["known_changes"] == ["0123abcd", "4567cdef"]
    # Code never flags caches it wrote itself
    assert get_postdating_changes(stamp) == []
    # Older code that only knew the first change
    assert get_postdating_changes({**stamp, "known_changes": ["0123abcd"]}) == [changes[1]]
