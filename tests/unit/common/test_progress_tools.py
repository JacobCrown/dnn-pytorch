"""Tests for shared Rich progress helpers."""

from src.common.progress_tools import iterate_with_progress


def test_iterate_with_progress_accepts_total_for_generator() -> None:
    items = (i for i in range(3))

    result = list(iterate_with_progress(items, "prep", total=3))

    assert result == [0, 1, 2]
