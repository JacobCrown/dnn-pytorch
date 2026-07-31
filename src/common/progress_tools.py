"""Rich progress iterators (elapsed + ETA), shared across training entrypoints."""

from __future__ import annotations

from collections.abc import Iterable, Iterator


def iterate_with_progress[T](
    iterable: Iterable[T],
    description: str,
    *,
    transient: bool = True,
) -> Iterator[T]:
    """Iterate with a Rich bar showing elapsed time and ETA (stderr, aligns with logging)."""
    from rich.console import Console
    from rich.progress import (BarColumn, Progress, TaskProgressColumn,
                               TextColumn, TimeElapsedColumn,
                               TimeRemainingColumn)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),
        transient=transient,
    )
    with progress:
        yield from progress.track(iterable, description=description)
