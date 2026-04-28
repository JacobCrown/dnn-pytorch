"""OmegaConf resolvers shared by Hydra entrypoints.

Register once per process before config is composed (import before @hydra.main).
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

_registered = False


def register_hydra_resolvers() -> None:
    """Register path basename resolver for Hydra/OmegaConf interpolations."""
    global _registered
    if _registered:
        return

    def basename(path_like: object) -> str:
        """Last path segment of a string path (dataset folder name etc.)."""
        return Path(str(path_like)).name

    OmegaConf.register_new_resolver("basename", basename, replace=True)
    _registered = True
