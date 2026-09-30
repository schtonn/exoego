"""Compact physical-state-conditioned H2O exo-to-ego baseline.

Keep the public training classes lazy so geometry/rendering utilities do not
need to import PyTorch just to load a package submodule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset import H2OPhysicalClipDataset
    from .model import PhysicalEgoVideoPredictor

__all__ = ["H2OPhysicalClipDataset", "PhysicalEgoVideoPredictor"]


def __getattr__(name: str):
    if name == "H2OPhysicalClipDataset":
        from .dataset import H2OPhysicalClipDataset

        return H2OPhysicalClipDataset
    if name == "PhysicalEgoVideoPredictor":
        from .model import PhysicalEgoVideoPredictor

        return PhysicalEgoVideoPredictor
    raise AttributeError(name)
