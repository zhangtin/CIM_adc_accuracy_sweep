# -*- coding: utf-8 -*-
"""
Configuration of physical array tiling.

    TilingConfig    logical matrix shape, physical tile shape, reduction mode, edge policy
    TileADCParams   per-tile ADC resolution and full-scale strategy

Terminology
    logical matrix   the weight matrix of the workload, e.g. one 512 x 512 layer
    physical tile    one independent crossbar array (own word/bit lines, drivers, TIAs, ADCs),
                     e.g. 16 x 16; the logical matrix is mapped onto a grid of such arrays
"""
from __future__ import annotations

from dataclasses import dataclass

REDUCTION_MODES = ("digital", "analog", "hierarchical")
"""Known partial-sum reduction modes. Only "digital" is implemented (see reduction.py)."""

IMPLEMENTED_REDUCTION_MODES = ("digital",)

EDGE_POLICIES = ("disabled", "absent")
"""
How tiles at the matrix boundary are modeled when the logical shape is not a multiple of the tile shape.

    disabled  Every tile is a full physical array of tile_shape (uniform hardware). Cells outside the
              valid region physically exist: they are programmed to `disabled_level` (default 0, i.e.
              g_min, the highest-resistance state), unused rows are driven at 0 V and the ADCs of unused
              columns are switched off. These cells still load the word/bit lines in the IR-drop
              solution and add column conductance to the read-noise model, as they would in silicon.
    absent    Boundary tiles are physically smaller arrays of exactly the valid shape (e.g. custom macros
              at the edge). Cells outside the valid region do not exist at all.
"""

FS_MODES = ("max", "stat")


def _shape2(name: str, value) -> tuple[int, int]:
    try:
        r, c = value
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a pair (rows, cols), got {value!r}") from None
    for v in (r, c):
        if isinstance(v, bool) or int(v) != v:
            raise ValueError(f"{name} must contain integers, got {value!r}")
    r, c = int(r), int(c)
    if r < 1 or c < 1:
        raise ValueError(f"{name} must be positive, got {(r, c)}")
    return r, c


@dataclass(frozen=True)
class TilingConfig:
    """
    Args:
        logical_shape:  (logical_rows, logical_cols) of the weight matrix W; rows = inputs, cols = outputs
        tile_shape:     (tile_rows, tile_cols) of one physical crossbar
        reduction_mode: "digital" (implemented); "analog" / "hierarchical" are reserved and raise
                        NotImplementedError when executed
        edge_policy:    "disabled" (default) or "absent", see EDGE_POLICIES
        disabled_level: integer weight level programmed into disabled cells (edge_policy="disabled")
    """
    logical_shape: tuple[int, int]
    tile_shape: tuple[int, int]
    reduction_mode: str = "digital"
    edge_policy: str = "disabled"
    disabled_level: int = 0

    def __post_init__(self):
        object.__setattr__(self, "logical_shape", _shape2("logical_shape", self.logical_shape))
        object.__setattr__(self, "tile_shape", _shape2("tile_shape", self.tile_shape))
        if self.reduction_mode not in REDUCTION_MODES:
            raise ValueError(f"unknown reduction_mode {self.reduction_mode!r}; expected one of {REDUCTION_MODES}")
        if self.edge_policy not in EDGE_POLICIES:
            raise ValueError(f"unknown edge_policy {self.edge_policy!r}; expected one of {EDGE_POLICIES}")
        if isinstance(self.disabled_level, bool) or int(self.disabled_level) != self.disabled_level \
                or self.disabled_level < 0:
            raise ValueError(f"disabled_level must be a non-negative integer, got {self.disabled_level!r}")


@dataclass(frozen=True)
class TileADCParams:
    """
    Per-tile ADC (one ADC per active physical column).

    Args:
        bits:       ADC resolution B. None bypasses quantization (ideal reference readout; use only for
                    verification, it does not correspond to hardware)
        fs_mode:    "max" -> [0, rows * v_read * g_max] of the physical tile;
                    "stat" -> mu +/- k*sigma calibrated at design time on independent arrays of the
                    physical tile shape with all rows active (CalibParams.stat_k, n_cal_arrays)
        calib_dist: input distribution assumed at design time for the "stat" calibration
    """
    bits: int | None = 8
    fs_mode: str = "stat"
    calib_dist: str = "uniform"

    def __post_init__(self):
        if self.bits is not None and (isinstance(self.bits, bool) or int(self.bits) != self.bits or self.bits < 1):
            raise ValueError(f"ADC bits must be a positive integer or None, got {self.bits!r}")
        if self.fs_mode not in FS_MODES:
            raise ValueError(f"unknown fs_mode {self.fs_mode!r}; expected one of {FS_MODES}")
