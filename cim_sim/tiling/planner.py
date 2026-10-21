# -*- coding: utf-8 -*-
"""
Tiling planner: maps a logical weight matrix onto a grid of independent physical crossbars.

For W of shape (L_r, L_c) and tiles of shape (t_r, t_c):

    grid = (ceil(L_r / t_r), ceil(L_c / t_c))
    tile (i, j) holds W[i*t_r : min((i+1)*t_r, L_r), j*t_c : min((j+1)*t_c, L_c)]
    tile (i, j) receives the input slice X[:, rows_i] and produces the partial sum P[i, j] = X_i @ W_ij
    logical output block j is Y_j = sum_i P[i, j]          (aggregation over the tile column j)

The planner is purely geometric. It knows nothing about device physics and is shared by every
tile executor backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import TilingConfig


@dataclass(frozen=True)
class TileSpec:
    """Geometry of one physical tile."""
    index: int                          # row-major tile id, 0 … n_tiles-1
    grid_pos: tuple[int, int]           # (tile_row i, tile_col j)
    row_range: tuple[int, int]          # logical input rows [start, stop)
    col_range: tuple[int, int]          # logical output columns [start, stop)
    physical_shape: tuple[int, int]     # shape of the physical array that is simulated
    nominal_shape: tuple[int, int]      # tile_shape of the plan

    @property
    def rows(self) -> slice:
        """Logical input rows routed to this tile."""
        return slice(*self.row_range)

    @property
    def cols(self) -> slice:
        """Logical output columns this tile contributes to."""
        return slice(*self.col_range)

    @property
    def valid_shape(self) -> tuple[int, int]:
        """Number of logical rows / columns actually mapped onto the tile."""
        return self.row_range[1] - self.row_range[0], self.col_range[1] - self.col_range[0]

    @property
    def active_cols(self) -> slice:
        """Physical columns whose ADC is active (the first valid_cols columns)."""
        return slice(0, self.valid_shape[1])

    @property
    def is_edge(self) -> bool:
        """True when the valid region is smaller than the nominal tile shape."""
        return self.valid_shape != self.nominal_shape

    @property
    def n_valid_cells(self) -> int:
        vr, vc = self.valid_shape
        return vr * vc

    @property
    def n_disabled_cells(self) -> int:
        """Physically present but unused cells (non-zero only for edge_policy="disabled")."""
        pr, pc = self.physical_shape
        return pr * pc - self.n_valid_cells


@dataclass(frozen=True)
class TilingPlan:
    """Complete mapping of one logical matrix onto physical tiles."""
    config: TilingConfig
    grid_shape: tuple[int, int]
    tiles: tuple[TileSpec, ...] = field(repr=False)

    # ---- basic properties ----
    @property
    def logical_shape(self) -> tuple[int, int]:
        return self.config.logical_shape

    @property
    def tile_shape(self) -> tuple[int, int]:
        return self.config.tile_shape

    @property
    def reduction_mode(self) -> str:
        return self.config.reduction_mode

    @property
    def edge_policy(self) -> str:
        return self.config.edge_policy

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    # ---- queries ----
    def tile(self, i: int, j: int) -> TileSpec:
        """Tile at grid position (i, j)."""
        gr, gc = self.grid_shape
        if not (0 <= i < gr and 0 <= j < gc):
            raise IndexError(f"tile ({i}, {j}) outside grid {self.grid_shape}")
        return self.tiles[i * gc + j]

    def row_block(self, i: int) -> tuple[TileSpec, ...]:
        """Tiles of grid row i; they all receive the same input slice X[:, rows_i]."""
        return tuple(self.tile(i, j) for j in range(self.grid_shape[1]))

    def col_block(self, j: int) -> tuple[TileSpec, ...]:
        """Tiles of grid column j; their partial sums are reduced into output block Y[:, cols_j]."""
        return tuple(self.tile(i, j) for i in range(self.grid_shape[0]))

    def physical_shapes(self) -> list[tuple[int, int]]:
        """Distinct physical array shapes in the plan (one entry for edge_policy="disabled")."""
        return sorted({t.physical_shape for t in self.tiles})

    @property
    def n_edge_tiles(self) -> int:
        return sum(t.is_edge for t in self.tiles)

    @property
    def n_physical_cells(self) -> int:
        return sum(t.physical_shape[0] * t.physical_shape[1] for t in self.tiles)

    @property
    def n_disabled_cells(self) -> int:
        return sum(t.n_disabled_cells for t in self.tiles)

    @property
    def cell_utilization(self) -> float:
        """Fraction of physical cells that hold a logical weight."""
        return self.logical_shape[0] * self.logical_shape[1] / self.n_physical_cells

    @property
    def adc_conversions_per_vector(self) -> int:
        """ADC conversions needed for one logical input vector (active columns of all tiles)."""
        return sum(t.valid_shape[1] for t in self.tiles)

    def describe(self) -> str:
        """Multi-line human-readable summary."""
        (lr, lc), (tr, tc), (gr, gc) = self.logical_shape, self.tile_shape, self.grid_shape
        lines = [
            f"logical matrix {lr} x {lc} -> physical tiles {tr} x {tc} "
            f"(edge policy '{self.edge_policy}', reduction '{self.reduction_mode}')",
            f"tile grid {gr} x {gc} = {self.n_tiles} tiles "
            f"({self.n_tiles - self.n_edge_tiles} full, {self.n_edge_tiles} edge)",
            f"physical cells {self.n_physical_cells}, mapped weights {lr * lc} "
            f"({self.cell_utilization:.1%} utilization, {self.n_disabled_cells} disabled cells)",
            f"distinct physical array shapes {self.physical_shapes()}",
            f"ADC conversions per input vector {self.adc_conversions_per_vector}",
        ]
        return "\n".join(lines)

    def layout(self, max_rows: int = 8, max_cols: int = 8) -> str:
        """Text map of the tile grid; each cell shows the valid shape of the tile (truncated for large grids)."""
        gr, gc = self.grid_shape
        ri = list(range(gr)) if gr <= max_rows else list(range(max_rows - 1)) + [gr - 1]
        ci = list(range(gc)) if gc <= max_cols else list(range(max_cols - 1)) + [gc - 1]
        cells = {(i, j): "{}x{}{}".format(*self.tile(i, j).valid_shape, "*" if self.tile(i, j).is_edge else "")
                 for i in ri for j in ci}
        w = max(len(s) for s in cells.values()) + 1
        out = []
        for a, i in enumerate(ri):
            if a > 0 and i != ri[a - 1] + 1:
                out.append("  " + " " * 5 + "...")
            row = []
            for b, j in enumerate(ci):
                if b > 0 and j != ci[b - 1] + 1:
                    row.append("...".rjust(w))
                row.append(cells[(i, j)].rjust(w))
            out.append(f"  r{i:<4d}" + "".join(row))
        out.append("  (valid rows x cols per tile; * = edge tile)")
        return "\n".join(out)

    def layout_table(self) -> list[dict]:
        """One record per tile: position, logical index ranges, valid and physical shapes."""
        return [dict(index=t.index, tile_row=t.grid_pos[0], tile_col=t.grid_pos[1],
                     row_start=t.row_range[0], row_stop=t.row_range[1],
                     col_start=t.col_range[0], col_stop=t.col_range[1],
                     valid_rows=t.valid_shape[0], valid_cols=t.valid_shape[1],
                     physical_rows=t.physical_shape[0], physical_cols=t.physical_shape[1],
                     edge=t.is_edge) for t in self.tiles]


def plan_tiling(logical_shape=None, tile_shape=None, reduction_mode: str = "digital",
                edge_policy: str = "disabled", disabled_level: int = 0, *,
                config: TilingConfig | None = None) -> TilingPlan:
    """
    Build the tiling plan.

    Either pass the individual arguments or a ready TilingConfig via `config=`.

    Args:
        logical_shape:  (logical_rows, logical_cols)
        tile_shape:     (tile_rows, tile_cols)
        reduction_mode: "digital" | "analog" | "hierarchical" (only "digital" can be executed)
        edge_policy:    "disabled" | "absent"
        disabled_level: weight level of disabled cells

    Returns:
        TilingPlan
    """
    if config is None:
        if logical_shape is None or tile_shape is None:
            raise ValueError("plan_tiling needs logical_shape and tile_shape, or config=TilingConfig(...)")
        config = TilingConfig(logical_shape, tile_shape, reduction_mode, edge_policy, disabled_level)
    (lr, lc), (tr, tc) = config.logical_shape, config.tile_shape
    gr, gc = -(-lr // tr), -(-lc // tc)
    tiles = []
    for i in range(gr):
        r0, r1 = i * tr, min((i + 1) * tr, lr)
        for j in range(gc):
            c0, c1 = j * tc, min((j + 1) * tc, lc)
            phys = (tr, tc) if config.edge_policy == "disabled" else (r1 - r0, c1 - c0)
            tiles.append(TileSpec(index=i * gc + j, grid_pos=(i, j), row_range=(r0, r1), col_range=(c0, c1),
                                  physical_shape=phys, nominal_shape=(tr, tc)))
    return TilingPlan(config=config, grid_shape=(gr, gc), tiles=tuple(tiles))
