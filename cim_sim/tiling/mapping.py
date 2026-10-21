# -*- coding: utf-8 -*-
"""
Operand mapping between the logical matrix and physical tiles.

    validate_operands  shape / range checks of the logical W and X
    tile_weights       weight slice of one tile, padded to the physical shape with disabled cells
    tile_inputs        input slice routed to one tile; disabled rows receive input code 0 (0 V)
    output_columns     logical output columns written by one tile

Only integer weight levels and input codes are handled here; conductances and voltages are
produced later by the physical modules (weights.program_arrays, inputs.inputs_to_voltage).
"""
from __future__ import annotations

import numpy as np

from .planner import TileSpec, TilingPlan


def validate_operands(W: np.ndarray, X: np.ndarray, plan: TilingPlan, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Check and normalise the logical operands.

    Args:
        W:    (logical_rows, logical_cols) integer weight levels in [0, 2^bits - 1]
        X:    (batch, logical_rows) integer input codes in [0, 2^bits - 1]
        plan: TilingPlan
        bits: weight / DAC resolution

    Returns:
        (W, X) as float64 arrays
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    lr, lc = plan.logical_shape
    if W.shape != (lr, lc):
        raise ValueError(f"W has shape {W.shape}, but the plan expects the logical shape {(lr, lc)}")
    if X.ndim != 2 or X.shape[1] != lr:
        raise ValueError(f"X must have shape (batch, {lr}); got {X.shape}. "
                         f"Use X[None, :] for a single input vector")
    if X.shape[0] < 1:
        raise ValueError("X must contain at least one input vector")
    L = 2 ** bits - 1
    for name, A in (("W", W), ("X", X)):
        if not np.all(np.isfinite(A)) or np.any(A != np.rint(A)) or A.min() < 0 or A.max() > L:
            raise ValueError(f"{name} must contain integers in [0, {L}] for {bits}-bit operands")
    if plan.config.disabled_level > L:
        raise ValueError(f"disabled_level {plan.config.disabled_level} exceeds the maximum level {L}")
    return W, X


def tile_weights(W: np.ndarray, spec: TileSpec, disabled_level: int = 0) -> np.ndarray:
    """
    Weight levels to program into one physical tile.

    Args:
        W:              (logical_rows, logical_cols) logical weight levels
        spec:           TileSpec
        disabled_level: level written into cells outside the valid region

    Returns:
        (physical_rows, physical_cols) float64; the valid region W[rows, cols] sits in the top-left corner
    """
    vr, vc = spec.valid_shape
    Wt = np.full(spec.physical_shape, float(disabled_level))
    Wt[:vr, :vc] = W[spec.rows, spec.cols]
    return Wt


def tile_inputs(X: np.ndarray, spec: TileSpec) -> np.ndarray:
    """
    Input codes routed to one physical tile.

    Args:
        X:    (batch, logical_rows) logical input codes
        spec: TileSpec

    Returns:
        (batch, physical_rows) float64; rows beyond the valid region are 0 (driven at 0 V)
    """
    vr = spec.valid_shape[0]
    Xt = np.zeros((X.shape[0], spec.physical_shape[0]))
    Xt[:, :vr] = X[:, spec.rows]
    return Xt


def output_columns(spec: TileSpec) -> slice:
    """Logical output columns written by the active ADC columns of this tile."""
    return spec.cols
