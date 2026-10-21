# -*- coding: utf-8 -*-
"""
Partial-sum reduction strategies.

    digital       (implemented) every tile digitizes its own column currents; each tile's codes are
                  converted to MAC units with that tile's own input sum, and the partial sums are
                  added in the digital domain.
    analog        (reserved) partial sums merged in the analog domain before a shared ADC.
    hierarchical  (reserved) multi-level mix of analog and digital reduction.

Baseline handling in digital mode. A tile column current is

    I_j = k * sum_i x_i * (g_min + w_ij * dG),       k = v_read / (2^b - 1)

so readout.current_to_mac() removes the baseline g_min * sum_i x_i with the input sum of THIS tile's
rows only. Disabled rows carry x_i = 0, so they add nothing to the baseline. Summing the
baseline-free partial sums over tiles therefore compensates every logical input exactly once.

The reserved modes raise NotImplementedError. They must not fall back to adding currents before
the ADC: a physical analog merge involves summing amplifiers or charge sharing, loading, additional
noise and a different full scale, none of which is modeled yet.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..readout import current_to_mac
from .config import IMPLEMENTED_REDUCTION_MODES, REDUCTION_MODES
from .planner import TilingPlan

VIEWS = ("prog", "ir", "analog", "adc")
"""
Signal views carried through the reduction (all in MAC units):
    prog    programmed conductances, ideal interconnect, no read noise, no ADC
    ir      programmed conductances with IR drop
    analog  IR drop + read noise (the ADC input)
    adc     after ADC quantization (the hardware output)
"""


class Reducer(ABC):
    mode: str = ""

    @abstractmethod
    def add(self, result) -> None:
        """Accumulate one TileResult."""

    @abstractmethod
    def finalize(self) -> dict:
        """Return {"Y": {view: (batch, logical_cols)}, "partial_sums": {view: (grid_rows, batch, logical_cols)}}."""


class DigitalReducer(Reducer):
    """
    Digital-domain reduction.

    Partial sums are stored per tile row and summed in a fixed order at the end, so the result does
    not depend on the order in which tiles are executed.

    Args:
        plan:   TilingPlan
        batch:  number of input vectors
        bits:   weight / DAC resolution
        g_min, g_max, v_read: device parameters used for the current-to-MAC conversion
    """
    mode = "digital"

    def __init__(self, plan: TilingPlan, batch: int, *, bits: int, g_min: float, g_max: float, v_read: float):
        self.plan, self.batch = plan, batch
        self._readout = dict(g_min=g_min, g_max=g_max, v_read=v_read)
        self.bits = bits
        gr, gc = plan.grid_shape
        self.partials = {v: np.zeros((gr, batch, plan.logical_shape[1])) for v in VIEWS}
        self._seen = np.zeros((gr, gc), dtype=bool)

    def add(self, result) -> None:
        spec = result.spec
        i, j = spec.grid_pos
        if self._seen[i, j]:
            raise RuntimeError(f"tile {spec.grid_pos} was reduced twice")
        self._seen[i, j] = True
        for view, I in result.currents().items():
            # Per-tile digital reconstruction: each tile removes the g_min baseline of its own rows.
            Y = current_to_mac(I, result.sum_x, self.bits, **self._readout)
            self.partials[view][i, :, spec.cols] = Y

    def finalize(self) -> dict:
        if not self._seen.all():
            missing = [tuple(int(v) for v in p) for p in np.argwhere(~self._seen)]
            raise RuntimeError(f"{len(missing)} tiles were never reduced, e.g. {missing[:5]}")
        Y = {v: p.sum(axis=0) for v, p in self.partials.items()}
        return dict(Y=Y, partial_sums=self.partials)


class _ReservedReducer(Reducer):
    _reason = ""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"reduction_mode '{self.mode}' is reserved but not implemented: {self._reason} "
            f"Implemented modes: {IMPLEMENTED_REDUCTION_MODES}.")

    def add(self, result) -> None:  # pragma: no cover
        raise NotImplementedError

    def finalize(self) -> dict:  # pragma: no cover
        raise NotImplementedError


class AnalogReducer(_ReservedReducer):
    mode = "analog"
    _reason = ("merging partial sums before the ADC requires a model of the summing circuit "
               "(current or charge domain), its loading, noise and the resulting ADC full scale.")


class HierarchicalReducer(_ReservedReducer):
    mode = "hierarchical"
    _reason = ("multi-level analog/digital reduction requires a grouping strategy plus the analog "
               "merge model of the 'analog' mode.")


REDUCERS = {"digital": DigitalReducer, "analog": AnalogReducer, "hierarchical": HierarchicalReducer}
assert set(REDUCERS) == set(REDUCTION_MODES)


def make_reducer(plan: TilingPlan, batch: int, **readout) -> Reducer:
    """
    Instantiate the reducer selected by plan.reduction_mode.

    Raises NotImplementedError for reserved modes (never falls back to "digital").
    """
    try:
        cls = REDUCERS[plan.reduction_mode]
    except KeyError:
        raise ValueError(f"unknown reduction_mode {plan.reduction_mode!r}") from None
    return cls(plan, batch, **readout)
