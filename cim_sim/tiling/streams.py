# -*- coding: utf-8 -*-
"""
Random-stream management for tiled simulations (kept separate from cim_sim.rng, which is unchanged).

Every random draw of a tiled run is tied to a physical object, never to the execution order:

    programming(spec)            static write noise of one physical tile, drawn once when it is programmed
    read_noise(spec, sample)     dynamic read noise of one tile for one input vector (global sample index)
    calibration(physical_shape)  independent design-time arrays for the ADC "stat" calibration

TileStreams derives each stream from a SeedSequence key

    [seed, TILING_STREAM_TAG, purpose, bits, sigma_ppm, physical_rows, physical_cols, tile_row, tile_col, (sample)]

so results do not depend on the order in which tiles are executed, on how the batch is chunked, or
on the backend's scheduling. A future GPU backend can implement the same TileStreamProvider
protocol with a counter-based generator (e.g. Philox keyed by the same tuple); bit-identical
results across different generators are not expected.

SequentialStreams hands out caller-supplied generators in call order. It exists to reproduce the
random sequence of the single-array path (simulate_crossbar) in consistency tests and should not
be used for multi-tile production runs.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..rng import sigma_key
from .planner import TileSpec

TILING_STREAM_TAG = 0x71E5          # separates tiling streams from every stream defined in cim_sim.rng
PURPOSE_PROGRAM, PURPOSE_READ, PURPOSE_CALIB, PURPOSE_WORKLOAD = 1, 2, 3, 4


class TileStreamProvider(Protocol):
    def programming(self, spec: TileSpec) -> np.random.Generator: ...

    def read_noise(self, spec: TileSpec, sample_index: int) -> np.random.Generator: ...

    def calibration(self, physical_shape: tuple[int, int]) -> np.random.Generator: ...


class TileStreams:
    """
    Default order-independent streams.

    Args:
        seed:  global seed
        bits:  weight / DAC resolution (part of the key, as in cim_sim.rng)
        sigma: relative write error (part of the key)
    """

    def __init__(self, seed: int, bits: int, sigma: float):
        self.seed, self.bits, self.sigma = int(seed), int(bits), float(sigma)
        self._base = [self.seed, TILING_STREAM_TAG]

    def _key(self, purpose: int, *rest: int) -> list[int]:
        return self._base + [purpose, self.bits, sigma_key(self.sigma), *map(int, rest)]

    def programming(self, spec: TileSpec) -> np.random.Generator:
        return np.random.default_rng(self._key(PURPOSE_PROGRAM, *spec.physical_shape, *spec.grid_pos))

    def read_noise(self, spec: TileSpec, sample_index: int) -> np.random.Generator:
        return np.random.default_rng(self._key(PURPOSE_READ, *spec.physical_shape, *spec.grid_pos, sample_index))

    def calibration(self, physical_shape: tuple[int, int]) -> np.random.Generator:
        return np.random.default_rng(self._key(PURPOSE_CALIB, *physical_shape))


class SequentialStreams:
    """
    Caller-supplied generators, consumed in call order (order-dependent; for consistency tests).

    Args:
        programming: generator used for every tile's write noise
        read_noise:  generator used for every read-noise draw
        calibration: generator used for the stat calibration (optional)
    """

    def __init__(self, programming: np.random.Generator, read_noise: np.random.Generator,
                 calibration: np.random.Generator | None = None):
        self._prog, self._read, self._cal = programming, read_noise, calibration

    def programming(self, spec: TileSpec) -> np.random.Generator:
        return self._prog

    def read_noise(self, spec: TileSpec, sample_index: int) -> np.random.Generator:
        return self._read

    def calibration(self, physical_shape: tuple[int, int]) -> np.random.Generator:
        if self._cal is None:
            raise ValueError("SequentialStreams was created without a calibration generator")
        return self._cal


def workload_rng(seed: int, bits: int, logical_shape: tuple[int, int]) -> np.random.Generator:
    """
    Generator for the logical workload (W and X). It depends only on the logical problem, so the
    same weights and inputs are used when different tile shapes are compared.
    """
    return np.random.default_rng([int(seed), TILING_STREAM_TAG, PURPOSE_WORKLOAD, int(bits), *map(int, logical_shape)])
