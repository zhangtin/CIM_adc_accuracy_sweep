# -*- coding: utf-8 -*-
"""
Tile execution: runs the existing physical models on every physical tile and reduces the results.

    TileContext            physical parameters shared by all tiles of one run
    TileJob                backend-independent description of one tile's work
    TileResult             analog / ADC outputs of one tile (active columns only)
    TileExecutor           backend interface (CPU now; GPU backends implement the same interface)
    CPUTileExecutor        reference backend built on weights / inputs / crossbar / read_noise / adc
    tile_wire_resistance   per-segment wire resistance of the physical tiles
    tile_adc_window        ADC full scale of a physical tile shape
    simulate_tiled_crossbar  public entry point: plan + W + X -> reconstructed logical outputs

Per tile, the CPU executor performs exactly the single-array signal chain:

    W slice --LUT + write noise (once)--> G --DAC/read_array (IR drop on the PHYSICAL tile)--> I_ir
         --add_read_noise (per read)--> I_read --quantize (per-tile ADC)--> I_q

The executor never sees the logical matrix; it only receives the tile's own operands.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

from ..adc import adc_window, quantize
from ..calibration import stat_calibration
from ..config import CalibParams, DeviceParams, InputParams, IRParams, NoiseParams
from ..crossbar import ideal_mvm, read_array
from ..inputs import inputs_to_voltage
from ..ir_drop import calibrate_r_wire
from ..read_noise import add_read_noise
from ..weights import conductance_lut, program_arrays
from .config import TileADCParams
from .mapping import tile_inputs, tile_weights, validate_operands
from .planner import TileSpec, TilingPlan
from .reduction import make_reducer
from .streams import TileStreamProvider, TileStreams


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TileContext:
    """Physical parameters shared by every tile of one run."""
    bits: int                      # weight / DAC resolution
    sigma: float                   # relative write error
    device: DeviceParams
    ir: IRParams                   # solver tolerance / iteration limit (r_wire is resolved separately)
    noise: NoiseParams
    lut: np.ndarray                # ideal weight LUT (2^bits,)
    r_wire: float                  # wire resistance per segment of every physical tile (Ohm); 0 = ideal
    adc_bits: int | None           # None = quantization bypassed (verification only)


@dataclass
class TileJob:
    """Work for one physical tile, expressed in integer levels / codes only."""
    spec: TileSpec
    W_phys: np.ndarray             # (pr, pc) weight levels incl. disabled cells
    X_phys: np.ndarray             # (B, pr) input codes; disabled rows are 0
    sample_ids: np.ndarray         # (B,) global sample indices (keys of the read-noise streams)
    adc_window: tuple[float, float] | None


@dataclass
class TileResult:
    """Outputs of one physical tile; current arrays cover the ACTIVE columns only (B, valid_cols)."""
    spec: TileSpec
    sum_x: np.ndarray              # (B,) sum of this tile's input codes (digital baseline compensation)
    I_prog: np.ndarray             # ideal interconnect, no read noise (A)
    I_ir: np.ndarray               # with IR drop (A)
    I_read: np.ndarray             # IR drop + read noise = ADC input (A)
    I_q: np.ndarray                # ADC output reconstructed as current (A)
    noise_sd: np.ndarray           # read-noise standard deviation (A)
    sat_lo: np.ndarray             # ADC underflow mask
    sat_hi: np.ndarray             # ADC overflow mask
    E_array: np.ndarray            # (B,) energy delivered by this tile's row drivers (J)
    iters: int                     # IR solver iterations (0 without IR drop)
    adc_window: tuple[float, float] | None
    G: np.ndarray | None = None    # (pr, pc) programmed conductances (kept on request)

    def currents(self) -> dict:
        """Signal views in the order of reduction.VIEWS."""
        return dict(prog=self.I_prog, ir=self.I_ir, analog=self.I_read, adc=self.I_q)


@dataclass
class TiledResult:
    """Result of simulate_tiled_crossbar()."""
    plan: TilingPlan
    Y: np.ndarray                  # (B, logical_cols) reconstructed digital MAC (hardware output)
    Y_analog: np.ndarray           # (B, logical_cols) before ADC quantization
    Y_ir: np.ndarray               # (B, logical_cols) write noise + IR drop, no read noise
    Y_prog: np.ndarray             # (B, logical_cols) write noise only
    partial_sums: np.ndarray       # (grid_rows, B, logical_cols) digital partial sums of Y
    E_array: np.ndarray            # (B,) array read energy summed over all tiles (J)
    adc_conversions: int           # ADC conversions per input vector
    sat_rate: float                # fraction of ADC conversions that clipped
    r_wire: float                  # wire resistance per segment (Ohm)
    adc_windows: dict              # {physical_shape: (lo, hi) or None}
    ir_iters: np.ndarray           # (n_tiles,) IR solver iterations per tile (plan order)
    tile_results: list | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class TileExecutor(ABC):
    """
    Backend interface. An implementation receives independent TileJobs and yields one TileResult per
    job (any order). It must use only `ctx` and `streams` for physics and randomness, so that a GPU
    backend can batch many jobs while the planner, mapping and reduction stay unchanged.
    """
    name = "abstract"

    @abstractmethod
    def execute(self, jobs: Sequence[TileJob], ctx: TileContext,
                streams: TileStreamProvider) -> Iterator[TileResult]:
        ...


class CPUTileExecutor(TileExecutor):
    """
    Reference NumPy/C backend; tiles are executed one after another.

    Args:
        keep_conductance: store each tile's programmed conductance matrix in TileResult.G
        batch_chunk:      maximum number of input vectors per read_array() call (None = all). Random
                          samples do not depend on it; with IR drop enabled the iterative solution can
                          differ within the solver tolerance (IRParams.tol) because convergence is
                          checked jointly over the vectors of one call.
    """
    name = "cpu"

    def __init__(self, keep_conductance: bool = False, batch_chunk: int | None = None):
        if batch_chunk is not None and batch_chunk < 1:
            raise ValueError("batch_chunk must be a positive integer or None")
        self.keep_conductance = keep_conductance
        self.batch_chunk = batch_chunk

    def execute(self, jobs, ctx, streams):
        for job in jobs:
            yield self.execute_one(job, ctx, streams)

    def execute_one(self, job: TileJob, ctx: TileContext, streams: TileStreamProvider) -> TileResult:
        spec = job.spec
        pr, pc = spec.physical_shape
        dev = ctx.device
        # (1) Programming: static write noise, drawn once for this physical tile.
        G = program_arrays(streams.programming(spec), 1, pr, pc, ctx.lut, ctx.sigma, W=job.W_phys[None]).G[0]
        # (2) DAC
        V = inputs_to_voltage(job.X_phys, ctx.bits, dev.v_read)                     # (B, pr)
        B = V.shape[0]
        # (3) Analog read on the physical tile (IR drop solved for the tile's own wires).
        I_ir = np.empty((B, pc))
        E = np.empty(B)
        sd = np.zeros((B, pc))
        iters = 0
        step = B if self.batch_chunk is None else self.batch_chunk
        for s in range(0, B, step):
            ro = read_array(G, V[s:s + step], ctx.r_wire, t_read=dev.t_read, ir=ctx.ir, noise=ctx.noise)
            I_ir[s:s + step], E[s:s + step] = ro.I, ro.E
            if ro.noise_sd is not None:
                sd[s:s + step] = ro.noise_sd
            iters = max(iters, ro.iters)
        I_prog = ideal_mvm(V, G)
        # (4) Dynamic read noise: one stream per (tile, input vector).
        if ctx.noise.enabled:
            I_read = np.empty_like(I_ir)
            for k in range(B):
                I_read[k] = add_read_noise(streams.read_noise(spec, int(job.sample_ids[k])), I_ir[k], sd[k])
        else:
            I_read = I_ir.copy()
        # (5) Per-tile ADC on the active columns only (ADCs of disabled columns are off).
        act = spec.active_cols
        if ctx.adc_bits is None:
            I_q = I_read[:, act].copy()
            sat_lo = sat_hi = np.zeros(I_q.shape, dtype=bool)
        else:
            I_q, sat_lo, sat_hi = quantize(I_read[:, act], *job.adc_window, ctx.adc_bits)
        return TileResult(spec=spec, sum_x=job.X_phys.sum(axis=1), I_prog=I_prog[:, act], I_ir=I_ir[:, act],
                          I_read=I_read[:, act], I_q=I_q, noise_sd=sd[:, act], sat_lo=sat_lo, sat_hi=sat_hi,
                          E_array=E, iters=iters, adc_window=job.adc_window,
                          G=G if self.keep_conductance else None)


# ---------------------------------------------------------------------------
# Hardware-level quantities shared by all tiles
# ---------------------------------------------------------------------------
def tile_wire_resistance(tile_shape: tuple[int, int], ir: IRParams, device: DeviceParams, seed: int) -> float:
    """
    Wire resistance per segment of the physical tiles (Ohm).

    The value is a property of the interconnect technology and is therefore shared by all tiles,
    including smaller boundary arrays under edge_policy="absent".

        ir.r_wire = 0   ideal interconnect
        ir.r_wire > 0   fixed value
        ir.r_wire < 0   calibrated with ir_drop.calibrate_r_wire for the square tile size N, i.e. the
                        same r_eff(N) a stand-alone N x N array uses. Defined for square tiles only.
    """
    if ir.r_wire == 0:
        return 0.0
    if ir.r_wire > 0:
        return float(ir.r_wire)
    tr, tc = tile_shape
    if tr != tc:
        raise ValueError(f"calibrated wire resistance (IRParams.r_wire < 0) is defined for square arrays only; "
                         f"tile shape {tile_shape} is rectangular. Pass IRParams(r_wire=<ohm per segment>) "
                         f"or IRParams(r_wire=0).")
    return float(calibrate_r_wire(tr, ir, device, seed)["r_eff_ohm"])


def tile_adc_window(physical_shape: tuple[int, int], *, adc: TileADCParams, ctx: TileContext,
                    calib: CalibParams, inputs: InputParams, rng: np.random.Generator | None,
                    seed: int) -> tuple[float, float] | None:
    """
    ADC full scale of one physical tile shape (the ADC is part of the tile hardware, so it does not
    depend on how many rows of a boundary tile carry logical weights).

    Args:
        physical_shape: (rows, cols) of the physical array
        adc:            TileADCParams
        ctx:            TileContext (bits, sigma, device, ir, noise, lut, r_wire)
        calib:          CalibParams (stat_k, n_cal_arrays, ...)
        inputs:         InputParams of the design-time input distribution
        rng:            calibration-array stream (fs_mode="stat" only)
        seed:           global seed (probe streams of large-array calibration)

    Returns:
        (lo, hi) in A, or None when quantization is bypassed
    """
    if adc.bits is None:
        return None
    rows, cols = physical_shape
    stat = None
    if adc.fs_mode == "stat":
        cal = stat_calibration(rng, ctx.bits, ctx.sigma, ctx.r_wire, rows, [adc.calib_dist], device=ctx.device,
                               inputs=inputs, ir=ctx.ir, noise=ctx.noise, calib=calib, seed=seed, lut=ctx.lut,
                               n_cols=cols)
        stat = cal[adc.calib_dist]
    return adc_window(adc.fs_mode, rows, v_read=ctx.device.v_read, g_max=ctx.device.g_max, stat=stat,
                      stat_k=calib.stat_k)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_jobs(W: np.ndarray, X: np.ndarray, plan: TilingPlan, adc_windows: dict,
               sample_offset: int = 0, order: Iterable[int] | None = None) -> list[TileJob]:
    """Map the logical operands onto per-tile jobs (in plan order unless `order` lists tile indices)."""
    ids = np.arange(X.shape[0]) + int(sample_offset)
    idx = range(plan.n_tiles) if order is None else list(order)
    if order is not None and sorted(idx) != list(range(plan.n_tiles)):
        raise ValueError("order must be a permutation of all tile indices")
    jobs = []
    for t in idx:
        spec = plan.tiles[t]
        jobs.append(TileJob(spec=spec, W_phys=tile_weights(W, spec, plan.config.disabled_level),
                            X_phys=tile_inputs(X, spec), sample_ids=ids,
                            adc_window=adc_windows[spec.physical_shape]))
    return jobs


def simulate_tiled_crossbar(W: np.ndarray, X: np.ndarray, plan: TilingPlan, *, bits: int, sigma: float,
                            device: DeviceParams = DeviceParams(), ir: IRParams = IRParams(),
                            noise: NoiseParams = NoiseParams(), adc: TileADCParams = TileADCParams(),
                            calib: CalibParams = CalibParams(), inputs: InputParams = InputParams(),
                            seed: int = 20260929, lut: np.ndarray | None = None,
                            executor: TileExecutor | None = None, streams: TileStreamProvider | None = None,
                            sample_offset: int = 0, tile_order: Iterable[int] | None = None,
                            keep_tile_results: bool = False) -> TiledResult:
    """
    Simulate W (logical) applied to X on the physical tiles described by `plan`.

    Args:
        W:            (logical_rows, logical_cols) integer weight levels in [0, 2^bits - 1]
        X:            (batch, logical_rows) integer input codes in [0, 2^bits - 1]
        plan:         TilingPlan from plan_tiling()
        bits:         weight / DAC resolution b
        sigma:        relative write error of every device
        device:       DeviceParams (g_min, g_max, v_read, t_read)
        ir:           IRParams; r_wire semantics as in tile_wire_resistance()
        noise:        NoiseParams (dynamic read noise)
        adc:          TileADCParams (per-tile ADC)
        calib:        CalibParams for the "stat" full scale
        inputs:       InputParams of the design-time input distribution (stat calibration only)
        seed:         global seed of the default TileStreams and of r_wire calibration
        lut:          ideal weight LUT; None = linear conductance_lut(bits, g_min, g_max)
        executor:     tile backend; None = CPUTileExecutor()
        streams:      random-stream provider; None = TileStreams(seed, bits, sigma)
        sample_offset: global index of X[0]; lets a long input stream be processed in several calls with
                      identical read-noise samples
        tile_order:   optional permutation of tile indices (execution order; does not change results)
        keep_tile_results: keep every TileResult in the returned object

    Returns:
        TiledResult
    """
    W, X = validate_operands(W, X, plan, bits)
    batch = X.shape[0]
    # Fail fast on reserved reduction modes before any physics runs.
    reducer = make_reducer(plan, batch, bits=bits, g_min=device.g_min, g_max=device.g_max, v_read=device.v_read)
    lut = conductance_lut(bits, device.g_min, device.g_max) if lut is None else np.asarray(lut, dtype=np.float64)
    if lut.shape != (2 ** bits,):
        raise ValueError(f"lut must have {2 ** bits} entries for {bits}-bit weights, got shape {lut.shape}")
    streams = TileStreams(seed, bits, sigma) if streams is None else streams
    executor = CPUTileExecutor() if executor is None else executor

    r_wire = tile_wire_resistance(plan.tile_shape, ir, device, seed)
    ctx = TileContext(bits=bits, sigma=sigma, device=device, ir=ir, noise=noise, lut=lut, r_wire=r_wire,
                      adc_bits=adc.bits)
    adc_windows = {shape: tile_adc_window(shape, adc=adc, ctx=ctx, calib=calib, inputs=inputs,
                                          rng=streams.calibration(shape) if (adc.bits and adc.fs_mode == "stat")
                                          else None, seed=seed)
                   for shape in plan.physical_shapes()}

    jobs = build_jobs(W, X, plan, adc_windows, sample_offset, tile_order)
    kept, E, n_sat, iters = [], np.zeros(batch), 0, np.zeros(plan.n_tiles, dtype=int)
    for res in executor.execute(jobs, ctx, streams):
        reducer.add(res)
        E += res.E_array
        n_sat += int((res.sat_lo | res.sat_hi).sum())
        iters[res.spec.index] = res.iters
        if keep_tile_results:
            kept.append(res)
    red = reducer.finalize()
    kept.sort(key=lambda r: r.spec.index)
    conv = plan.adc_conversions_per_vector
    return TiledResult(plan=plan, Y=red["Y"]["adc"], Y_analog=red["Y"]["analog"], Y_ir=red["Y"]["ir"],
                       Y_prog=red["Y"]["prog"], partial_sums=red["partial_sums"]["adc"], E_array=E,
                       adc_conversions=conv, sat_rate=n_sat / (conv * batch), r_wire=r_wire,
                       adc_windows=adc_windows, ir_iters=iters, tile_results=kept if keep_tile_results else None)
