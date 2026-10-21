# -*- coding: utf-8 -*-
"""
Tests of the physical array tiling layer.

    A  ideal correctness: Y_tiled == X @ W with every non-ideality off and the ADC bypassed
    B  physical consistency with the single-array path; per-tile physics at the physical tile size
    C  edge cases: non-divisible shapes, rectangular tiles, batches, invalid input, reserved modes,
       padding, output indexing, random-stream invariances

Run:  python -m unittest tests.test_tiling -v      (from the repository root)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cim_sim import CalibParams, DeviceParams, InputParams, IRParams, NoiseParams  # noqa: E402
from cim_sim.adc import max_window, quantize  # noqa: E402
from cim_sim.calibration import stat_calibration  # noqa: E402
from cim_sim.crossbar import ideal_mvm, ideal_read_energy, read_array, read_array_ir, simulate_crossbar  # noqa: E402
from cim_sim.inputs import inputs_to_voltage, sample_inputs  # noqa: E402
from cim_sim.ir_drop import (IRSolver, RectIRSolver, calibrate_r_wire, exact_solve_rect,  # noqa: E402
                             make_ir_solver)
from cim_sim.read_noise import cell_current_sq_ideal, read_noise_std  # noqa: E402
from cim_sim.readout import current_to_mac  # noqa: E402
from cim_sim.rng import point_streams  # noqa: E402
from cim_sim.tiling import (CPUTileExecutor, SequentialStreams, TileADCParams, TileStreams,  # noqa: E402
                            TilingConfig, plan_tiling, simulate_tiled_crossbar, tile_inputs, tile_weights,
                            workload_rng)
from cim_sim.weights import sample_weight_levels  # noqa: E402

BITS = 4
DEV = DeviceParams()
IDEAL = dict(sigma=0.0, ir=IRParams(r_wire=0.0), noise=NoiseParams(enabled=False), adc=TileADCParams(bits=None))


def workload(shape, batch=8, bits=BITS, seed=7, dist="uniform"):
    rng = workload_rng(seed, bits, shape)
    W = sample_weight_levels(rng, shape, 2 ** bits)
    X = sample_inputs(rng, dist, bits, (batch, shape[0]))
    return W, X


# ---------------------------------------------------------------------------
# A. Ideal correctness
# ---------------------------------------------------------------------------
class IdealCorrectnessTest(unittest.TestCase):
    SHAPES = [(16, 16), (32, 32), (64, 64), (32, 64), (35, 50), (512, 512)]

    def _check(self, shape, tile, edge_policy="disabled", batch=8):
        W, X = workload(shape, batch)
        plan = plan_tiling(shape, tile, edge_policy=edge_policy)
        r = simulate_tiled_crossbar(W, X, plan, bits=BITS, **IDEAL)
        ref = X @ W
        for name in ("Y", "Y_analog", "Y_ir", "Y_prog"):
            np.testing.assert_allclose(getattr(r, name), ref, rtol=1e-12, atol=1e-8, err_msg=f"{shape} {name}")
        # every digital partial sum equals the exact block product
        for i in range(plan.grid_shape[0]):
            for j in range(plan.grid_shape[1]):
                t = plan.tile(i, j)
                np.testing.assert_allclose(r.partial_sums[i][:, t.cols], X[:, t.rows] @ W[t.rows, t.cols],
                                           rtol=1e-12, atol=1e-8)
        return r

    def test_square_tiles_disabled_policy(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                self._check(shape, (16, 16))

    def test_square_tiles_absent_policy(self):
        for shape in self.SHAPES[:-1]:
            with self.subTest(shape=shape):
                self._check(shape, (16, 16), edge_policy="absent")

    def test_rectangular_tiles(self):
        for shape in [(35, 50), (64, 32)]:
            for tile in [(8, 16), (16, 8), (5, 7)]:
                with self.subTest(shape=shape, tile=tile):
                    self._check(shape, tile)

    def test_tile_larger_than_matrix(self):
        r = self._check((10, 12), (16, 16))
        self.assertEqual(r.plan.n_tiles, 1)

    def test_512_full_plan_geometry(self):
        plan = plan_tiling((512, 512), (16, 16))
        self.assertEqual(plan.grid_shape, (32, 32))
        self.assertEqual(plan.n_tiles, 1024)
        self.assertEqual(plan.n_edge_tiles, 0)
        self.assertEqual(plan.adc_conversions_per_vector, 1024 * 16)


# ---------------------------------------------------------------------------
# B. Physical consistency
# ---------------------------------------------------------------------------
class PhysicalConsistencyTest(unittest.TestCase):

    def test_single_tile_matches_single_array_path(self):
        """16x16 logical on one 16x16 tile reproduces simulate_crossbar with the same random samples."""
        n, b, sigma, B, K = 16, BITS, 0.02, 8, 24
        ir, noise, calib = IRParams(), NoiseParams(), CalibParams()
        r16 = calibrate_r_wire(n, ir, DEV, 20260929)["r_eff_ohm"]
        s = point_streams(20260929, b, sigma, n)
        X = sample_inputs(s.inputs, "uniform", b, (K, n))
        # single-array reference: W and write noise from the 'arrays' stream, noise from the 'noise' stream
        ref = simulate_crossbar(X[None], b, sigma, r16, rng_arrays=s.arrays, rng_noise=s.noise, device=DEV, ir=ir,
                                noise=noise)
        cal = stat_calibration(s.cal_arrays, b, sigma, r16, n, ["uniform"], device=DEV,
                               inputs=InputParams(), ir=ir, noise=noise, calib=calib, seed=20260929)
        lo, hi = max(0.0, cal["uniform"][0] - 4 * cal["uniform"][1]), cal["uniform"][0] + 4 * cal["uniform"][1]
        Iq_ref, *_ = quantize(ref.I[0], lo, hi, B)
        Y_ref = current_to_mac(Iq_ref, X.sum(1), b, g_min=DEV.g_min, g_max=DEV.g_max, v_read=DEV.v_read)

        # tiled run with the same samples: re-derive the streams, consume W first as simulate_crossbar does
        s2 = point_streams(20260929, b, sigma, n)
        W = sample_weight_levels(s2.arrays, (1, n, n), 2 ** b)[0]
        streams = SequentialStreams(programming=s2.arrays, read_noise=s2.noise, calibration=s2.cal_arrays)
        r = simulate_tiled_crossbar(W, X, plan_tiling((n, n), (n, n)), bits=b, sigma=sigma, ir=ir, noise=noise,
                                    adc=TileADCParams(bits=B, fs_mode="stat"), calib=calib, streams=streams,
                                    keep_tile_results=True)
        t = r.tile_results[0]
        np.testing.assert_array_equal(X @ W, ref.Y_true[0])                 # same weights
        np.testing.assert_array_equal(t.I_ir, ref.I_ir[0])                  # identical IR solution
        np.testing.assert_array_equal(t.I_read, ref.I[0])                   # identical read-noise samples
        np.testing.assert_allclose(t.I_prog, ref.I_prog[0], rtol=1e-13)     # GEMM order differs only
        self.assertEqual(r.adc_windows[(n, n)], (lo, hi))                   # identical stat calibration
        np.testing.assert_array_equal(r.Y, Y_ref)                           # identical digital output
        self.assertEqual(r.r_wire, r16)

    def test_multi_tile_ir_uses_physical_tile_size(self):
        shape, tile = (64, 64), (16, 16)
        W, X = workload(shape, batch=6)
        ir = IRParams()
        r = simulate_tiled_crossbar(W, X, plan_tiling(shape, tile), bits=BITS, sigma=0.02, ir=ir,
                                    noise=NoiseParams(enabled=False), adc=TileADCParams(bits=8, fs_mode="max"),
                                    executor=CPUTileExecutor(keep_conductance=True), keep_tile_results=True)
        r16 = calibrate_r_wire(16, ir, DEV, 20260929)["r_eff_ohm"]
        r64 = calibrate_r_wire(64, ir, DEV, 20260929)["r_eff_ohm"]
        self.assertEqual(r.r_wire, r16)
        self.assertNotEqual(r16, r64)
        self.assertEqual(r.adc_windows, {(16, 16): max_window(16, DEV.v_read, DEV.g_max)})
        for t in r.tile_results:
            self.assertEqual(t.G.shape, tile)
            V = inputs_to_voltage(tile_inputs(X, t.spec), BITS, DEV.v_read)
            ro = read_array_ir(t.G, V, r16, t_read=DEV.t_read, ir=ir, noise=NoiseParams(enabled=False))
            np.testing.assert_array_equal(t.I_ir, ro.I)                    # solved on the 16x16 tile
            self.assertEqual(t.adc_window, max_window(16, DEV.v_read, DEV.g_max))
        # tiles see the attenuation of 16x16 arrays, much less than one monolithic 64x64 array
        att_tiled = 1 - r.Y_ir.sum() / r.Y_prog.sum()
        G64 = np.block([[r.tile_results[i * 4 + j].G for j in range(4)] for i in range(4)])
        V64 = inputs_to_voltage(X, BITS, DEV.v_read)
        I64 = read_array_ir(G64, V64, r64, t_read=DEV.t_read, ir=ir, noise=NoiseParams(enabled=False)).I
        att_mono = 1 - I64.sum() / ideal_mvm(V64, G64).sum()
        self.assertGreater(att_tiled, 0)
        self.assertLess(att_tiled, 0.6 * att_mono)

    def test_static_write_noise_fixed_dynamic_read_noise_redrawn(self):
        shape = (32, 32)
        W, X = workload(shape, batch=1)
        Xrep = np.repeat(X, 4, axis=0)                                      # four reads of the same vector
        kw = dict(bits=BITS, sigma=0.05, ir=IRParams(r_wire=0.0), noise=NoiseParams(),
                  adc=TileADCParams(bits=None), executor=CPUTileExecutor(keep_conductance=True),
                  keep_tile_results=True)
        a = simulate_tiled_crossbar(W, Xrep, plan_tiling(shape, (16, 16)), **kw)
        b = simulate_tiled_crossbar(W, Xrep, plan_tiling(shape, (16, 16)), sample_offset=4, **kw)
        for ta, tb in zip(a.tile_results, b.tile_results):
            np.testing.assert_array_equal(ta.G, tb.G)                       # same programmed devices
        np.testing.assert_array_equal(a.Y_ir[0], a.Y_ir[1])                 # static error identical per read
        self.assertFalse(np.array_equal(a.Y_analog[0], a.Y_analog[1]))      # read noise differs per read
        self.assertFalse(np.array_equal(a.Y_analog, b.Y_analog))            # different sample indices

    def test_rectangular_tile_physics(self):
        shape, tile = (40, 24), (8, 12)
        W, X = workload(shape, batch=5)
        ir = IRParams(r_wire=1.5)
        r = simulate_tiled_crossbar(W, X, plan_tiling(shape, tile), bits=BITS, sigma=0.02, ir=ir, noise=NoiseParams(),
                                    adc=TileADCParams(bits=8, fs_mode="stat"),
                                    executor=CPUTileExecutor(keep_conductance=True), keep_tile_results=True)
        for t in r.tile_results:
            self.assertEqual(t.G.shape, tile)
            V = inputs_to_voltage(tile_inputs(X, t.spec), BITS, DEV.v_read)
            ref = RectIRSolver(t.G, 1.5).solve(V.T, tol=ir.tol, max_iter=ir.max_iter).I
            np.testing.assert_array_equal(t.I_ir, ref[:, t.spec.active_cols])
        self.assertEqual(set(r.adc_windows), {tile})

    def test_rectangular_solver_against_direct_solve(self):
        rng = np.random.default_rng(1)
        for R, C in [(1, 5), (5, 1), (7, 3), (3, 7), (16, 24), (24, 16)]:
            G = rng.uniform(1e-6, 100e-6, (R, C))
            V = rng.uniform(0, 0.2, (R, 3))
            I = RectIRSolver(G, 0.8).solve(V, tol=1e-12, max_iter=5000).I
            np.testing.assert_allclose(I, exact_solve_rect(G, V, 0.8), rtol=1e-9, atol=1e-18)
        self.assertIsInstance(make_ir_solver(np.ones((4, 4)), 1.0), IRSolver)
        self.assertIsInstance(make_ir_solver(np.ones((4, 5)), 1.0), RectIRSolver)


# ---------------------------------------------------------------------------
# C. Edge cases
# ---------------------------------------------------------------------------
class EdgeCaseTest(unittest.TestCase):

    def test_non_divisible_geometry_partitions_matrix(self):
        plan = plan_tiling((35, 50), (16, 16))
        self.assertEqual(plan.grid_shape, (3, 4))
        cover = np.zeros((35, 50), dtype=int)
        for t in plan.tiles:
            cover[t.rows, t.cols] += 1
            self.assertEqual(t.physical_shape, (16, 16))
        np.testing.assert_array_equal(cover, 1)
        self.assertEqual(plan.tile(2, 3).valid_shape, (3, 2))
        self.assertEqual(plan.tile(2, 3).n_disabled_cells, 256 - 6)
        self.assertEqual(plan.n_edge_tiles, 6)
        self.assertEqual([t.grid_pos for t in plan.col_block(3)], [(0, 3), (1, 3), (2, 3)])
        self.assertEqual([t.grid_pos for t in plan.row_block(2)], [(2, 0), (2, 1), (2, 2), (2, 3)])
        self.assertEqual(len(plan.layout_table()), 12)
        self.assertEqual(plan.adc_conversions_per_vector, 3 * 50)
        absent = plan_tiling((35, 50), (16, 16), edge_policy="absent")
        self.assertEqual(absent.tile(2, 3).physical_shape, (3, 2))
        self.assertTrue(absent.tile(2, 3).is_edge)
        self.assertEqual(absent.n_disabled_cells, 0)
        self.assertEqual(absent.physical_shapes(), [(3, 2), (3, 16), (16, 2), (16, 16)])
        with self.assertRaises(IndexError):
            plan.tile(3, 0)

    def test_padding_of_disabled_cells(self):
        plan = plan_tiling((35, 50), (16, 16), disabled_level=0)
        W, X = workload((35, 50), batch=3)
        t = plan.tile(2, 3)
        Wt, Xt = tile_weights(W, t, 0), tile_inputs(X, t)
        self.assertEqual(Wt.shape, (16, 16))
        self.assertEqual(Xt.shape, (3, 16))
        np.testing.assert_array_equal(Wt[:3, :2], W[32:35, 48:50])
        self.assertTrue(np.all(Wt[3:, :] == 0) and np.all(Wt[:, 2:] == 0))
        np.testing.assert_array_equal(Xt[:, :3], X[:, 32:35])
        self.assertTrue(np.all(Xt[:, 3:] == 0))                            # disabled rows at 0 V

    def test_disabled_cells_contribute_no_signal_or_drive_energy(self):
        """Ideal interconnect: disabled rows (0 V) add no current and no energy; disabled columns are not converted."""
        shape = (20, 20)
        W, X = workload(shape, batch=4)
        r = simulate_tiled_crossbar(W, X, plan_tiling(shape, (16, 16)), bits=BITS, sigma=0.03,
                                    ir=IRParams(r_wire=0.0), noise=NoiseParams(enabled=False),
                                    adc=TileADCParams(bits=None), executor=CPUTileExecutor(keep_conductance=True),
                                    keep_tile_results=True)
        for t in r.tile_results:
            vr, vc = t.spec.valid_shape
            V = inputs_to_voltage(X[:, t.spec.rows], BITS, DEV.v_read)
            np.testing.assert_allclose(t.I_prog, ideal_mvm(V, t.G[:vr, :vc]), rtol=1e-13)
            # drive energy: only valid rows are driven; they also drive the disabled columns (HRS cells)
            np.testing.assert_allclose(t.E_array, ideal_read_energy(V, t.G[:vr, :], DEV.t_read), rtol=1e-13)
            self.assertEqual(t.I_q.shape, (4, vc))
        self.assertEqual(r.adc_conversions, 16 + 4 + 16 + 4)

    def test_disabled_cells_load_noise_model_but_absent_cells_do_not(self):
        shape = (20, 20)
        W, X = workload(shape, batch=2)
        kw = dict(bits=BITS, sigma=0.0, ir=IRParams(r_wire=0.0), noise=NoiseParams(), adc=TileADCParams(bits=None),
                  executor=CPUTileExecutor(keep_conductance=True), keep_tile_results=True)
        dis = simulate_tiled_crossbar(W, X, plan_tiling(shape, (16, 16), edge_policy="disabled"), **kw)
        ab = simulate_tiled_crossbar(W, X, plan_tiling(shape, (16, 16), edge_policy="absent"), **kw)
        td, ta = dis.tile_results[2], ab.tile_results[2]                 # tile (1, 0): 4 valid rows of 16
        self.assertEqual(td.G.shape, (16, 16))
        self.assertEqual(ta.G.shape, (4, 16))
        Gc = td.G.sum(axis=0)
        V = inputs_to_voltage(tile_inputs(X, td.spec), BITS, DEV.v_read)
        sd = read_noise_std(ideal_mvm(V, td.G), cell_current_sq_ideal(V, td.G), Gc, t_read=DEV.t_read,
                            **NoiseParams().model_kwargs())
        np.testing.assert_allclose(td.noise_sd, sd, rtol=1e-13)          # full physical column conductance
        self.assertTrue(np.all(td.noise_sd > ta.noise_sd))                # absent cells add nothing
        np.testing.assert_allclose(td.I_prog, ta.I_prog, rtol=1e-13)     # signal identical (0 V rows)

    def test_absent_policy_solves_ir_on_smaller_arrays(self):
        shape = (20, 20)
        W, X = workload(shape, batch=3)
        ir = IRParams()
        r = simulate_tiled_crossbar(W, X, plan_tiling(shape, (16, 16), edge_policy="absent"), bits=BITS, sigma=0.02,
                                    ir=ir, noise=NoiseParams(enabled=False), adc=TileADCParams(bits=8, fs_mode="max"),
                                    executor=CPUTileExecutor(keep_conductance=True), keep_tile_results=True)
        r16 = calibrate_r_wire(16, ir, DEV, 20260929)["r_eff_ohm"]
        self.assertEqual(r.r_wire, r16)                                   # same metal for all tiles
        shapes = {t.spec.grid_pos: t.G.shape for t in r.tile_results}
        self.assertEqual(shapes, {(0, 0): (16, 16), (0, 1): (16, 4), (1, 0): (4, 16), (1, 1): (4, 4)})
        for t in r.tile_results:
            V = inputs_to_voltage(tile_inputs(X, t.spec), BITS, DEV.v_read)
            ref = read_array(t.G, V, r16, t_read=DEV.t_read, ir=ir, noise=NoiseParams(enabled=False)).I
            np.testing.assert_array_equal(t.I_ir, ref)
            self.assertEqual(t.adc_window, max_window(t.G.shape[0], DEV.v_read, DEV.g_max))

    def test_batch_sizes_and_output_shapes(self):
        for batch in (1, 37):
            W, X = workload((35, 50), batch=batch)
            r = simulate_tiled_crossbar(W, X, plan_tiling((35, 50), (16, 16)), bits=BITS, **IDEAL)
            self.assertEqual(r.Y.shape, (batch, 50))
            self.assertEqual(r.partial_sums.shape, (3, batch, 50))
            self.assertEqual(r.E_array.shape, (batch,))
            np.testing.assert_allclose(r.Y, X @ W, rtol=1e-12, atol=1e-8)

    def test_invalid_shapes_and_operands(self):
        for bad in [((0, 5), (4, 4)), ((5, 5), (4, 0)), ((5.5, 5), (4, 4)), ((5, 5), (4,)), ((5, 5), None)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plan_tiling(*bad)
        with self.assertRaises(ValueError):
            TilingConfig((8, 8), (4, 4), edge_policy="clip")
        with self.assertRaises(ValueError):
            TilingConfig((8, 8), (4, 4), disabled_level=-1)
        plan = plan_tiling((8, 8), (4, 4))
        W, X = workload((8, 8), batch=2)
        cases = [(W[:, :7], X), (W, X[0]), (W, X[:, :7]), (W + 0.5, X), (W * 0 + 16, X), (W, -X), (W, X[:0])]
        for Wb, Xb in cases:
            with self.assertRaises(ValueError):
                simulate_tiled_crossbar(Wb, Xb, plan, bits=BITS, **IDEAL)
        with self.assertRaises(ValueError):                               # disabled level outside the LUT
            simulate_tiled_crossbar(W, X, plan_tiling((8, 8), (4, 4), disabled_level=16), bits=BITS, **IDEAL)
        with self.assertRaises(ValueError):
            TileADCParams(bits=0)
        with self.assertRaises(ValueError):
            TileADCParams(fs_mode="auto")

    def test_reduction_modes(self):
        W, X = workload((32, 32), batch=2)
        for mode in ("analog", "hierarchical"):
            plan = plan_tiling((32, 32), (16, 16), reduction_mode=mode)
            with self.subTest(mode=mode), self.assertRaises(NotImplementedError):
                simulate_tiled_crossbar(W, X, plan, bits=BITS, **IDEAL)
        with self.assertRaises(ValueError):
            plan_tiling((32, 32), (16, 16), reduction_mode="mixed")

    def test_calibrated_wire_resistance_requires_square_tiles(self):
        W, X = workload((32, 32), batch=2)
        with self.assertRaises(ValueError):
            simulate_tiled_crossbar(W, X, plan_tiling((32, 32), (8, 16)), bits=BITS, sigma=0.0, ir=IRParams(),
                                    noise=NoiseParams(enabled=False), adc=TileADCParams(bits=None))

    # ---- random streams ----
    def _physical_run(self, X, W, **kw):
        base = dict(bits=BITS, sigma=0.03, ir=IRParams(r_wire=0.0), noise=NoiseParams(),
                    adc=TileADCParams(bits=8, fs_mode="stat"))
        base.update(kw)
        return simulate_tiled_crossbar(W, X, plan_tiling((48, 40), (16, 16)), **base)

    def test_execution_order_and_batch_chunk_invariance(self):
        W, X = workload((48, 40), batch=10)
        a = self._physical_run(X, W)
        b = self._physical_run(X, W, tile_order=list(range(a.plan.n_tiles))[::-1])
        c = self._physical_run(X, W, executor=CPUTileExecutor(batch_chunk=3))
        np.testing.assert_array_equal(a.Y, b.Y)
        np.testing.assert_array_equal(a.Y, c.Y)
        # IR enabled: same random samples, solution equal within the solver tolerance
        d = self._physical_run(X, W, ir=IRParams(r_wire=0.8))
        e = self._physical_run(X, W, ir=IRParams(r_wire=0.8), executor=CPUTileExecutor(batch_chunk=3))
        np.testing.assert_allclose(d.Y_analog, e.Y_analog, rtol=1e-5, atol=1e-3)

    def test_sample_offset_reproduces_split_batches(self):
        W, X = workload((48, 40), batch=10)
        full = self._physical_run(X, W)
        p1 = self._physical_run(X[:4], W)
        p2 = self._physical_run(X[4:], W, sample_offset=4)
        np.testing.assert_array_equal(full.Y, np.vstack([p1.Y, p2.Y]))

    def test_determinism_and_seed_dependence(self):
        W, X = workload((48, 40), batch=4)
        a, b = self._physical_run(X, W), self._physical_run(X, W)
        c = self._physical_run(X, W, seed=1)
        np.testing.assert_array_equal(a.Y, b.Y)
        self.assertFalse(np.array_equal(a.Y_analog, c.Y_analog))

    def test_tile_streams_are_independent(self):
        plan = plan_tiling((32, 32), (16, 16))
        s = TileStreams(3, BITS, 0.02)
        z = [s.programming(t).standard_normal(8) for t in plan.tiles]
        self.assertEqual(len({tuple(v) for v in z}), plan.n_tiles)
        np.testing.assert_array_equal(s.programming(plan.tiles[1]).standard_normal(8), z[1])   # stable
        self.assertFalse(np.array_equal(s.read_noise(plan.tiles[0], 0).standard_normal(4),
                                        s.read_noise(plan.tiles[0], 1).standard_normal(4)))


if __name__ == "__main__":
    unittest.main()
