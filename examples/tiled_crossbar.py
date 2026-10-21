#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
examples/tiled_crossbar.py — a logical weight matrix executed on a grid of physical crossbar tiles.

Default: a 64x64 logical matrix on 16x16 physical tiles (4 x 4 = 16 independent arrays). Each tile
is programmed, read (IR drop on its own 16x16 wires, dynamic read noise) and digitized by its own
ADCs; the digital partial sums are then added to form the logical output, which is compared with
the ideal digital MAC X @ W.

Usage:
    python examples/tiled_crossbar.py
    python examples/tiled_crossbar.py --logical 512 512 --tile 16 16 --batch 16
    python examples/tiled_crossbar.py --logical 35 50 --tile 16 16 --edge-policy absent
    python examples/tiled_crossbar.py --logical 128 128 --tile-sweep 8 16 32 64 128 \
        --energy-csv ADC_22nm_energy_vs_bits.csv
    python examples/tiled_crossbar.py --tile 8 16 --r-wire 0.8      # rectangular tiles need an explicit r_wire

Only the CPU reference backend is used; no GPU is required.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cim_sim import CalibParams, DeviceParams, InputParams, IRParams, NoiseParams  # noqa: E402
from cim_sim.energy import load_adc_energy  # noqa: E402
from cim_sim.inputs import sample_inputs  # noqa: E402
from cim_sim.metrics import (fs_effective_bits, mac_fullscale, pooled_metrics, snr_to_bits,  # noqa: E402
                             trial_metrics)
from cim_sim.tiling import (CPUTileExecutor, TileADCParams, plan_tiling, simulate_tiled_crossbar,  # noqa: E402
                            tile_inputs, tile_weights, workload_rng)
from cim_sim.weights import sample_weight_levels  # noqa: E402


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 72 - len(title)))


def shape_line(name: str, shape, note: str = "") -> None:
    print(f"  {name:<38s} {str(tuple(shape)):<16s} {note}".rstrip())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logical", type=int, nargs=2, default=[64, 64], metavar=("ROWS", "COLS"),
                   help="logical weight-matrix shape (rows = inputs, cols = outputs)")
    p.add_argument("--tile", type=int, nargs=2, default=[16, 16], metavar=("ROWS", "COLS"),
                   help="physical crossbar tile shape")
    p.add_argument("--batch", type=int, default=32, help="number of input vectors")
    p.add_argument("--mem-bits", type=int, default=4, help="weight / DAC resolution b")
    p.add_argument("--sigma", type=float, default=0.02, help="relative write error")
    p.add_argument("--adc-bits", type=int, default=8, help="per-tile ADC resolution")
    p.add_argument("--fs-mode", choices=["stat", "max"], default="stat", help="per-tile ADC full scale")
    p.add_argument("--input-dist", choices=["uniform", "sparse", "skewed"], default="uniform")
    p.add_argument("--edge-policy", choices=["disabled", "absent"], default="disabled")
    p.add_argument("--r-wire", type=float, default=-1.0,
                   help="wire resistance per segment in Ohm; <0 = calibrated for square tiles (default), 0 = off")
    p.add_argument("--no-ir", action="store_true", help="disable IR drop (same as --r-wire 0)")
    p.add_argument("--no-noise", action="store_true", help="disable dynamic read noise")
    p.add_argument("--energy-csv", default=None, help="ADC energy table; enables ADC energy estimates")
    p.add_argument("--tile-sweep", type=int, nargs="+", default=None, metavar="N",
                   help="also compare square tile sizes N x N on the same logical workload")
    p.add_argument("--seed", type=int, default=20260929)
    return p


def run_tiled(W, X, tile, args, keep=False):
    plan = plan_tiling(tuple(args.logical), tuple(tile), edge_policy=args.edge_policy)
    res = simulate_tiled_crossbar(
        W, X, plan, bits=args.mem_bits, sigma=args.sigma, device=DeviceParams(),
        ir=IRParams(r_wire=0.0 if args.no_ir else args.r_wire), noise=NoiseParams(enabled=not args.no_noise),
        adc=TileADCParams(bits=args.adc_bits, fs_mode=args.fs_mode, calib_dist=args.input_dist),
        calib=CalibParams(), inputs=InputParams(), seed=args.seed,
        executor=CPUTileExecutor(keep_conductance=keep), keep_tile_results=keep)
    return plan, res


def main(argv: list | None = None) -> None:
    args = build_parser().parse_args(argv)
    (lr, lc), b = args.logical, args.mem_bits

    # -----------------------------------------------------------------------
    # 1. Logical workload: integer weights W and inputs X. The stream depends only on the logical
    #    problem, so different tile shapes are compared on identical data.
    # -----------------------------------------------------------------------
    rng = workload_rng(args.seed, b, (lr, lc))
    W = sample_weight_levels(rng, (lr, lc), 2 ** b)
    X = sample_inputs(rng, args.input_dist, b, (args.batch, lr))
    Y_ideal = X @ W                                         # ideal digital MAC (reference)

    # -----------------------------------------------------------------------
    # 2. Plan + simulate. Every tile runs the full single-array physics at its PHYSICAL size.
    # -----------------------------------------------------------------------
    plan, res = run_tiled(W, X, args.tile, args, keep=True)

    section("Tiling plan")
    print("  " + plan.describe().replace("\n", "\n  "))
    print(plan.layout())

    section("Stage shapes")
    t0 = res.tile_results[0]
    edge = next((t for t in res.tile_results if t.spec.is_edge), None)
    shape_line("W  logical weights", W.shape, f"levels 0..{2 ** b - 1}")
    shape_line("X  logical inputs", X.shape, "(batch, logical_rows)")
    for t in [t0] + ([edge] if edge is not None else []):
        tag = f"tile {t.spec.grid_pos}" + (" (edge)" if t.spec.is_edge else "")
        print(f"  -- {tag}: logical rows {t.spec.row_range}, cols {t.spec.col_range}, "
              f"valid {t.spec.valid_shape}, physical {t.spec.physical_shape}")
        shape_line("   W_tile  programmed levels", tile_weights(W, t.spec, plan.config.disabled_level).shape)
        shape_line("   X_tile  routed inputs", tile_inputs(X, t.spec).shape, "disabled rows = 0 V")
        shape_line("   G       programmed conductance", t.G.shape, "S, static write error")
        shape_line("   I_read  column currents (active)", t.I_read.shape, "A, IR drop + read noise")
        shape_line("   I_q     ADC output (active)", t.I_q.shape, f"A, {args.adc_bits}-bit ADC")
    shape_line("partial sums P[i]", res.partial_sums.shape, "(tile_rows, batch, logical_cols), MAC units")
    shape_line("Y  reconstructed output", res.Y.shape, "sum over tile rows")
    shape_line("Y_ideal = X @ W", Y_ideal.shape)

    section("Digital partial-sum reconstruction (input vector 0, output column 0)")
    col = 0
    for i in range(plan.grid_shape[0]):
        t = plan.tile(i, 0)
        exact = float(X[0, t.rows] @ W[t.rows, col])
        print(f"  tile ({i},0) rows {t.row_range}: P = {res.partial_sums[i, 0, col]:10.2f}   exact {exact:8.0f}")
        if i == 5 and plan.grid_shape[0] > 8:
            print(f"  ... {plan.grid_shape[0] - 7} more tile rows ...")
            break
    print(f"  Y[0,0] = sum_i P = {res.Y[0, col]:.2f}   ideal X@W = {Y_ideal[0, col]:.0f}")

    section("Accuracy (MAC units; logical full scale = rows*(2^b-1)^2 = %d)" % mac_fullscale(lr, b))
    y_fs = mac_fullscale(lr, b)
    m = trial_metrics(Y_ideal, res.Y_prog, res.Y_ir, res.Y_analog, res.Y, np.zeros_like(res.Y, bool), y_fs)
    for key, label in (("rmse_prog", "programming error (static)"), ("rmse_ir", "IR drop increment"),
                       ("rmse_noise", "read noise increment (dynamic)"), ("rmse_adc", "ADC quantization increment"),
                       ("rmse", "total")):
        print(f"  rms {label:<32s} {m[key].mean():10.3f}")
    mse, _, snr = pooled_metrics(Y_ideal, res.Y)
    print(f"  pooled RMSE {np.sqrt(mse):.3f} ({np.sqrt(mse) / y_fs:.2e} of full scale), SNR {snr:.2f} dB, "
          f"signal-referred effective bits {snr_to_bits(snr):.2f}, FS-referred {fs_effective_bits(np.sqrt(mse), y_fs):.2f}")
    print(f"  r_wire {res.r_wire:.4g} Ohm/segment, mean IR attenuation {1 - res.Y_ir.sum() / res.Y_prog.sum():.3%}, "
          f"ADC clipping {res.sat_rate:.3%}")
    print(f"  ADC windows per physical shape: " + ", ".join(
        f"{s}: [{w[0] * 1e6:.2f}, {w[1] * 1e6:.2f}] uA" for s, w in res.adc_windows.items() if w))

    section("Cost per input vector")
    print(f"  ADC conversions {res.adc_conversions} (monolithic {lr}x{lc} array would need {lc})")
    print(f"  array read energy {res.E_array.mean() * 1e12:.3f} pJ")
    if args.energy_csv:
        e = load_adc_energy(args.energy_csv)["typical"][args.adc_bits]
        print(f"  ADC energy (typical tier) {res.adc_conversions * e * 1e12:.3f} pJ "
              f"({e * 1e15:.2f} fJ per {args.adc_bits}-bit conversion)")

    # -----------------------------------------------------------------------
    # 3. Optional tile-size sweep on the same logical workload.
    # -----------------------------------------------------------------------
    if args.tile_sweep:
        section("Tile-size sweep (same W, X)")
        e = load_adc_energy(args.energy_csv)["typical"][args.adc_bits] if args.energy_csv else None
        print(f"  {'tile':>9s} {'tiles':>6s} {'r_wire':>8s} {'IR att.':>8s} {'SNR dB':>7s} {'ENOB_sig':>8s} "
              f"{'ADC conv':>8s} {'E_array pJ':>10s}" + (f" {'E_ADC pJ':>9s}" if e else ""))
        for n in args.tile_sweep:
            p, r = run_tiled(W, X, (n, n), args)
            _, _, s = pooled_metrics(Y_ideal, r.Y)
            att = 1 - r.Y_ir.sum() / r.Y_prog.sum()
            line = (f"  {f'{n}x{n}':>9s} {p.n_tiles:>6d} {r.r_wire:>8.3g} {att:>8.3%} {s:>7.2f} {snr_to_bits(s):>8.2f} "
                    f"{r.adc_conversions:>8d} {r.E_array.mean() * 1e12:>10.3f}")
            print(line + (f" {r.adc_conversions * e * 1e12:>9.3f}" if e else ""))


if __name__ == "__main__":
    main()
