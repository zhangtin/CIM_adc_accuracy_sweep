#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — end-to-end walkthrough of the CIM compute-side simulation framework.

Runs one 16x16 memristor crossbar through the complete signal chain:

    digital weights W --LUT--> target conductance G_target --write noise--> actual conductance G
    digital inputs  X --DAC--> read voltages V
    (V, G) --crossbar MAC (IR drop + read noise)--> column currents I
    I --ADC--> quantized currents --digital readout--> reconstructed MAC Y_adc
    Y_adc vs. ideal digital MAC Y_ideal = X @ W  -->  error metrics

Scope: compute side only. Weight programming is abstracted as a conductance assignment
(static, per-device error); write pulses and device switching dynamics are not modeled.

Every computation is delegated to the public cim_sim API; this script only orchestrates the
calls and reports results. It needs no external data files and finishes in about a second.

Usage:
    python main.py                       # default 16x16 example
    python main.py --no-ir --no-noise    # isolate programming error + ADC quantization
    python main.py --n 64 --adc-bits 6 --fs-mode max
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cim_sim import CalibParams, DeviceParams, InputParams, IRParams, NoiseParams  # noqa: E402
from cim_sim.adc import adc_lsb, adc_window, quantize  # noqa: E402
from cim_sim.calibration import stat_calibration  # noqa: E402
from cim_sim.crossbar import ideal_mvm, ideal_read_energy, read_array_ir  # noqa: E402
from cim_sim.inputs import inputs_to_voltage, sample_inputs  # noqa: E402
from cim_sim.ir_drop import calibrate_r_wire  # noqa: E402
from cim_sim.metrics import (fs_effective_bits, mac_fullscale, pooled_metrics, snr_to_bits,  # noqa: E402
                             trial_metrics)
from cim_sim.read_noise import add_read_noise, cell_current_sq_ideal, read_noise_std  # noqa: E402
from cim_sim.readout import current_to_mac  # noqa: E402
from cim_sim.rng import point_streams  # noqa: E402
from cim_sim.weights import (apply_write_noise, conductance_lut, levels_to_conductance,  # noqa: E402
                             sample_weight_levels, sample_write_noise)


# ---------------------------------------------------------------------------
# Reporting helpers (presentation only; no simulation logic)
# ---------------------------------------------------------------------------
def describe(name: str, x: np.ndarray, unit: str, scale: float = 1.0, prefix: str = "") -> None:
    """Print shape, unit and value range of one intermediate quantity."""
    x = np.asarray(x, dtype=np.float64)
    print(f"  {name:<26s} shape={str(x.shape):<10s} {'[' + prefix + unit + ']':<5s} "
          f"min={x.min() / scale:10.4g}  mean={x.mean() / scale:10.4g}  max={x.max() / scale:10.4g}")


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 72 - len(title)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=16, help="crossbar size N (N x N array, one ADC per column)")
    p.add_argument("--k", type=int, default=64, help="number of input vectors read through the same array")
    p.add_argument("--mem-bits", type=int, default=4, help="weight / DAC resolution b")
    p.add_argument("--sigma", type=float, default=0.02, help="relative write (programming) error, 0.02 = 2 %%")
    p.add_argument("--adc-bits", type=int, default=8, help="ADC resolution B")
    p.add_argument("--fs-mode", choices=["stat", "max"], default="stat", help="ADC full-scale strategy")
    p.add_argument("--input-dist", choices=["uniform", "sparse", "skewed"], default="uniform")
    p.add_argument("--no-ir", action="store_true", help="disable IR drop (ideal interconnect)")
    p.add_argument("--no-noise", action="store_true", help="disable dynamic read noise")
    p.add_argument("--seed", type=int, default=20260929, help="global random seed")
    return p


def main(argv: list | None = None) -> dict:
    args = build_parser().parse_args(argv)
    n, K, b, sigma, B = args.n, args.k, args.mem_bits, args.sigma, args.adc_bits

    # -----------------------------------------------------------------------
    # Step 1. Parameters.
    #   DeviceParams : conductance window [g_min, g_max] (S), DAC full-scale v_read (V), read time t_read (s)
    #   IRParams     : r_wire < 0 -> per-segment wire resistance calibrated so that the mean signal
    #                  attenuation follows A(N) (1 % at N=32, 10 % at N=4096); r_wire = 0 -> no IR drop
    #   NoiseParams  : dynamic read noise (relative per-cell read noise, thermal, shot, TIA input noise)
    #   CalibParams  : ADC "stat" full scale = mu +/- k*sigma of the column current
    # -----------------------------------------------------------------------
    device = DeviceParams()
    inputs = InputParams()
    ir = IRParams(r_wire=0.0) if args.no_ir else IRParams()
    noise = NoiseParams(enabled=not args.no_noise)
    calib = CalibParams()
    # One SeedSequence per (b, sigma, N) point: independent streams for array programming,
    # inputs, calibration arrays and read noise (same convention as the full sweep).
    streams = point_streams(args.seed, b, sigma, n)

    section("Step 1  Configuration")
    print(f"  array {n}x{n}, {K} input vectors, weight/DAC {b} bit, write sigma {sigma:.1%}, "
          f"ADC {B} bit ({args.fs_mode} full scale), input '{args.input_dist}'")
    print(f"  G window [{device.g_min * 1e6:.1f}, {device.g_max * 1e6:.1f}] uS, "
          f"v_read {device.v_read} V, t_read {device.t_read * 1e9:.1f} ns, "
          f"IR drop {'off' if args.no_ir else 'on'}, read noise {'off' if args.no_noise else 'on'}, "
          f"seed {args.seed}")

    # -----------------------------------------------------------------------
    # Step 2. Digital weight matrix.
    #   Output W: (N_rows, N_cols) unsigned integers in [0, 2^b - 1] (stored as float64).
    # -----------------------------------------------------------------------
    W = sample_weight_levels(streams.arrays, (n, n), n_levels=2 ** b)

    # -----------------------------------------------------------------------
    # Step 3. Ideal conductance LUT.
    #   Output lut: (2^b,) conductance per weight level, linear between g_min and g_max (S).
    #   Replace this table to model non-linear level placement or measured device levels.
    # -----------------------------------------------------------------------
    lut = conductance_lut(b, device.g_min, device.g_max)

    # -----------------------------------------------------------------------
    # Step 4. Weight-to-conductance mapping (LUT lookup).
    #   Output G_target: (N, N) conductance the programming step aims for (S).
    # -----------------------------------------------------------------------
    G_target = levels_to_conductance(W, lut)

    # -----------------------------------------------------------------------
    # Step 5. Write noise -> actual conductance.
    #   G = max(G_target * (1 + sigma*z), 0), z ~ N(0, 1) i.i.d. per device.
    #   This is a STATIC error: drawn once when the array is programmed and frozen for all reads.
    #   Drawn from the same stream right after W, exactly as weights.program_arrays() does.
    # -----------------------------------------------------------------------
    if sigma > 0:
        G = apply_write_noise(G_target, sample_write_noise(streams.arrays, G_target.shape, sigma))
    else:
        G = G_target.copy()

    section("Steps 2-5  Weights -> LUT -> target conductance -> programmed conductance")
    describe("W (digital weights)", W, "-")
    describe("lut", lut, "S", 1e-6, "u")
    describe("G_target", G_target, "S", 1e-6, "u")
    describe("G (after write noise)", G, "S", 1e-6, "u")
    rel = (G - G_target) / G_target
    print(f"  realized write error: mean {rel.mean():+.3%}, std {rel.std():.3%} (target sigma {sigma:.1%})")

    # -----------------------------------------------------------------------
    # Step 6. Digital inputs and DAC.
    #   X: (K, N_rows) unsigned integers in [0, 2^b - 1]; V = X * v_read / (2^b - 1) (V).
    # -----------------------------------------------------------------------
    X = sample_inputs(streams.inputs, args.input_dist, b, (K, n), **inputs.as_kwargs())
    V = inputs_to_voltage(X, b, device.v_read)

    section("Step 6  Inputs -> DAC")
    describe("X (digital inputs)", X, "-")
    describe("V (read voltages)", V, "V", 1e-3, "m")

    # -----------------------------------------------------------------------
    # Step 7. Crossbar analog MAC.
    #   I_prog : (K, N_cols) Kirchhoff column currents V @ G with ideal wires (A).
    #   I_ir   : (K, N_cols) currents with wire resistance, from the full nodal solution (A).
    #   I_read : (K, N_cols) I_ir plus DYNAMIC read noise, redrawn on every read (A).
    #   E_read : (K,) energy delivered by the row drivers during one read (J).
    # -----------------------------------------------------------------------
    r_wire = calibrate_r_wire(n, ir, device, args.seed)["r_eff_ohm"]
    I_prog = ideal_mvm(V, G)
    if r_wire > 0:
        ro = read_array_ir(G, V, r_wire, t_read=device.t_read, ir=ir, noise=noise)
        I_ir, E_read, noise_sd, iters = ro.I, ro.E, ro.noise_sd, ro.iters
    else:
        I_ir, E_read, iters = I_prog, ideal_read_energy(V, G, device.t_read), 0
        noise_sd = (read_noise_std(I_ir, cell_current_sq_ideal(V, G), G.sum(axis=0), t_read=device.t_read,
                                   **noise.model_kwargs()) if noise.enabled else None)
    if noise_sd is None:
        noise_sd = np.zeros_like(I_ir)
    I_read = add_read_noise(streams.noise, I_ir, noise_sd) if noise.enabled else I_ir.copy()

    section("Step 7  Crossbar MAC (IR drop + read noise)")
    print(f"  wire resistance per segment r_wire = {r_wire:.4g} Ohm"
          + (f" (nodal solver converged in {iters} iterations)" if r_wire > 0 else ""))
    describe("I_prog (ideal wires)", I_prog, "A", 1e-6, "u")
    describe("I_ir (with IR drop)", I_ir, "A", 1e-6, "u")
    describe("noise_sd (per read)", noise_sd, "A", 1e-9, "n")
    describe("I_read (ADC input)", I_read, "A", 1e-6, "u")
    describe("E_read (array energy)", E_read, "J", 1e-15, "f")
    ir_atten = 1.0 - I_ir.sum() / I_prog.sum()
    print(f"  mean IR-drop attenuation {ir_atten:.3%}; "
          f"read-noise rms / mean signal {np.sqrt(np.mean(noise_sd ** 2)) / I_ir.mean():.3%}")

    # -----------------------------------------------------------------------
    # Step 8. ADC full scale.
    #   "max":  [0, N * v_read * g_max], the largest current an ideal array can produce.
    #   "stat": mu +/- k*sigma of the column current, estimated at design time on INDEPENDENT
    #           calibration arrays (same statistics as the deployed array, different devices).
    # -----------------------------------------------------------------------
    stat = None
    if args.fs_mode == "stat":
        cal = stat_calibration(streams.cal_arrays, b, sigma, r_wire, n, [args.input_dist], device=device,
                               inputs=inputs, ir=ir, noise=noise, calib=calib, seed=args.seed)
        stat = cal[args.input_dist]
    lo, hi = adc_window(args.fs_mode, n, v_read=device.v_read, g_max=device.g_max, stat=stat,
                        stat_k=calib.stat_k)

    # -----------------------------------------------------------------------
    # Step 9. ADC quantization and digital reconstruction.
    #   quantize(): uniform mid-tread ADC, codes 0 .. 2^B - 1 over [lo, hi], clipped outside.
    #   current_to_mac(): Y = (I / k_dac - g_min * sum(X)) / dG inverts the affine
    #   weight/input-to-current mapping; sum(X) is known in the digital domain.
    # -----------------------------------------------------------------------
    I_q, sat_lo, sat_hi = quantize(I_read, lo, hi, B)

    def to_mac(I: np.ndarray) -> np.ndarray:
        return current_to_mac(I, X.sum(axis=1), b, g_min=device.g_min, g_max=device.g_max, v_read=device.v_read)

    Y_ideal = X @ W                    # ideal digital MAC (integers)
    Y_target = to_mac(ideal_mvm(V, G_target))   # error-free analog chain; must equal Y_ideal
    Y_prog, Y_ir, Y_analog, Y_adc = to_mac(I_prog), to_mac(I_ir), to_mac(I_read), to_mac(I_q)

    section("Steps 8-9  ADC and digital readout")
    print(f"  ADC window [{lo * 1e6:.3f}, {hi * 1e6:.3f}] uA, LSB {adc_lsb(lo, hi, B) * 1e9:.3f} nA, "
          f"clipped low {sat_lo.mean():.2%}, high {sat_hi.mean():.2%}")
    describe("I_q (quantized)", I_q, "A", 1e-6, "u")
    describe("Y_ideal = X @ W", Y_ideal, "-")
    describe("Y_adc (reconstructed)", Y_adc, "-")
    print(f"  readout inversion check: max |Y_target - Y_ideal| = {np.abs(Y_target - Y_ideal).max():.2e}")

    # -----------------------------------------------------------------------
    # Step 10. Error metrics against the ideal digital MAC.
    #   trial_metrics(): per input vector (rows of the (K, N) arrays); the error is split into
    #   programming (static), IR drop, read noise (dynamic) and ADC quantization contributions.
    #   pooled_metrics(): MSE / SNR pooled over all K x N outputs.
    # -----------------------------------------------------------------------
    y_fs = mac_fullscale(n, b)
    m = trial_metrics(Y_ideal, Y_prog, Y_ir, Y_analog, Y_adc, sat_lo | sat_hi, y_fs)
    mse, _, snr_db = pooled_metrics(Y_ideal, Y_adc)
    rmse = float(np.sqrt(mse))

    section("Step 10  Error metrics (MAC units; full scale = N*(2^b-1)^2 = %d)" % y_fs)
    for key, label in (("rmse_prog", "programming error (static)"), ("rmse_ir", "IR drop increment"),
                       ("rmse_noise", "read noise increment (dynamic)"), ("rmse_device", "all analog errors"),
                       ("rmse_adc", "ADC quantization increment"), ("rmse", "total, ADC output")):
        print(f"  rms {label:<32s} {m[key].mean():10.4f}")
    print(f"  pooled RMSE {rmse:.4f}  ({rmse / y_fs:.3e} of full scale), max |error| {m['max_abs_err'].max():.3f}")
    print(f"  pooled SNR {snr_db:.2f} dB -> signal-referred effective bits {snr_to_bits(snr_db):.2f}")
    print(f"  full-scale-referred effective bits {fs_effective_bits(rmse, y_fs):.2f}")
    print(f"  outputs exactly recovered after rounding: {m['exact_rate'].mean():.1%}")
    print(f"  array read energy {E_read.mean() * 1e15:.2f} fJ per read, "
          f"{E_read.mean() / (n * n) * 1e15:.3f} fJ per MAC (ADC energy not included)")

    return dict(W=W, lut=lut, G_target=G_target, G=G, X=X, V=V, I_prog=I_prog, I_ir=I_ir, I_read=I_read,
                noise_sd=noise_sd, E_read=E_read, r_wire=r_wire, adc_window=(lo, hi), I_q=I_q,
                Y_ideal=Y_ideal, Y_target=Y_target, Y_adc=Y_adc, metrics=m, snr_db=snr_db, rmse=rmse)


if __name__ == "__main__":
    main()
