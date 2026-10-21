# -*- coding: utf-8 -*-
"""
主扫描（原 cim_adc_mc.py v3）：忆阻器 crossbar 模拟 MAC + ADC 量化的 Monte Carlo 扫描，
含 IR drop、读出噪声与 stat 量程校准。

扫描变量（全组合）：mem_bits × mem_sigma × adc_bits × N × fs_mode × input_dist
输出（cfg.out_dir）：raw_trials.csv.gz / summary.csv / knee.csv / config.json / ir_calibration.csv / fig_*.png

外部调用：
    from cim_sim import SweepConfig
    from cim_sim.experiments.sweep import run_sweep, simulate_point
    cfg = SweepConfig(sizes=[64], mem_bits=[4], mem_sigma=[0.02], n_mc=50, out_dir="res")
    raw, summary = run_sweep(cfg)
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..adc import adc_lsb, adc_window, quantize
from ..analysis import knee_table, summarize
from ..calibration import StatCalibration, stat_calibration, stat_calibration_sampled
from ..config import SweepConfig
from ..constants import INPUT_DISTS
from ..crossbar import CrossbarResult, simulate_crossbar
from ..energy import add_energy_columns, check_adc_bits, load_adc_energy
from ..inputs import sample_inputs
from ..ir_drop import calibrate_r_wire, reference_attenuation_at
from ..metrics import corrected_metrics, mac_fullscale, trial_metrics
from ..readout import apply_column_gain, current_to_mac
from ..rng import point_streams


# ---------------------------------------------------------------------------
def ir_calibration_table(cfg: SweepConfig) -> pd.DataFrame:
    """每个 N 的等效线电阻 r_eff(N)（ir_calibration.csv）。"""
    irc = pd.DataFrame([calibrate_r_wire(n, cfg.ir, cfg.device, cfg.seed) for n in cfg.sizes])
    irc["ref_atten_1ohm"] = [reference_attenuation_at(n, 1.0, cfg.ir, cfg.device, cfg.seed) if n <= 1024
                             else np.nan for n in cfg.sizes] if cfg.r_wire != 0 else np.nan
    return irc


def calibrate_point(cfg: SweepConfig, streams, bits: int, sigma: float, n: int, r_wire: float) -> StatCalibration | None:
    """扫描点的 stat 校准；fs_modes 不含 "stat" 时返回 None。"""
    if "stat" not in cfg.fs_modes:
        return None
    dists = list(cfg.input_dists)
    if cfg.cal_mode == "sampled":
        return stat_calibration_sampled(streams.cal_arrays, streams.cal_inputs, bits, sigma, n, dists,
                                        device=cfg.device, inputs=cfg.inputs, calib=cfg.calib,
                                        chunk_elems=cfg.chunk_elems)
    return stat_calibration(streams.cal_arrays, bits, sigma, r_wire, n, dists, device=cfg.device,
                            inputs=cfg.inputs, ir=cfg.ir, noise=cfg.noise_params, calib=cfg.calib,
                            seed=cfg.seed, chunk_elems=cfg.chunk_elems)


def simulate_point(cfg: SweepConfig, n: int, bits: int, sigma: float, r_wire: float,
                   energy_table: dict) -> tuple[list[pd.DataFrame], CrossbarResult]:
    """
    单个扫描点 (N, b, σ)：模拟一次模拟域，然后对所有 input_dist × fs_mode × adc_bits 量化与统计。

    Args:
        cfg:          SweepConfig
        n, bits, sigma: 阵列规模、忆阻器位宽、写入误差
        r_wire:       该 N 的每段线电阻 (Ω)
        energy_table: load_adc_energy() 的输出

    Returns:
        (逐 trial 结果表的列表, CrossbarResult)
    """
    dev = cfg.device
    dists = list(cfg.input_dists)
    streams = point_streams(cfg.seed, bits, sigma, n)

    def to_mac(I, sX):
        return current_to_mac(I, sX, bits, g_min=dev.g_min, g_max=dev.g_max, v_read=dev.v_read)

    X = np.stack([sample_inputs(streams.inputs, d, bits, (cfg.n_mc, n), **cfg.inputs.as_kwargs())
                  for d in dists], axis=1)
    sim = simulate_crossbar(X, bits, sigma, r_wire, rng_arrays=streams.arrays, rng_noise=streams.noise,
                            device=dev, ir=cfg.ir, noise=cfg.noise_params, chunk_elems=cfg.chunk_elems,
                            chip_gain=cfg.chip_cal)
    sumX = X.sum(axis=2)
    Y_analog, Y_prog, Y_ir = to_mac(sim.I, sumX), to_mac(sim.I_prog, sumX), to_mac(sim.I_ir, sumX)
    y_fs = mac_fullscale(n, bits)

    cal = calibrate_point(cfg, streams, bits, sigma, n, r_wire)
    gain = np.ones(n) if (cal is None or cal.col_gain is None) else cal.col_gain

    blocks = []
    for di, dist in enumerate(dists):
        I_d, Yt_d, sX_d = sim.I[:, di], sim.Y_true[:, di], sumX[:, di]
        E_arr = sim.E_array[:, di]
        ideal_sum = sim.I_prog[:, di].sum(axis=1)
        ir_atten = 1.0 - sim.I_ir[:, di].sum(axis=1) / np.maximum(ideal_sum, 1e-30)
        noise_rms = np.sqrt(np.mean(sim.noise_sd[:, di] ** 2, axis=1))
        for mode in cfg.fs_modes:
            lo, hi = adc_window(mode, n, v_read=dev.v_read, g_max=dev.g_max,
                                stat=None if cal is None else cal[dist], stat_k=cfg.stat_k)
            for B in cfg.adc_bits:
                I_q, s_lo, s_hi = quantize(I_d, lo, hi, B)
                m = trial_metrics(Yt_d, Y_prog[:, di], Y_ir[:, di], Y_analog[:, di],
                                  to_mac(I_q, sX_d), s_lo | s_hi, y_fs)
                m.update(corrected_metrics(Yt_d, Y_prog[:, di],
                                           to_mac(apply_column_gain(sim.I_ir[:, di], gain), sX_d),
                                           to_mac(apply_column_gain(I_q, gain), sX_d), y_fs, "cal"))
                if cfg.chip_cal:
                    gc = sim.gain_chip
                    m.update(corrected_metrics(Yt_d, Y_prog[:, di],
                                               to_mac(apply_column_gain(sim.I_ir[:, di], gc), sX_d),
                                               to_mac(apply_column_gain(I_q, gc), sX_d), y_fs, "chip"))
                df = pd.DataFrame(m)
                df.insert(0, "trial", np.arange(cfg.n_mc))
                for k, v in dict(input_dist=dist, mem_bits=bits, mem_sigma=sigma, adc_bits=B,
                                 N=n, fs_mode=mode).items():
                    df.insert(0, k, v)
                df["adc_lo_A"], df["adc_hi_A"], df["adc_lsb_A"] = lo, hi, adc_lsb(lo, hi, B)
                df["I_col_mean_A"] = I_d.mean(axis=1)
                df["input_mean_frac"] = X[:, di].mean(axis=1) / (2 ** bits - 1)
                df["r_eff_ohm"] = r_wire
                df["ir_atten"] = ir_atten
                df["noise_rms_A"] = noise_rms
                df["ir_iters"] = sim.iters
                df["col_gain_mean"] = float(gain.mean())
                add_energy_columns(df, n, B, energy_table, E_arr)
                blocks.append(df)
    return blocks, sim


def run_sweep(cfg: SweepConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """完整扫描并写出全部结果文件。返回 (raw, summary)。"""
    energy = load_adc_energy(cfg.energy_csv)
    check_adc_bits(energy, cfg.adc_bits)
    if cfg.cal_mode == "sampled" and cfg.r_wire != 0:
        raise ValueError("cal_mode=sampled 仅在 --r-wire 0 时可用")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")

    t0 = time.time()
    irc = ir_calibration_table(cfg)
    irc.to_csv(out / "ir_calibration.csv", index=False)
    print(irc.to_string(index=False), f"\n(IR calibration {time.time() - t0:.1f}s)", flush=True)
    r_of = dict(zip(irc.N, irc.r_eff_ohm))

    blocks = []
    grid = list(itertools.product(cfg.sizes, cfg.mem_bits, cfg.mem_sigma))
    for idx, (n, b, sigma) in enumerate(grid, 1):
        r = float(r_of[n])
        bl, sim = simulate_point(cfg, n, b, sigma, r, energy)
        blocks += bl
        print(f"[{idx:>3}/{len(grid)}] N={n:<5} b={b} σ={sigma:.3f}  r={r:.3g}Ω  "
              f"iters≈{sim.iters.mean():.1f}  elapsed {time.time() - t0:7.1f}s", flush=True)
        del sim, bl

    raw = pd.concat(blocks, ignore_index=True)
    raw.to_csv(out / "raw_trials.csv.gz", index=False, float_format="%.6g")
    summary = summarize(raw)
    summary.to_csv(out / "summary.csv", index=False)
    knee_table(summary, cfg.knee_tol).to_csv(out / "knee.csv", index=False)
    return raw, summary


def resummarize(cfg: SweepConfig) -> pd.DataFrame:
    """不重跑仿真：由已有 raw_trials.csv.gz 重新生成 summary.csv / knee.csv。"""
    out = Path(cfg.out_dir)
    summary = summarize(pd.read_csv(out / "raw_trials.csv.gz"))
    summary.to_csv(out / "summary.csv", index=False)
    knee_table(summary, cfg.knee_tol).to_csv(out / "knee.csv", index=False)
    return summary


# ---------------------------------------------------------------------------
# CLI（参数与原 cim_adc_mc.py 完全一致）
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    d = SweepConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mem-bits", type=int, nargs="+", default=d.mem_bits)
    p.add_argument("--mem-sigma", type=float, nargs="+", default=d.mem_sigma, help="相对电导误差 σ（0.02 = 2%%）")
    p.add_argument("--adc-bits", type=int, nargs="+", default=d.adc_bits)
    p.add_argument("--sizes", type=int, nargs="+", default=d.sizes)
    p.add_argument("--fs-modes", nargs="+", default=d.fs_modes, choices=["max", "stat"])
    p.add_argument("--input-dists", nargs="+", default=d.input_dists, choices=list(INPUT_DISTS))
    p.add_argument("--n-mc", type=int, default=d.n_mc)
    p.add_argument("--sparsity", type=float, default=d.sparsity)
    p.add_argument("--skew-a", type=float, default=d.skew_a)
    p.add_argument("--skew-b", type=float, default=d.skew_b)
    p.add_argument("--g-min", type=float, default=d.g_min)
    p.add_argument("--g-max", type=float, default=d.g_max)
    p.add_argument("--v-read", type=float, default=d.v_read)
    p.add_argument("--t-read", type=float, default=d.t_read)
    p.add_argument("--r-wire", type=float, default=d.r_wire,
                   help="<0：按 A(N) 标定等效线电阻（默认）；0：关闭 IR drop；>0：固定线电阻 Ω/段")
    p.add_argument("--ir-atten-lo", type=float, default=d.ir_atten_lo)
    p.add_argument("--ir-atten-hi", type=float, default=d.ir_atten_hi)
    p.add_argument("--ir-n-lo", type=int, default=d.ir_n_lo)
    p.add_argument("--ir-n-hi", type=int, default=d.ir_n_hi)
    p.add_argument("--ir-tol", type=float, default=d.ir_tol)
    p.add_argument("--ir-max-iter", type=int, default=d.ir_max_iter)
    p.add_argument("--no-noise", dest="noise", action="store_false")
    p.add_argument("--chip-cal", action="store_true", help="另算逐芯片逐列增益校正指标（*_chip）")
    p.add_argument("--read-noise-rel", type=float, default=d.read_noise_rel)
    p.add_argument("--temp-K", type=float, default=d.temp_K)
    p.add_argument("--tia-en", type=float, default=d.tia_en)
    p.add_argument("--tia-in", type=float, default=d.tia_in)
    p.add_argument("--stat-k", type=float, default=d.stat_k)
    p.add_argument("--n-cal-arrays", type=int, default=d.n_cal_arrays)
    p.add_argument("--n-cal-inputs", type=int, default=d.n_cal_inputs)
    p.add_argument("--cal-mode", default=d.cal_mode, choices=["analytic", "sampled"])
    p.add_argument("--cal-probes", type=int, default=d.cal_probes)
    p.add_argument("--cal-exact-max", type=int, default=d.cal_exact_max)
    p.add_argument("--energy-csv", default=d.energy_csv)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--out", dest="out_dir", default=d.out_dir)
    p.add_argument("--chunk-elems", type=int, default=d.chunk_elems)
    p.add_argument("--plot-sigma", type=float, default=d.plot_sigma)
    p.add_argument("--plot-sizes", type=int, nargs="+", default=d.plot_sizes)
    p.add_argument("--plot-mem-bits", type=int, default=d.plot_mem_bits)
    p.add_argument("--knee-tol", type=float, default=d.knee_tol)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-self-check", action="store_true")
    p.add_argument("--resummarize", action="store_true",
                   help="不重跑仿真：从已有 raw_trials.csv.gz 重新生成 summary / knee / 图")
    return p


def main(argv: list | None = None) -> None:
    a = vars(build_parser().parse_args(argv))
    no_plots, resum, no_check = a.pop("no_plots"), a.pop("resummarize"), a.pop("no_self_check")
    cfg = SweepConfig(**a)

    def plots(summary):
        if no_plots:
            return
        from ..plotting.sweep import make_plots
        make_plots(summary, cfg)
        if "eff_bits_cal_pooled" in summary.columns:
            make_plots(summary, cfg, cal=True)

    if resum:
        summary = resummarize(cfg)
        plots(summary)
        print(f"re-summarized → {Path(cfg.out_dir).resolve()}")
        return
    if not no_check:
        from ..selfcheck import self_check
        self_check(cfg)
    raw, summary = run_sweep(cfg)
    plots(summary)
    print(f"done → {Path(cfg.out_dir).resolve()}  ({len(raw)} trials, {len(summary)} configs)")


if __name__ == "__main__":
    main()
