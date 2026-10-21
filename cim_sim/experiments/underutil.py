# -*- coding: utf-8 -*-
"""
运算规模 < 硬件规模时 ADC 量程策略的影响（原 cim_adc_underutil.py；IR drop 与读噪声关闭）。

场景
    硬件为 N_hw × N_hw 阵列（每列一个 ADC），某次运算只用到其中 n × n 子块（n < N_hw）。
    未用行输入 0 V、未用列 ADC 关闭。无 IR / 无噪声时活动列电流与独立 n × n 阵列完全相同，
    区别只在 ADC 量程按什么规模来定。

量程策略
    stat_hw        按 N_hw、满行激活校准的 μ±kσ
    stat_matched   按实际规模 n 重新校准的 μ±kσ（上限参照）
    stat_scaled    用硬件校准值解析缩放：μ·(n/N_hw)、σ·√(n/N_hw)
    max_hw         [0, N_hw·V·G_max]
    max_matched    [0, n·V·G_max]
    stat_hw_dummy  量程同 stat_hw；空闲 N_hw − n 行驱动到输入均值附近的固定电平，
                   该部分已知电流按目标电导在数字域扣除，残差来自这些单元的写入误差

随机流与主扫描一致：同一 (b, σ, n) 使用相同的权重、器件误差与输入。
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..adc import adc_lsb, max_window, quantize, stat_window
from ..analysis import knee_underutil
from ..calibration import stat_calibration
from ..config import CalibParams, DeviceParams, InputParams, IRParams, NoiseParams
from ..constants import INPUT_DISTS
from ..crossbar import ideal_mvm, ideal_read_energy
from ..inputs import inputs_to_voltage, mean_level_pattern, sample_inputs
from ..metrics import fs_effective_bits, pooled_metrics, snr_to_bits
from ..readout import current_to_mac
from ..rng import dummy_rows_stream, point_streams
from ..weights import conductance_lut, iter_programmed_arrays

STRATEGIES = ("stat_hw", "stat_matched", "stat_scaled", "max_hw", "max_matched", "stat_hw_dummy")


@dataclass
class UnderutilConfig:
    # ---- 扫描 ----
    mem_bits: list = field(default_factory=lambda: [4])
    mem_sigma: list = field(default_factory=lambda: [0.0, 0.01, 0.02, 0.05])
    adc_bits: list = field(default_factory=lambda: [4, 6, 8, 10, 12])
    hw_sizes: list = field(default_factory=lambda: [64, 128, 256, 512, 1024, 2048, 4096])
    utils: list = field(default_factory=lambda: [1.0, 15 / 16, 7 / 8, 3 / 4, 1 / 2, 1 / 4, 1 / 8])
    extra_pairs: list = field(default_factory=list)      # [(N_hw, n), ...]
    n_min: int = 8
    input_dists: list = field(default_factory=lambda: list(INPUT_DISTS))
    strategies: list = field(default_factory=lambda: list(STRATEGIES))
    n_mc: int = 100
    out: str = "results_underutil"
    # ---- 物理常数（与主扫描默认值相同）----
    sparsity: float = 0.8
    skew_a: float = 0.5
    skew_b: float = 3.0
    g_min: float = 1e-6
    g_max: float = 100e-6
    v_read: float = 0.2
    t_read: float = 10e-9
    stat_k: float = 4.0
    n_cal_arrays: int = 4
    seed: int = 20260929
    chunk_elems: int = 2 ** 24
    knee_tol: float = 0.25

    @property
    def device(self) -> DeviceParams:
        return DeviceParams(self.g_min, self.g_max, self.v_read, self.t_read)

    @property
    def inputs(self) -> InputParams:
        return InputParams(self.sparsity, self.skew_a, self.skew_b)

    @property
    def calib(self) -> CalibParams:
        return CalibParams(stat_k=self.stat_k, n_cal_arrays=self.n_cal_arrays)


def make_pairs(hw_sizes: list, utils: list, n_min: int = 8, extra_pairs: list = ()) -> list[tuple[int, int]]:
    """(N_hw, n) 组合：n = round(u·N_hw) ≥ n_min，加上 extra_pairs。"""
    pairs = set()
    for n_hw in hw_sizes:
        for u in utils:
            n = int(round(u * n_hw))
            if n >= n_min:
                pairs.add((n_hw, n))
    for n_hw, n in extra_pairs:
        pairs.add((n_hw, n))
    return sorted(pairs)


def calibrate_size(bits: int, sigma: float, size: int, dists: list, cfg: UnderutilConfig) -> dict:
    """size × size、满行激活、无 IR 无噪声的 stat 校准：{dist: (μ, σ)}。"""
    streams = point_streams(cfg.seed, bits, sigma, size)
    cal = stat_calibration(streams.cal_arrays, bits, sigma, 0.0, size, dists, device=cfg.device,
                           inputs=cfg.inputs, ir=IRParams(r_wire=0.0), noise=NoiseParams(enabled=False),
                           calib=cfg.calib, seed=cfg.seed, chunk_elems=cfg.chunk_elems)
    return cal.moments


def simulate_active(bits: int, sigma: float, n: int, n_mc: int, dists: list, cfg: UnderutilConfig) -> dict:
    """
    活动 n × n 子块的模拟（与 N_hw 无关）。

    Returns:
        dict(X (T,K,n), I (T,K,n), Y_true (T,K,n), E_array (T,K))
    """
    dev = cfg.device
    streams = point_streams(cfg.seed, bits, sigma, n)
    lut = conductance_lut(bits, dev.g_min, dev.g_max)
    X = np.stack([sample_inputs(streams.inputs, d, bits, (n_mc, n), **cfg.inputs.as_kwargs())
                  for d in dists], axis=1)
    V = inputs_to_voltage(X, bits, dev.v_read)
    I = np.empty_like(X)
    Yt = np.empty_like(X)
    E_arr = np.empty(X.shape[:2])
    for s, e, W, G, _ in iter_programmed_arrays(streams.arrays, n_mc, n, n, lut, sigma, cfg.chunk_elems):
        Yt[s:e] = X[s:e] @ W
        I[s:e] = ideal_mvm(V[s:e], G)
        E_arr[s:e] = ideal_read_energy(V[s:e], G, dev.t_read)
    return dict(X=X, I=I, Y_true=Yt, E_array=E_arr)


def simulate_dummy(bits: int, sigma: float, n: int, n_hw: int, n_mc: int, dists: list,
                   cfg: UnderutilConfig) -> dict:
    """
    空闲 d = N_hw − n 行（与活动列相交部分）的电流。

    Returns:
        dict(I_act 实际电流（含写入误差）, I_tgt 按目标电导的数字扣除值, E_array)，形状 (T,K,n) / (T,K)
    """
    d = n_hw - n
    K = len(dists)
    if d == 0:
        z = np.zeros((n_mc, K, n))
        return dict(I_act=z, I_tgt=z.copy(), E_array=np.zeros((n_mc, K)))
    dev = cfg.device
    rng = dummy_rows_stream(cfg.seed, bits, sigma, n, n_hw)
    lut = conductance_lut(bits, dev.g_min, dev.g_max)
    Xd = np.stack([mean_level_pattern(dd, bits, d, **cfg.inputs.as_kwargs()) for dd in dists])   # (K, d)
    Vd = inputs_to_voltage(Xd, bits, dev.v_read)
    I_act = np.empty((n_mc, K, n))
    I_tgt = np.empty((n_mc, K, n))
    E = np.empty((n_mc, K))
    for s, e, W, G, Gt in iter_programmed_arrays(rng, n_mc, d, n, lut, sigma, cfg.chunk_elems, keep_target=True):
        I_tgt[s:e] = ideal_mvm(Vd, Gt)
        I_act[s:e] = ideal_mvm(Vd, G)
        E[s:e] = ideal_read_energy(Vd, G, dev.t_read)
    return dict(I_act=I_act, I_tgt=I_tgt, E_array=E)


def strategy_window(strategy: str, n: int, n_hw: int, cal_hw: tuple, cal_n: tuple,
                    cfg: UnderutilConfig) -> tuple[float, float]:
    """
    各量程策略的 ADC 窗口 (lo, hi)。

    Args:
        strategy: STRATEGIES 之一
        n, n_hw:  实际 / 硬件规模
        cal_hw:   按 N_hw 校准的 (μ, σ)
        cal_n:    按 n 校准的 (μ, σ)
    """
    k = cfg.stat_k
    if strategy in ("stat_hw", "stat_hw_dummy"):
        return stat_window(*cal_hw, k=k)
    if strategy == "stat_matched":
        return stat_window(*cal_n, k=k)
    if strategy == "stat_scaled":
        u = n / n_hw
        return stat_window(cal_hw[0] * u, cal_hw[1] * np.sqrt(u), k=k)
    if strategy == "max_hw":
        return max_window(n_hw, cfg.v_read, cfg.g_max)
    if strategy == "max_matched":
        return max_window(n, cfg.v_read, cfg.g_max)
    raise ValueError(strategy)


def run_underutil(cfg: UnderutilConfig) -> pd.DataFrame:
    """完整 under-utilization 扫描，返回 summary 表（每行一个配置）。"""
    dev = cfg.device
    dists = list(cfg.input_dists)
    rows = []
    t0 = time.time()
    pairs = make_pairs(cfg.hw_sizes, cfg.utils, cfg.n_min, cfg.extra_pairs)
    ns = sorted({n for _, n in pairs})
    cal_sizes = sorted({x for p in pairs for x in p})

    for b, sigma in itertools.product(cfg.mem_bits, cfg.mem_sigma):
        L = 2 ** b - 1

        def to_mac(I, sX):
            return current_to_mac(I, sX, b, g_min=dev.g_min, g_max=dev.g_max, v_read=dev.v_read)

        cal = {sz: calibrate_size(b, sigma, sz, dists, cfg) for sz in cal_sizes}
        for n in ns:
            sim = simulate_active(b, sigma, n, cfg.n_mc, dists, cfg)
            sumX = sim["X"].sum(axis=2)
            y_fs = n * L * L
            for n_hw in sorted({h for h, m in pairs if m == n}):
                u = n / n_hw
                dum = simulate_dummy(b, sigma, n, n_hw, cfg.n_mc, dists, cfg) \
                    if "stat_hw_dummy" in cfg.strategies else None
                for di, dist in enumerate(dists):
                    Yt = sim["Y_true"][:, di]
                    I0 = sim["I"][:, di]
                    mse_an, sigvar, _ = pooled_metrics(Yt, to_mac(I0, sumX[:, di]))
                    for strat in cfg.strategies:
                        lo, hi = strategy_window(strat, n, n_hw, cal[n_hw][dist], cal[n][dist], cfg)
                        I_adc, I_sub = I0, 0.0
                        E_arr = sim["E_array"][:, di].mean()
                        if strat == "stat_hw_dummy":
                            I_adc = I0 + dum["I_act"][:, di]
                            I_sub = dum["I_tgt"][:, di]
                            E_arr = E_arr + dum["E_array"][:, di].mean()
                        for B in cfg.adc_bits:
                            Iq, slo, shi = quantize(I_adc, lo, hi, B)
                            mse, _, snr = pooled_metrics(Yt, to_mac(Iq - I_sub, sumX[:, di]))
                            Yh_noq = to_mac(I_adc - I_sub, sumX[:, di])       # ADC 前（含 dummy 残差）
                            rows.append(dict(
                                input_dist=dist, mem_bits=b, mem_sigma=sigma, N_hw=n_hw, n=n, util=u,
                                strategy=strat, adc_bits=B,
                                adc_lo_A=lo, adc_hi_A=hi, adc_lsb_A=adc_lsb(lo, hi, B),
                                I_mean_A=float(I_adc.mean()), I_std_A=float(I_adc.std()),
                                mse=mse, rmse=np.sqrt(mse), sig_std=np.sqrt(sigvar),
                                mse_analog=float(np.mean((Yh_noq - Yt) ** 2)),
                                snr_db=snr, eff_bits_sig=snr_to_bits(snr),
                                eff_bits_fs=fs_effective_bits(np.sqrt(mse), y_fs),
                                sat_lo=float(slo.mean()), sat_hi=float(shi.mean()),
                                sat_rate=float((slo | shi).mean()),
                                E_array_J=float(E_arr), mse_prog=mse_an))
            print(f"b={b} σ={sigma:.2f} n={n:<5} done  ({time.time() - t0:6.1f}s)", flush=True)
            del sim
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    d = UnderutilConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mem-bits", type=int, nargs="+", default=d.mem_bits)
    p.add_argument("--mem-sigma", type=float, nargs="+", default=d.mem_sigma)
    p.add_argument("--adc-bits", type=int, nargs="+", default=d.adc_bits)
    p.add_argument("--hw-sizes", type=int, nargs="+", default=d.hw_sizes)
    p.add_argument("--utils", type=float, nargs="+", default=d.utils)
    p.add_argument("--extra-pairs", type=str, nargs="*", default=[], help="额外的 N_hw:n 组合，如 64:40 64:44")
    p.add_argument("--n-min", type=int, default=d.n_min)
    p.add_argument("--input-dists", nargs="+", default=d.input_dists)
    p.add_argument("--strategies", nargs="+", default=d.strategies)
    p.add_argument("--n-mc", type=int, default=d.n_mc)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--knee-tol", type=float, default=d.knee_tol)
    p.add_argument("--out", default=d.out)
    return p


def main(argv: list | None = None) -> None:
    a = vars(build_parser().parse_args(argv))
    a["extra_pairs"] = [tuple(int(x) for x in s.split(":")) for s in a["extra_pairs"]]
    cfg = UnderutilConfig(**a)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str))
    df = run_underutil(cfg)
    df.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    knee_underutil(df, cfg.knee_tol).to_csv(out / "knee.csv", index=False, float_format="%.6g")
    print(f"done → {out.resolve()}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
