# -*- coding: utf-8 -*-
"""
ADC 满量程的 stat 校准：在独立的校准阵列上估计列电流的均值 μ 与标准差 σ，
ADC 量程取 [max(0, μ − kσ), μ + kσ]（见 adc.stat_window）。

半解析法（默认）
    对每个校准阵列求有效传输矩阵 T（I = Vᵀ·T，含 IR drop 与写入误差）的列矩
        S1_j = Σ_i T_ij，S2_j = Σ_i T_ij²
    输入 V_i i.i.d.（均值 m、方差 v）⇒ μ_j = m·S1_j，Var_j = v·S2_j + 噪声方差。
    跨列、跨阵列合并：μ = ⟨μ_j⟩，σ² = ⟨Var_j⟩ + Var(μ_j)。
    N ≤ exact_max 时用单位向量精确求 T；更大 N 用 Rademacher 探针估计 S2。

抽样法（mode = "sampled"，仅 IR 关闭时）
    直接在校准阵列上抽样输入、统计列电流。

校准阵列同时给出逐列位置增益 col_gain = S1 / ΣG（跨校准阵列平均），供 readout 的数字增益校正使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CalibParams, DeviceParams, InputParams, IRParams, NoiseParams
from .inputs import dac_step, input_moments, inputs_to_voltage, sample_inputs
from .ir_drop import IRSolver
from .read_noise import read_noise_std
from .rng import probe_stream
from .weights import conductance_lut, iter_programmed_arrays


@dataclass
class StatCalibration:
    moments: dict = field(default_factory=dict)   # {input_dist: (μ, σ)}，单位 A
    col_gain: np.ndarray | None = None            # (N,) 逐列位置增益；抽样法为 None

    def __getitem__(self, dist: str) -> tuple[float, float]:
        return self.moments[dist]


def column_transfer_moments(G: np.ndarray, r_wire: float, *, tol: float = 1e-6, max_iter: int = 500,
                            exact_max: int = 256, n_probes: int = 32,
                            probe_rng: np.random.Generator | None = None,
                            block: int = 64, probe_block: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    单个阵列传输矩阵的列矩 (S1, S2)，单位 A/V 与 (A/V)²。

    Args:
        G:          (N, N) 实际电导 (S)
        r_wire:     每段线电阻 (Ω)；0 时 T = G，直接返回 (ΣG, ΣG²)
        tol, max_iter: IR 求解精度
        exact_max:  N ≤ exact_max 时精确求 T（N 次单位向量求解，按 block 列一批）
        n_probes:   N > exact_max 时 Rademacher 探针数
        probe_rng:  探针随机流（N > exact_max 且 r_wire > 0 时必需）
        block:      精确法每批单位向量数
        probe_block: 探针法每批探针数
    """
    n = G.shape[0]
    if r_wire == 0:
        return G.sum(axis=0), (G ** 2).sum(axis=0)
    sol = IRSolver(G, r_wire)
    S1 = sol.solve(np.ones((n, 1)), tol=tol, max_iter=max_iter).I[0]
    if n <= exact_max:                           # 精确 T：单位向量输入
        T = np.empty((n, n))
        for c0 in range(0, n, block):
            c1 = min(n, c0 + block)
            Vb = np.zeros((n, c1 - c0))
            Vb[np.arange(c0, c1), np.arange(c1 - c0)] = 1.0
            T[c0:c1] = sol.solve(Vb, tol=tol, max_iter=max_iter).I
        S2 = (T ** 2).sum(axis=0)
    else:                                        # Rademacher 探针：E[(zᵀT)_j²] = Σ_i T_ij²
        if probe_rng is None:
            raise ValueError("N > exact_max 时需要 probe_rng")
        acc = np.zeros(n)
        for p0 in range(0, n_probes, probe_block):
            kk = min(probe_block, n_probes - p0)
            Z = probe_rng.choice([-1.0, 1.0], size=(n, kk))
            acc += (sol.solve(Z, tol=tol, max_iter=max_iter).I ** 2).sum(axis=0)
        S2 = acc / n_probes
    return S1, S2


def column_current_stats(S1: np.ndarray, S2: np.ndarray, G_col: np.ndarray, dist: str, bits: int, *,
                         device: DeviceParams, inputs: InputParams, noise: NoiseParams) -> tuple[float, float]:
    """
    由传输矩阵列矩和输入分布的解析矩求列电流 (μ, σ)。

    Args:
        S1, S2: (A, N) 各校准阵列的列矩（A = 校准阵列数）
        G_col:  (A, N) 各校准阵列的列电导和
        dist:   输入分布名
        bits:   DAC 位宽
    """
    ex, ex2 = input_moments(dist, bits, **inputs.as_kwargs())
    k = dac_step(bits, device.v_read)
    m, m2 = ex * k, ex2 * k * k
    v = m2 - m * m
    mu_j = m * S1
    var_j = v * S2
    if noise.enabled:
        var_j = var_j + read_noise_std(mu_j, m2 * S2, G_col, t_read=device.t_read, **noise.model_kwargs()) ** 2
    return float(mu_j.mean()), float(np.sqrt(var_j.mean() + mu_j.var()))


def stat_calibration(rng_cal_arrays: np.random.Generator, bits: int, sigma: float, r_wire: float, n: int,
                     dists: list, *, device: DeviceParams, inputs: InputParams, ir: IRParams,
                     noise: NoiseParams, calib: CalibParams, seed: int, lut: np.ndarray | None = None,
                     chunk_elems: int = 2 ** 24) -> StatCalibration:
    """
    半解析 stat 校准。

    Args:
        rng_cal_arrays: 校准阵列的编程随机流（rng.point_streams(...).cal_arrays）
        bits, sigma:    忆阻器位宽与写入误差
        r_wire:         每段线电阻 (Ω)
        n:              阵列规模
        dists:          需要校准的输入分布列表
        device, inputs, ir, noise, calib: 参数组
        seed:           全局种子（探针随机流用）
        lut:            理想权重 LUT；None 时用线性 LUT

    Returns:
        StatCalibration(moments={dist: (μ, σ)}, col_gain)
    """
    if lut is None:
        lut = conductance_lut(bits, device.g_min, device.g_max)
    S1s, S2s, Gcols = [], [], []
    for s, e, W, G, _ in iter_programmed_arrays(rng_cal_arrays, calib.n_cal_arrays, n, n, lut, sigma,
                                                chunk_elems):
        for t in range(e - s):
            Gt = G[t]
            prng = probe_stream(seed, n, bits, sigma, s + t) if (r_wire != 0 and n > calib.exact_max) else None
            S1, S2 = column_transfer_moments(Gt, r_wire, tol=ir.tol, max_iter=ir.max_iter,
                                             exact_max=calib.exact_max, n_probes=calib.probes, probe_rng=prng)
            S1s.append(S1); S2s.append(S2); Gcols.append(Gt.sum(axis=0))
        del W, G
    S1, S2, Gc = np.array(S1s), np.array(S2s), np.array(Gcols)
    out = StatCalibration(col_gain=(S1 / np.maximum(Gc, 1e-30)).mean(axis=0))
    for d in dists:
        out.moments[d] = column_current_stats(S1, S2, Gc, d, bits, device=device, inputs=inputs, noise=noise)
    return out


def stat_calibration_sampled(rng_cal_arrays: np.random.Generator, rng_cal_inputs: np.random.Generator,
                             bits: int, sigma: float, n: int, dists: list, *, device: DeviceParams,
                             inputs: InputParams, calib: CalibParams, lut: np.ndarray | None = None,
                             chunk_elems: int = 2 ** 24) -> StatCalibration:
    """
    抽样 stat 校准（v2 方法，仅 IR 关闭、不含噪声）。

    Args:
        rng_cal_arrays, rng_cal_inputs: 校准阵列与校准输入随机流
        其余同 stat_calibration
    """
    if lut is None:
        lut = conductance_lut(bits, device.g_min, device.g_max)
    A, M = calib.n_cal_arrays, calib.n_cal_inputs
    Xc = np.stack([sample_inputs(rng_cal_inputs, d, bits, (A, M, n), **inputs.as_kwargs())
                   for d in dists], axis=2).reshape(A, M * len(dists), n)
    V = inputs_to_voltage(Xc, bits, device.v_read)
    Is = []
    for s, e, W, G, _ in iter_programmed_arrays(rng_cal_arrays, A, n, n, lut, sigma, chunk_elems):
        Is.append(V[s:e] @ G)
    I = np.concatenate(Is).reshape(A, M, len(dists), n)
    return StatCalibration(moments={d: (float(I[:, :, i].mean()), float(I[:, :, i].std()))
                                    for i, d in enumerate(dists)})
