# -*- coding: utf-8 -*-
"""
crossbar 模拟 MAC：已编程阵列 + 输入电压 → 列电流（理想 / IR drop / 读出噪声）与阵列读能量。

    ideal_mvm                理想列电流 Vᵀ·G
    ideal_read_energy        理想阵列读能量 t_read·Σ V_i² G_ij
    read_array_ir            单个阵列带线电阻读出（电流、驱动能量、噪声）
    simulate_crossbar        一组 Monte Carlo 阵列的完整模拟（主扫描用）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DeviceParams, IRParams, NoiseParams
from .inputs import inputs_to_voltage
from .ir_drop import IRSolver
from .read_noise import add_read_noise, cell_current_sq_from_fields, cell_current_sq_ideal, read_noise_std
from .weights import conductance_lut, iter_programmed_arrays


def ideal_mvm(V: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    理想列电流 I_j = Σ_i V_i G_ij。

    Args:
        V: (..., K, N_rows) 输入电压 (V)；G: (..., N_rows, N_cols) 电导 (S)

    Returns:
        (..., K, N_cols) 列电流 (A)
    """
    return V @ G


def ideal_read_energy(V: np.ndarray, G: np.ndarray, t_read: float) -> np.ndarray:
    """
    理想阵列读能量 E = t_read · Σ_ij V_i² G_ij (J)。

    Args:
        V: (..., K, N_rows)；G: (..., N_rows, N_cols)

    Returns:
        (..., K)
    """
    return t_read * ((V ** 2) @ G).sum(axis=-1)


@dataclass
class IRReadout:
    I: np.ndarray            # (K, N) 列电流（含 IR drop，无噪声）
    E: np.ndarray            # (K,) 驱动器输出能量 (J)
    iters: int
    noise_sd: np.ndarray | None   # (K, N) 读出噪声标准差（noise.enabled 时）
    col_gain: np.ndarray | None   # (N,) 本阵列逐列增益（want_gain=True 时）


def read_array_ir(G: np.ndarray, V: np.ndarray, r_wire: float, *, t_read: float, ir: IRParams,
                  noise: NoiseParams, want_gain: bool = False) -> IRReadout:
    """
    单个阵列带线电阻读出。

    Args:
        G:         (N, N) 实际电导 (S)
        V:         (K, N) 输入电压 (V)（每行一组输入）
        r_wire:    每段线电阻 (Ω)，> 0
        t_read:    读出时间 (s)
        ir:        IRParams（tol / max_iter）
        noise:     NoiseParams；enabled 时返回噪声标准差
        want_gain: 是否另求本阵列逐列增益（多一次求解）
    """
    sol = IRSolver(G, r_wire)
    res = sol.solve(V.T, tol=ir.tol, max_iter=ir.max_iter, return_fields=noise.enabled)
    gain = sol.column_gain(tol=ir.tol, max_iter=ir.max_iter) if want_gain else None
    E = t_read * (V * res.I_wl).sum(axis=1)
    sd = None
    if noise.enabled:
        cell_sq = cell_current_sq_from_fields(res.u, res.w, G, overwrite_u=True)
        sd = read_noise_std(res.I, cell_sq, G.sum(axis=0), t_read=t_read, **noise.model_kwargs())
    return IRReadout(I=res.I, E=E, iters=res.iters, noise_sd=sd, col_gain=gain)


@dataclass
class CrossbarResult:
    """simulate_crossbar 的输出。形状中 T = Monte Carlo 阵列数，K = 输入组数，N = 列数。"""
    I: np.ndarray           # (T,K,N) ADC 输入电流（IR drop + 噪声）
    I_prog: np.ndarray      # (T,K,N) 仅写入误差的理想 crossbar 电流
    I_ir: np.ndarray        # (T,K,N) 含 IR drop、无噪声
    noise_sd: np.ndarray    # (T,K,N) 读出噪声标准差
    Y_true: np.ndarray      # (T,K,N) 理想整数 MAC 结果 X·W
    E_array: np.ndarray     # (T,K) 阵列读能量 (J)
    iters: np.ndarray       # (T,) IR 求解迭代次数（无 IR 时为 0）
    gain_chip: np.ndarray   # (T,N) 逐阵列逐列增益（chip_gain=False 时全为 1）


def simulate_crossbar(X: np.ndarray, bits: int, sigma: float, r_wire: float, *,
                      rng_arrays: np.random.Generator, rng_noise: np.random.Generator,
                      device: DeviceParams, ir: IRParams, noise: NoiseParams,
                      lut: np.ndarray | None = None, chunk_elems: int = 2 ** 24,
                      chip_gain: bool = False) -> CrossbarResult:
    """
    T 个独立阵列，每个阵列读 K 组输入。

    Args:
        X:          (T, K, N) 整数输入（第 t 个阵列读 X[t]）
        bits:       忆阻器 / DAC 位宽 b
        sigma:      相对写入误差 σ
        r_wire:     每段线电阻 (Ω)；0 = 无 IR drop
        rng_arrays: 器件编程随机流（权重 + 写入误差）
        rng_noise:  读出噪声随机流
        device:     DeviceParams
        ir:         IRParams（求解精度）
        noise:      NoiseParams
        lut:        理想权重 LUT；None 时用线性 LUT conductance_lut(bits, g_min, g_max)
        chunk_elems: 编程分块大小
        chip_gain:  是否另求每个阵列的逐列增益（*_chip 指标）
    """
    trials, K, n = X.shape
    if lut is None:
        lut = conductance_lut(bits, device.g_min, device.g_max)
    V = inputs_to_voltage(X, bits, device.v_read)
    A = np.concatenate([V, V * V], axis=1)       # 一次 GEMM 同时得到电流和 Σ V²G
    I_prog = np.empty((trials, K, n))
    I_ir = np.empty((trials, K, n))
    sd = np.zeros((trials, K, n))
    E_array = np.empty((trials, K))
    Y_true = np.empty((trials, K, n))
    iters = np.zeros(trials, dtype=int)
    gain_chip = np.ones((trials, n))
    for s, e, W, G, _ in iter_programmed_arrays(rng_arrays, trials, n, n, lut, sigma, chunk_elems):
        Y_true[s:e] = X[s:e] @ W
        R = A[s:e] @ G
        I_prog[s:e] = R[:, :K]
        if r_wire == 0:
            I_ir[s:e] = R[:, :K]
            E_array[s:e] = device.t_read * R[:, K:].sum(axis=2)
            if noise.enabled:
                cell_sq = cell_current_sq_ideal(V[s:e], G)
                for t in range(e - s):
                    sd[s + t] = read_noise_std(I_ir[s + t], cell_sq[t], G[t].sum(axis=0),
                                               t_read=device.t_read, **noise.model_kwargs())
        else:
            for t in range(e - s):
                ro = read_array_ir(G[t], V[s + t], r_wire, t_read=device.t_read, ir=ir, noise=noise,
                                   want_gain=chip_gain)
                I_ir[s + t] = ro.I
                iters[s + t] = ro.iters
                E_array[s + t] = ro.E
                if ro.noise_sd is not None:
                    sd[s + t] = ro.noise_sd
                if ro.col_gain is not None:
                    gain_chip[s + t] = ro.col_gain
                del ro
        del W, G, R
    I = add_read_noise(rng_noise, I_ir, sd) if noise.enabled else I_ir.copy()
    return CrossbarResult(I=I, I_prog=I_prog, I_ir=I_ir, noise_sd=sd, Y_true=Y_true, E_array=E_array,
                          iters=iters, gain_chip=gain_chip)
