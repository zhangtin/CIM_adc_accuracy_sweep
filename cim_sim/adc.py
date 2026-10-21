# -*- coding: utf-8 -*-
"""
ADC：满量程窗口 + 理想均匀量化（中点取整，超出量程削顶）。

满量程窗口
    max_window   [0, N·v_read·g_max]           理论最大列电流（理想阵列）
    stat_window  [max(0, μ − kσ), μ + kσ]      按校准统计量
"""
from __future__ import annotations

import numpy as np


def max_window(n_rows: int, v_read: float, g_max: float) -> tuple[float, float]:
    """
    理论最大电流量程。

    Args:
        n_rows: 参与求和的行数 N
        v_read: 满量程读电压 (V)
        g_max:  最大电导 (S)
    """
    return 0.0, n_rows * v_read * g_max


def stat_window(mu: float, sd: float, k: float = 4.0) -> tuple[float, float]:
    """
    统计量程 [max(0, μ − kσ), μ + kσ] (A)。

    Args:
        mu, sd: 列电流均值、标准差 (A)（calibration.stat_calibration 的输出）
        k:      覆盖倍数
    """
    return max(0.0, mu - k * sd), mu + k * sd


def adc_window(mode: str, n_rows: int, *, v_read: float, g_max: float,
               stat: tuple[float, float] | None = None, stat_k: float = 4.0) -> tuple[float, float]:
    """
    按 fs_mode 取 ADC 量程。

    Args:
        mode:   "max" 或 "stat"
        n_rows: 行数 N
        stat:   (μ, σ)，mode = "stat" 时必需
    """
    if mode == "max":
        return max_window(n_rows, v_read, g_max)
    if mode == "stat":
        if stat is None:
            raise ValueError("stat 量程需要校准统计量 (μ, σ)")
        return stat_window(*stat, k=stat_k)
    raise ValueError(f"未知 fs_mode: {mode}")


def adc_lsb(lo: float, hi: float, bits: int) -> float:
    """LSB 大小 (A)。"""
    return (hi - lo) / (2 ** bits - 1)


def quantize(I: np.ndarray, lo: float, hi: float, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    理想 ADC。

    Args:
        I:      模拟电流 (A)，任意形状
        lo, hi: 量程 (A)；码 0 ↔ lo，码 2^B − 1 ↔ hi
        bits:   ADC 位宽 B

    Returns:
        I_q:    量化后重建电流 (A)
        sat_lo: 低于量程的布尔掩码
        sat_hi: 高于量程的布尔掩码
    """
    n_codes = 2 ** bits
    lsb = (hi - lo) / (n_codes - 1)
    raw = np.rint((I - lo) / lsb)
    sat_lo, sat_hi = raw < 0, raw > n_codes - 1
    return lo + np.clip(raw, 0, n_codes - 1) * lsb, sat_lo, sat_hi
