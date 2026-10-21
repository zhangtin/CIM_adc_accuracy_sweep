# -*- coding: utf-8 -*-
"""
理想权重 LUT：整数权重电平 → 目标电导。

忆阻器存 b 位无符号整数权重 w ∈ {0, 1, …, 2^b − 1}，线性映射到电导
    G_target(w) = g_min + w · ΔG,     ΔG = (g_max − g_min) / (2^b − 1)
LUT 就是长度为 2^b 的数组 lut[w] = G_target(w)。

把映射做成显式 LUT 的好处：后续若要换成非线性电平分配、差分对、实测电平表等，
只需要传入另一张 LUT，写入噪声和阵列赋值代码无需改动。
"""
from __future__ import annotations

import numpy as np


def n_levels(bits: int) -> int:
    """电平数 2^b。"""
    return 2 ** bits


def max_level(bits: int) -> int:
    """最大整数权重 2^b − 1。"""
    return 2 ** bits - 1


def conductance_step(bits: int, g_min: float, g_max: float) -> float:
    """
    相邻电平的电导间隔 ΔG (S)。

    Args:
        bits:  权重位宽 b
        g_min: 最小电导 (S)
        g_max: 最大电导 (S)
    """
    return (g_max - g_min) / max_level(bits)


def conductance_lut(bits: int, g_min: float, g_max: float) -> np.ndarray:
    """
    线性理想权重 LUT。

    Args:
        bits:  权重位宽 b
        g_min: 权重 0 对应的电导 (S)
        g_max: 权重 2^b − 1 对应的电导 (S)

    Returns:
        lut: (2^b,) float64，lut[w] = g_min + w·ΔG
    """
    return np.arange(n_levels(bits), dtype=np.float64) * conductance_step(bits, g_min, g_max) + g_min


def levels_to_conductance(W: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    整数权重 → 目标电导（查表）。

    Args:
        W:   任意形状的整数权重（int 或取整数值的 float），取值 0 … len(lut) − 1
        lut: conductance_lut() 或自定义 LUT

    Returns:
        G_target: 与 W 同形状的 float64 电导 (S)
    """
    return lut[np.asarray(W, dtype=np.int64)]


def conductance_to_levels(G: np.ndarray, bits: int, g_min: float, g_max: float,
                          round_to_int: bool = False) -> np.ndarray:
    """
    电导 → 等效（连续）权重，线性 LUT 的逆映射；round_to_int=True 时取最近电平。

    Args:
        G:     电导 (S)
        bits, g_min, g_max: 与 conductance_lut 相同
    """
    w = (np.asarray(G) - g_min) / conductance_step(bits, g_min, g_max)
    return np.rint(w) if round_to_int else w
