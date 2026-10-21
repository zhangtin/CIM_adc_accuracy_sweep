# -*- coding: utf-8 -*-
"""
数字域读出：列电流 → 整数 MAC 结果，以及（可选）逐列数字增益校正。

由 I_j = k·Σ_i x_i·(g_min + w_ij·ΔG)，k = v_read/(2^b − 1)，得
    Y_j = Σ_i x_i·w_ij = (I_j / k − g_min·Σ_i x_i) / ΔG
其中 Σ_i x_i 在数字域已知（输入本来就是数字量）。
"""
from __future__ import annotations

import numpy as np

from .inputs import dac_step
from .weights import conductance_step


def current_to_mac(I: np.ndarray, sum_x: np.ndarray, bits: int, *, g_min: float, g_max: float,
                   v_read: float) -> np.ndarray:
    """
    Args:
        I:      (..., N) 列电流 (A)
        sum_x:  (...)    每组输入的 Σ_i x_i
        bits:   忆阻器 / DAC 位宽 b
        g_min, g_max, v_read: 器件参数（可用 DeviceParams.as_kwargs(exclude=("t_read",)) 展开）

    Returns:
        (..., N) 等效整数 MAC 结果（浮点）
    """
    k = dac_step(bits, v_read)
    dG = conductance_step(bits, g_min, g_max)
    return (I / k - g_min * sum_x[..., None]) / dG


def apply_column_gain(I: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """逐列数字增益校正 I / g_j（gain 可广播到 I 的最后一维）。"""
    return I / gain
