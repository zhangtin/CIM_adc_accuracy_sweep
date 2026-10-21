# -*- coding: utf-8 -*-
"""
写入（编程）噪声。

模型：相对高斯误差
    G_actual = max(G_target · (1 + σ·z), 0),   z ~ N(0, 1)，各单元独立
z 以 float32 生成（与原 v2/v3 完全一致，也节省大阵列内存）。

拆成两步：
    sample_write_noise  只负责从随机流取样，返回乘性因子 (1 + σ·z)
    apply_write_noise   把因子作用到目标电导上并截断负值
这样可以单独替换噪声模型（例如电平相关 σ、对数正态等），或复用同一组因子。
"""
from __future__ import annotations

import numpy as np


def sample_write_noise(rng: np.random.Generator, shape: tuple, sigma: float,
                       dtype=np.float32) -> np.ndarray:
    """
    取样乘性写入误差因子 (1 + σ·z)。

    注意：调用一次就会消耗 rng；σ = 0 时调用方应跳过本函数（原代码在 σ = 0 时不取样，
    保证随机流位置不变）。

    Args:
        rng:   随机流
        shape: 输出形状，一般为 (n_arrays, n_rows, n_cols)
        sigma: 相对标准差（0.02 = 2 %）
        dtype: 因子的数据类型，默认 float32

    Returns:
        factor: shape 形状数组，元素 = 1 + σ·z
    """
    factor = rng.standard_normal(shape, dtype=dtype)
    factor *= dtype(sigma)
    factor += dtype(1.0)
    return factor


def apply_write_noise(G_target: np.ndarray, factor: np.ndarray, g_floor: float = 0.0,
                      inplace: bool = False) -> np.ndarray:
    """
    G_actual = max(G_target · factor, g_floor)。

    Args:
        G_target: 目标电导 (S)
        factor:   sample_write_noise() 的输出（可广播到 G_target）
        g_floor:  电导下限 (S)，默认 0（电导不可能为负）
        inplace:  True 时直接改写 G_target 以省内存

    Returns:
        G_actual: 与 G_target 同形状 float64
    """
    G = G_target if inplace else G_target.copy()
    G *= factor
    np.maximum(G, g_floor, out=G)
    return G
