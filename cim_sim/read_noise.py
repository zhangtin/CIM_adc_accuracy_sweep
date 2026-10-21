# -*- coding: utf-8 -*-
"""
读出噪声（每次读出独立）。列电流噪声方差为以下各项之和：

    器件读噪声    read_noise_rel² · Σ_i I_cell,ij²
    热噪声        4kT · G_col · Δf
    散粒噪声      2q · |I_col| · Δf
    TIA 输入噪声  tia_en² · G_col² · Δf + tia_in² · Δf    （电压噪声经列电导放大）
    Δf = 1 / (2 · t_read)
"""
from __future__ import annotations

import numpy as np

from .constants import K_BOLTZ, Q_E


def noise_bandwidth(t_read: float) -> float:
    """等效噪声带宽 Δf = 1/(2·t_read) (Hz)。"""
    return 1.0 / (2.0 * t_read)


def read_noise_std(I_col: np.ndarray, cell_sq: np.ndarray, G_col: np.ndarray, *, t_read: float,
                   read_noise_rel: float = 0.01, temp_K: float = 300.0, tia_en: float = 10e-9,
                   tia_in: float = 1e-12) -> np.ndarray:
    """
    列电流噪声标准差 (A)。

    Args:
        I_col:   (..., N) 列电流 (A)
        cell_sq: (..., N) Σ_i I_cell,ij²，单元电流平方和 (A²)
        G_col:   (N,) 或可广播形状，列总电导 Σ_i G_ij (S)
        t_read:  读出时间 (s)
        read_noise_rel, temp_K, tia_en, tia_in: 见模块说明（可用 NoiseParams.model_kwargs() 展开）

    Returns:
        与 I_col 同形状的标准差
    """
    df = noise_bandwidth(t_read)
    var = (read_noise_rel ** 2) * cell_sq \
        + 4 * K_BOLTZ * temp_K * G_col * df \
        + 2 * Q_E * np.abs(I_col) * df \
        + (tia_en ** 2) * G_col ** 2 * df + (tia_in ** 2) * df
    return np.sqrt(var)


def cell_current_sq_ideal(V: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    无 IR drop 时 Σ_i (V_i·G_ij)²。

    Args:
        V: (..., K, N_rows) 输入电压；G: (..., N_rows, N_cols) 电导

    Returns:
        (..., K, N_cols)
    """
    return (V ** 2) @ (G ** 2)


def cell_current_sq_from_fields(u: np.ndarray, w: np.ndarray, G: np.ndarray, overwrite_u: bool = False) -> np.ndarray:
    """
    有 IR drop 时由节点电压求 Σ_i ((u_ij − w_ij)·G_ij)²。

    Args:
        u, w:        (N, N, K) WL / BL 节点电压（IRSolver.solve(return_fields=True)）
        G:           (N, N) 电导
        overwrite_u: True 时就地使用 u 的内存（省一份 N²K 内存，u 之后不可再用）

    Returns:
        (K, N)
    """
    c = u if overwrite_u else u.copy()
    c -= w
    c *= G[:, :, None]
    c *= c
    return c.sum(axis=0).T


def add_read_noise(rng: np.random.Generator, I_clean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """
    I = I_clean + z·sd，z ~ N(0, 1) 独立。

    Args:
        rng:     读出噪声随机流
        I_clean: 无噪声电流 (A)
        sd:      与 I_clean 同形状的噪声标准差 (A)
    """
    I = I_clean.copy()
    z = rng.standard_normal(I.shape)
    I += z * sd
    return I
