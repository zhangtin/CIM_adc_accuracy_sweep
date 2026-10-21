# -*- coding: utf-8 -*-
"""
实际忆阻器矩阵赋值：随机权重 → 查 LUT 得目标电导 → 叠加写入噪声。

随机流使用顺序（与原 v2/v3 完全一致）：每个分块先取整数权重，再（σ > 0 时）取写入误差。
"""
from __future__ import annotations

from typing import Iterator, NamedTuple

import numpy as np

from .lut import levels_to_conductance
from .write_noise import apply_write_noise, sample_write_noise


class ArrayBatch(NamedTuple):
    """一批已编程阵列。"""
    start: int                      # 本批在全部阵列中的起始下标
    stop: int                       # 结束下标（不含）
    W: np.ndarray                   # (k, n_rows, n_cols) float64 整数权重
    G: np.ndarray                   # (k, n_rows, n_cols) 实际电导（含写入噪声）(S)
    G_target: np.ndarray | None     # (k, n_rows, n_cols) 目标电导；keep_target=False 时为 None


def sample_weight_levels(rng: np.random.Generator, shape: tuple, n_levels: int) -> np.ndarray:
    """
    均匀随机整数权重。

    Args:
        rng:      随机流
        shape:    输出形状
        n_levels: 电平数 2^b（权重取 0 … n_levels − 1）

    Returns:
        W: float64 数组（整数值）
    """
    return rng.integers(0, n_levels, size=shape).astype(np.float64)


def program_arrays(rng: np.random.Generator, n_arrays: int, n_rows: int, n_cols: int,
                   lut: np.ndarray, sigma: float, W: np.ndarray | None = None,
                   keep_target: bool = False) -> ArrayBatch:
    """
    编程一批阵列。

    Args:
        rng:         随机流（权重与写入误差都从这里取）
        n_arrays:    阵列个数
        n_rows:      行数（字线 / 输入）
        n_cols:      列数（位线 / 输出）
        lut:         理想权重 LUT（weights.lut.conductance_lut）
        sigma:       相对写入误差 σ；0 表示理想写入，此时不消耗写入噪声随机数
        W:           可选，外部给定的整数权重 (n_arrays, n_rows, n_cols)；给定时不从 rng 取权重
                     （例如把训练好的网络权重写入阵列）
        keep_target: 是否同时返回目标电导（多占一份内存）

    Returns:
        ArrayBatch(start=0, stop=n_arrays, W, G, G_target)
    """
    shape = (n_arrays, n_rows, n_cols)
    if W is None:
        W = sample_weight_levels(rng, shape, len(lut))
    elif W.shape != shape:
        raise ValueError(f"W 形状 {W.shape} 与 {shape} 不符")
    G = levels_to_conductance(W, lut)
    G_target = G.copy() if keep_target else None
    if sigma > 0:
        factor = sample_write_noise(rng, shape, sigma)
        apply_write_noise(G, factor, inplace=True)
        del factor
    return ArrayBatch(0, n_arrays, W, G, G_target)


def iter_programmed_arrays(rng: np.random.Generator, n_arrays: int, n_rows: int, n_cols: int,
                           lut: np.ndarray, sigma: float, chunk_elems: int = 2 ** 24,
                           keep_target: bool = False) -> Iterator[ArrayBatch]:
    """
    分块编程 n_arrays 个阵列（控制内存）。每块最多 chunk_elems 个单元（至少 1 个阵列）。

    Args:
        rng, n_rows, n_cols, lut, sigma, keep_target: 同 program_arrays
        n_arrays:    阵列总数（Monte Carlo 次数）
        chunk_elems: 每块单元数上限

    Yields:
        ArrayBatch，其中 start/stop 为本块在全部阵列中的下标范围
    """
    step = max(1, chunk_elems // (n_rows * n_cols))
    for s in range(0, n_arrays, step):
        e = min(n_arrays, s + step)
        b = program_arrays(rng, e - s, n_rows, n_cols, lut, sigma, keep_target=keep_target)
        yield ArrayBatch(s, e, b.W, b.G, b.G_target)
