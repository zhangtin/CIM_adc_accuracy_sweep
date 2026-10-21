# -*- coding: utf-8 -*-
"""
输入向量与 DAC。

输入为 b 位无符号整数 x ∈ {0 … 2^b − 1}（DAC 位宽 = 忆阻器位宽 b），
DAC 线性映射到读电压 V = x · v_read / (2^b − 1)。

三种分布：
    uniform  均匀
    sparse   比例 sparsity 为 0，其余在 1 … 2^b − 1 均匀（ReLU 后的激活）
    skewed   Beta(skew_a, skew_b) 量化（偏向小值）
"""
from __future__ import annotations

import numpy as np

from .constants import INPUT_DISTS


def dac_step(bits: int, v_read: float) -> float:
    """DAC 每个码对应的电压 (V)。"""
    return v_read / (2 ** bits - 1)


def inputs_to_voltage(X: np.ndarray, bits: int, v_read: float) -> np.ndarray:
    """
    整数输入 → 读电压。

    Args:
        X:      整数输入（任意形状）
        bits:   DAC 位宽 b
        v_read: 满量程读电压 (V)
    """
    return X * dac_step(bits, v_read)


def sample_inputs(rng: np.random.Generator, dist: str, bits: int, shape: tuple, *,
                  sparsity: float = 0.8, skew_a: float = 0.5, skew_b: float = 3.0) -> np.ndarray:
    """
    取样整数输入向量。

    Args:
        rng:      随机流
        dist:     "uniform" / "sparse" / "skewed"
        bits:     输入位宽 b
        shape:    输出形状，最后一维通常是行数 N
        sparsity, skew_a, skew_b: 分布参数（见模块说明）

    Returns:
        float64 数组（整数值）
    """
    levels = 2 ** bits - 1
    if dist == "uniform":
        return rng.integers(0, levels + 1, size=shape).astype(np.float64)
    if dist == "sparse":
        x = rng.integers(1, levels + 1, size=shape).astype(np.float64)
        x[rng.random(shape) < sparsity] = 0.0
        return x
    if dist == "skewed":
        return np.rint(rng.beta(skew_a, skew_b, size=shape) * levels)
    raise ValueError(f"未知输入分布: {dist}（可选 {INPUT_DISTS}）")


def input_pmf(dist: str, bits: int, *, sparsity: float = 0.8, skew_a: float = 0.5,
              skew_b: float = 3.0) -> np.ndarray:
    """输入整数 X 的概率质量函数 p[k] = P(X = k)，k = 0 … 2^b − 1。"""
    L = 2 ** bits - 1
    if dist == "uniform":
        return np.full(L + 1, 1.0 / (L + 1))
    if dist == "sparse":
        return np.r_[sparsity, np.full(L, (1 - sparsity) / L)]
    if dist == "skewed":
        from scipy.stats import beta
        edges = np.clip((np.arange(L + 2) - 0.5) / L, 0, 1)
        return np.diff(beta.cdf(edges, skew_a, skew_b))
    raise ValueError(f"未知输入分布: {dist}（可选 {INPUT_DISTS}）")


def input_moments(dist: str, bits: int, *, sparsity: float = 0.8, skew_a: float = 0.5,
                  skew_b: float = 3.0) -> tuple[float, float]:
    """
    输入整数 X 的解析矩。

    Returns:
        (E[X], E[X²])
    """
    k = np.arange(2 ** bits, dtype=np.float64)
    p = input_pmf(dist, bits, sparsity=sparsity, skew_a=skew_a, skew_b=skew_b)
    return float((p * k).sum()), float((p * k * k).sum())


def mean_level_pattern(dist: str, bits: int, n_rows: int, *, sparsity: float = 0.8,
                       skew_a: float = 0.5, skew_b: float = 3.0) -> np.ndarray:
    """
    长度 n_rows 的固定输入：floor/ceil(E[X]) 按比例混合，使平均值等于 E[X]。
    （under-utilization 实验中驱动空闲行用）
    """
    ex, _ = input_moments(dist, bits, sparsity=sparsity, skew_a=skew_a, skew_b=skew_b)
    lo = np.floor(ex)
    n_up = int(round((ex - lo) * n_rows))
    x = np.full(n_rows, lo)
    x[:n_up] = lo + 1
    return x
