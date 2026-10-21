# -*- coding: utf-8 -*-
"""
随机流约定（集中定义，保证不同实验之间、拆分前后结果逐位一致）。

每个扫描点 (b, σ, N) 由 SeedSequence([seed, b, round(σ·1e6), N]) 派生 5 条独立随机流：
    arrays      权重电平 + 写入噪声（器件编程）
    inputs      输入向量
    cal_arrays  stat 校准用的独立阵列
    cal_inputs  抽样校准的输入（cal_mode = sampled）
    noise       读出噪声
同一 (b, σ, N) 下，所有实验（主扫描、under-utilization 等）拿到的权重 / 器件误差 / 输入完全相同。
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


def sigma_key(sigma: float) -> int:
    """σ 在种子中的整数编码（ppm）。"""
    return int(round(sigma * 1e6))


class PointStreams(NamedTuple):
    arrays: np.random.Generator
    inputs: np.random.Generator
    cal_arrays: np.random.Generator
    cal_inputs: np.random.Generator
    noise: np.random.Generator


def point_streams(seed: int, bits: int, sigma: float, n: int) -> PointStreams:
    """
    扫描点 (bits, sigma, n) 的全部随机流。

    Args:
        seed:  全局种子
        bits:  忆阻器 / DAC 位宽 b
        sigma: 相对写入误差 σ
        n:     阵列规模（N×N）
    """
    ss = np.random.SeedSequence([seed, bits, sigma_key(sigma), n])
    arr, inp, cal_arr, cal_in = (np.random.default_rng(s) for s in ss.spawn(4))
    noise = np.random.default_rng(ss.spawn(1)[0])
    return PointStreams(arr, inp, cal_arr, cal_in, noise)


def probe_stream(seed: int, n: int, bits: int, sigma: float, array_index: int) -> np.random.Generator:
    """stat 校准中第 array_index 个校准阵列的 Rademacher 探针随机流。"""
    return np.random.default_rng([seed, 4242, n, bits, sigma_key(sigma), array_index])


def reference_stream(seed: int, n: int) -> np.random.Generator:
    """IR drop 标定参考工况的随机流。"""
    return np.random.default_rng([seed, 777, n])


def dummy_rows_stream(seed: int, bits: int, sigma: float, n: int, n_hw: int) -> np.random.Generator:
    """under-utilization 实验中空闲行（dummy 行）阵列的随机流。"""
    return np.random.default_rng(np.random.SeedSequence([seed, bits, sigma_key(sigma), n, n_hw, 99]))
