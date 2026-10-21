# -*- coding: utf-8 -*-
"""
等效线电阻标定：对每个 N 找 r_eff(N)，使参考工况（均匀 4-bit 权重、均匀输入、σ = 0）
下阵列平均信号衰减等于
    A(N) = atten_lo · (N / n_lo)^α，α 由 A(n_lo) = atten_lo、A(n_hi) = atten_hi 决定
（默认 A(32) = 1 %，A(4096) = 10 %，α ≈ 0.47）。
"""
from __future__ import annotations

import numpy as np

from ..config import DeviceParams, IRParams
from ..rng import reference_stream
from .solver import IRSolver


def target_attenuation(n: int, atten_lo: float = 0.01, atten_hi: float = 0.10,
                       n_lo: int = 32, n_hi: int = 4096) -> float:
    """
    目标平均衰减 A(n)。

    Args:
        n: 阵列规模
        atten_lo, atten_hi: n_lo、n_hi 处的衰减（0.01 = 1 %）
    """
    alpha = np.log(atten_hi / atten_lo) / np.log(n_hi / n_lo)
    return atten_lo * (n / n_lo) ** alpha


def reference_case(n: int, seed: int, g_min: float, g_max: float, v_read: float,
                   bits: int = 4, n_vectors: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    标定用参考阵列与输入。

    注意：电导按 g_min + W·(g_max − g_min)/L 计算（与原 v3 保持逐位一致，
    故不经过 weights.lut）。

    Returns:
        G: (n, n) 电导 (S)；V: (n, n_vectors) 驱动电压 (V)
    """
    rng = reference_stream(seed, n)
    L = 2 ** bits - 1
    W = rng.integers(0, L + 1, size=(n, n))
    G = g_min + W * (g_max - g_min) / L
    V = rng.integers(0, L + 1, size=(n, n_vectors)) * (v_read / L)
    return G, V


def measure_attenuation(G: np.ndarray, V: np.ndarray, r_wire: float, tol: float = 1e-6,
                        max_iter: int = 500) -> float:
    """
    阵列平均信号衰减 1 − ΣI_IR / ΣI_ideal。

    Args:
        G: (N, N) 电导；V: (N, K) 输入电压；r_wire: 每段线电阻 (Ω)
    """
    I = IRSolver(G, r_wire).solve(V, tol=tol, max_iter=max_iter).I
    return 1.0 - I.sum() / (V.T @ G).sum()


def calibrate_r_wire(n: int, ir: IRParams, device: DeviceParams, seed: int) -> dict:
    """
    求规模 n 的等效每段线电阻。

    Args:
        n:      阵列规模
        ir:     IRParams；ir.r_wire < 0 按 A(N) 标定，= 0 关闭，> 0 直接使用该值
        device: DeviceParams（g_min, g_max, v_read）
        seed:   全局种子

    Returns:
        dict(N, r_eff_ohm, target_atten, ref_atten)
    """
    if ir.r_wire == 0:
        return dict(N=n, r_eff_ohm=0.0, target_atten=0.0, ref_atten=0.0)
    G, V = reference_case(n, seed, device.g_min, device.g_max, device.v_read)
    if ir.r_wire > 0:
        r = ir.r_wire
        return dict(N=n, r_eff_ohm=r, target_atten=np.nan,
                    ref_atten=measure_attenuation(G, V, r, ir.tol, ir.max_iter))
    tgt = target_attenuation(n, ir.atten_lo, ir.atten_hi, ir.n_lo, ir.n_hi)
    # 小压降区衰减 ≈ 线性于 r：比例迭代 + 对数空间割线法
    g = float(G.mean())
    r = tgt / (g * n * n / 1.5)
    hist = []
    for _ in range(30):
        a = measure_attenuation(G, V, r, ir.tol, ir.max_iter)
        hist.append((np.log(r), np.log(a)))
        if abs(a / tgt - 1) < 1e-4:
            break
        if len(hist) >= 2:
            (x0, y0), (x1, y1) = hist[-2], hist[-1]
            slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 1.0
            r = float(np.exp(x1 + (np.log(tgt) - y1) / max(slope, 0.05)))
        else:
            r = r * tgt / a
    return dict(N=n, r_eff_ohm=r, target_atten=tgt, ref_atten=a)


def reference_attenuation_at(n: int, r_wire: float, ir: IRParams, device: DeviceParams, seed: int) -> float:
    """参考工况在给定线电阻下的衰减（ir_calibration.csv 中 ref_atten_1ohm 列用 r_wire = 1 Ω）。"""
    G, V = reference_case(n, seed, device.g_min, device.g_max, device.v_read)
    return measure_attenuation(G, V, r_wire, ir.tol, ir.max_iter)
