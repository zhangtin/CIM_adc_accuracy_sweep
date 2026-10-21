# -*- coding: utf-8 -*-
"""自检：理想链路 MAC 反推精确、IR 求解器 vs 稀疏直接法、KCL、ADC、输入分布矩、权重 LUT。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .adc import quantize
from .config import NoiseParams, SweepConfig
from .constants import INPUT_DISTS
from .crossbar import simulate_crossbar
from .inputs import input_moments, sample_inputs
from .ir_drop import IRSolver, exact_solve
from .readout import current_to_mac
from .weights import conductance_lut, conductance_to_levels, program_arrays


def self_check(cfg: SweepConfig | None = None) -> None:
    cfg = cfg or SweepConfig()
    rng = np.random.default_rng(0)
    dev, inp = cfg.device, cfg.inputs

    # 0. LUT 正反映射
    for b in (1, 4, 8):
        lut = conductance_lut(b, dev.g_min, dev.g_max)
        assert lut[0] == dev.g_min and abs(lut[-1] / dev.g_max - 1) < 1e-12, "LUT 端点错误"
        k = np.arange(2 ** b)
        assert np.allclose(conductance_to_levels(lut, b, dev.g_min, dev.g_max), k), "LUT 逆映射错误"
        bt = program_arrays(rng, 2, 8, 8, lut, 0.0, keep_target=True)
        assert np.array_equal(bt.G, bt.G_target), "σ = 0 时实际电导应等于目标电导"

    # 1. 无 IR、无噪声：MAC 反推精确
    for b in (1, 4, 8):
        X = np.stack([sample_inputs(rng, d, b, (4, 64), **inp.as_kwargs()) for d in INPUT_DISTS], axis=1)
        sim = simulate_crossbar(X, b, 0.0, 0.0, rng_arrays=rng, rng_noise=rng, device=dev,
                                ir=replace(cfg.ir, r_wire=0.0), noise=NoiseParams(enabled=False),
                                chunk_elems=64 * 64)
        Y = current_to_mac(sim.I, X.sum(axis=2), b, g_min=dev.g_min, g_max=dev.g_max, v_read=dev.v_read)
        err = np.max(np.abs(Y - sim.Y_true))
        assert err < 1e-6, f"自检失败 b={b}: 最大误差 {err}"

    # 2. IR 求解器 vs 稀疏直接法；KCL
    for n, r in ((1, 1.0), (7, 2.0), (48, 1.0), (96, 0.5)):
        G = rng.uniform(1e-6, 100e-6, (n, n))
        V = rng.uniform(-0.2, 0.2, (n, 3))
        ex = exact_solve(G, V, r)
        res = IRSolver(G, r).solve(V, tol=1e-12, max_iter=2000)
        rel = np.abs(res.I - ex).max() / np.abs(ex).max()
        assert rel < 1e-8, f"IR 求解器自检失败 N={n}: {rel:.2e}"
        kcl = np.abs(res.I_wl.sum(axis=1) - res.I.sum(axis=1)).max() / np.abs(res.I).max()
        assert kcl < 1e-8, f"KCL 自检失败 N={n}: {kcl:.2e}"

    # 3. ADC / 输入分布
    I = np.linspace(0, 1, 1001)
    I_q, s_lo, s_hi = quantize(I, 0.0, 1.0, 4)
    assert not (s_lo | s_hi).any() and np.max(np.abs(I_q - I)) <= 0.5 / 15 + 1e-12, "ADC 自检失败"
    x = sample_inputs(rng, "sparse", 4, (200000,), **inp.as_kwargs())
    assert abs((x == 0).mean() - inp.sparsity) < 0.01, "稀疏分布自检失败"
    for d in INPUT_DISTS:
        xs = sample_inputs(rng, d, 4, (400000,), **inp.as_kwargs())
        m, m2 = input_moments(d, 4, **inp.as_kwargs())
        assert abs(xs.mean() / m - 1) < 0.01 and abs((xs ** 2).mean() / m2 - 1) < 0.01, f"矩自检失败 {d}"
    print("self-check passed", flush=True)


if __name__ == "__main__":
    self_check()
