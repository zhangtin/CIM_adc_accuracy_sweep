#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cim_adc_mc.py — 忆阻器 crossbar 模拟 MAC + ADC 量化的 Monte Carlo 扫描  (v3：加入电路级非理想)

v3 相对 v2 的变化
    1. IR drop：完整的线电阻节点网络（WL 从一端驱动、BL 在另一端由 TIA 虚地读出），
       用 block-SOR + 逐线精确三对角求解（C/OpenMP 内核，首次运行自动编译），
       N ≤ 256 时与稀疏直接法逐位一致（自检）。
       线电阻取"常见量级"标定：对每个 N 选取等效每段线电阻 r_eff(N)，使参考工况
       （均匀权重、均匀输入、σ=0）下阵列平均信号衰减 = A(N)，
           A(N) = ir_atten_lo · (N / N_lo)^α，A(32) = 1 %，A(4096) = 10 %（α ≈ 0.47，约 ∝ √N），
       代表已做过 IR 缓解（加宽/加厚金属、分段驱动等）的实际宏单元。
       其他输入分布 / 位宽 / 器件误差下的实际衰减由网络方程决定，不再强制。
       （--r-wire R 可改为固定物理线电阻；--r-wire 0 关闭 IR drop。）
    2. 噪声（每次读出独立，按物理量级，随 N 自然增大）：
         器件读噪声    每个单元电流相对 σ_read（默认 1 %），列上 ∝ √(Σ I_cell²)
         热噪声        4kT·G_col·Δf，G_col = 该列总电导
         散粒噪声      2q·I_col·Δf
         TIA 输入噪声  e_n²·G_col²·Δf + i_n²·Δf（电压噪声经列电导放大，∝ N）
         Δf = 1/(2·t_read)
    3. 阵列读能量改为驱动器实际输出能量 t_read·Σ V_i·I_WL,i（含线损）。
    4. stat 量程校准改为半解析：由校准阵列的有效传输矩阵 T（N ≤ 256 精确求出，
       更大 N 用 32 个 Rademacher 探针估计 Σ_i T_ij²）和输入分布的解析矩计算列电流的 μ、σ
       （再加噪声方差），与 v2 的抽样估计期望相同、方差更小；带 IR 求解时抽样法代价过高。
       （--cal-mode sampled 恢复 v2 抽样校准，仅在 IR 关闭时可用。）
    5. 读出校正：IR drop 主要表现为随列位置变化的系统性增益衰减。除原始指标外，另给出
       "逐列数字增益校正"后的指标（*_cal）：校正系数 g_j = 校准阵列上全 1 输入的 IR 响应 / 理想
       列电导和（跨 n_cal_arrays 个独立阵列平均），数字域 Y ← Y(I/g_j)。只校正位置相关的平均增益，
       阵列间权重差异、行位置相关的残差与噪声不被校正。knee.csv 同时给出 knee_adc_bits_cal。
    其余扫描变量、器件 / 电路常数、ADC 模型、能量表、随机种子与阵列生成顺序均与 v2 相同：
    同一 (b, σ, N) 下 v3 与 v2 使用完全相同的权重、器件误差和输入向量。

扫描变量（全组合）
    mem_bits    忆阻器数据位宽 b（DAC 位宽 = b，输入为 b 位无符号整数）
    mem_sigma   忆阻器量化误差：相对目标电导的百分比，G = G_target·(1 + N(0, σ))
    adc_bits    ADC 位宽 B
    N           crossbar 规模（N×N 方阵，每列一个 ADC），默认 32 … 4096
    fs_mode     ADC 满量程："max"（理论最大电流，理想阵列）/ "stat"（按同分布输入统计校准 μ±kσ，超出削顶）
    input_dist  uniform / sparse / skewed（见 v2）

输出（--out 目录）
    raw_trials.csv.gz / summary.csv / knee.csv / config.json / ir_calibration.csv / fig_*.png

用法
    python cim_adc_mc.py --energy-csv ADC_22nm_energy_vs_bits.csv --out results_v3
    python cim_adc_mc.py --r-wire 0 --no-noise --out results_ideal       # 关闭非理想（v2 等价）
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import json
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ENERGY_TIERS = ("best", "typical", "conservative")
INPUT_DISTS = ("uniform", "sparse", "skewed")
K_BOLTZ = 1.380649e-23
Q_E = 1.602176634e-19


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # ---- 扫描变量（同 v2）----
    mem_bits: list = field(default_factory=lambda: [1, 2, 3, 4, 6, 8])
    mem_sigma: list = field(default_factory=lambda: [0.0, 0.01, 0.02, 0.05])
    adc_bits: list = field(default_factory=lambda: [4, 6, 8, 10, 12])
    sizes: list = field(default_factory=lambda: [32, 64, 128, 256, 512, 1024, 2048, 4096])
    fs_modes: list = field(default_factory=lambda: ["max", "stat"])
    input_dists: list = field(default_factory=lambda: list(INPUT_DISTS))
    n_mc: int = 100

    # ---- 输入分布参数（同 v2）----
    sparsity: float = 0.8
    skew_a: float = 0.5
    skew_b: float = 3.0

    # ---- 器件 / 电路常数（同 v2）----
    g_min: float = 1e-6
    g_max: float = 100e-6
    v_read: float = 0.2
    t_read: float = 10e-9

    # ---- v3：IR drop ----
    r_wire: float = -1.0            # <0：按 A(N) 标定等效线电阻；0：关闭；>0：固定物理线电阻（Ω/段）
    ir_atten_lo: float = 0.01       # A(N_lo)
    ir_atten_hi: float = 0.10       # A(N_hi)
    ir_n_lo: int = 32
    ir_n_hi: int = 4096
    ir_tol: float = 1e-6            # SOR 相对收敛判据
    ir_max_iter: int = 500

    # ---- v3：噪声 ----
    chip_cal: bool = False          # 另算"逐芯片逐列增益校正"指标（*_chip，需额外一次 IR 求解/阵列）
    noise: bool = True
    read_noise_rel: float = 0.01    # 单元读噪声（相对单元电流）
    temp_K: float = 300.0
    tia_en: float = 10e-9           # V/√Hz
    tia_in: float = 1e-12           # A/√Hz

    # ---- ADC 满量程统计校准 ----
    stat_k: float = 4.0
    n_cal_arrays: int = 4
    n_cal_inputs: int = 100         # 仅 cal_mode = sampled 使用
    cal_mode: str = "analytic"      # analytic（v3 默认）/ sampled（v2，IR 关闭时）
    cal_probes: int = 32            # N > cal_exact_max 时的 Rademacher 探针数
    cal_exact_max: int = 256

    # ---- 其他（同 v2）----
    # Resolve bundled data independently of the caller's working directory.
    energy_csv: str = str(Path(__file__).resolve().parent / "ADC_22nm_energy_vs_bits.csv")
    seed: int = 20260929
    out_dir: str = "results"
    chunk_elems: int = 2 ** 24
    plot_sigma: float = 0.02
    plot_sizes: list = field(default_factory=lambda: [32, 256, 1024, 4096])
    plot_mem_bits: int = 4
    knee_tol: float = 0.25


# ---------------------------------------------------------------------------
# C 内核：带线电阻 crossbar 的 block-SOR 求解
# ---------------------------------------------------------------------------
_C_SRC = r"""
// 拓扑：G[i,j]，i = 行（WL，输入），j = 列（BL，输出）
//   WL i：驱动器 V_i —r— (i,0) —r— (i,1) … (i,N-1)（末端开路）
//   BL j：(0,j)（顶端开路）—r— … (N-1,j) —r— TIA 虚地（0 V）；输出 I_j = w[N-1,j]/r
// 布局：G, mwl, mbl : (N,N)；u, w, tmp : (N,N,K)；V : (N,K)
#include <math.h>
#include <stddef.h>

void ir_factor(const double *G, long N, double r, double *mwl, double *mbl)
{
    const double gr = 1.0 / r, gr2 = gr * gr;
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
        const double *g = G + i * N; double *m = mwl + i * N; double prev = 0.0;
        for (long j = 0; j < N; j++) {
            double diag = g[j] + gr + (j < N - 1 ? gr : 0.0);
            prev = 1.0 / (diag - (j > 0 ? gr2 * prev : 0.0));
            m[j] = prev;
        }
    }
    for (long j = 0; j < N; j++) mbl[j] = 1.0 / (G[j] + gr);
    for (long i = 1; i < N; i++) {
        const double *g = G + i * N, *mp = mbl + (i - 1) * N; double *m = mbl + i * N;
        #pragma omp parallel for schedule(static)
        for (long j = 0; j < N; j++) m[j] = 1.0 / (g[j] + 2.0 * gr - gr2 * mp[j]);
    }
}

void ir_sor_iter(const double *G, const double *mwl, const double *mbl, const double *V,
                 double *u, double *w, double *tmp, long N, long K, double r, double omega, double *out)
{
    const double gr = 1.0 / r;
    double du_max = 0.0, dw_max = 0.0, dI_max = 0.0;
    #pragma omp parallel for schedule(static) reduction(max:du_max)
    for (long i = 0; i < N; i++) {
        const double *g = G + i * N, *m = mwl + i * N;
        double *dp = tmp + i * N * K; const double *wr = w + i * N * K; double *ur = u + i * N * K;
        for (long k = 0; k < K; k++) dp[k] = (g[0] * wr[k] + V[i * K + k] * gr) * m[0];
        for (long j = 1; j < N; j++) {
            const double gj = g[j], mj = m[j]; const double *wj = wr + j * K, *dq = dp + (j - 1) * K;
            double *dj = dp + j * K;
            for (long k = 0; k < K; k++) dj[k] = (gj * wj[k] + dq[k] * gr) * mj;
        }
        double *xn = dp + (N - 1) * K;
        for (long k = 0; k < K; k++) {
            double *uu = ur + (N - 1) * K + k; double nu = (1.0 - omega) * (*uu) + omega * xn[k];
            double d = fabs(nu - *uu); if (d > du_max) du_max = d; *uu = nu;
        }
        for (long j = N - 2; j >= 0; j--) {
            const double c = m[j] * gr; double *dj = dp + j * K, *dn = dp + (j + 1) * K; double *uj = ur + j * K;
            for (long k = 0; k < K; k++) {
                double x = dj[k] + c * dn[k]; dj[k] = x;
                double nu = (1.0 - omega) * uj[k] + omega * x;
                double d = fabs(nu - uj[k]); if (d > du_max) du_max = d; uj[k] = nu;
            }
        }
    }
    for (long i = 0; i < N; i++) {
        const double *g = G + i * N, *m = mbl + i * N; const double *ui = u + i * N * K;
        double *di = tmp + i * N * K; const double *dq = (i > 0) ? tmp + (i - 1) * N * K : NULL;
        #pragma omp parallel for schedule(static)
        for (long j = 0; j < N; j++) {
            const double gj = g[j], mj = m[j];
            for (long k = 0; k < K; k++)
                di[j * K + k] = (gj * ui[j * K + k] + (dq ? dq[j * K + k] * gr : 0.0)) * mj;
        }
    }
    for (long i = N - 1; i >= 0; i--) {
        const double *m = mbl + i * N; double *di = tmp + i * N * K;
        const double *dn = (i < N - 1) ? tmp + (i + 1) * N * K : NULL; double *wi = w + i * N * K;
        double lw = 0.0, lI = 0.0;
        #pragma omp parallel for schedule(static) reduction(max:lw, lI)
        for (long j = 0; j < N; j++) {
            const double c = m[j] * gr;
            for (long k = 0; k < K; k++) {
                long idx = j * K + k; double x = di[idx] + (dn ? c * dn[idx] : 0.0); di[idx] = x;
                double nw = (1.0 - omega) * wi[idx] + omega * x; double d = fabs(nw - wi[idx]);
                if (d > lw) lw = d; if (i == N - 1 && d * gr > lI) lI = d * gr; wi[idx] = nw;
            }
        }
        if (lw > dw_max) dw_max = lw; if (lI > dI_max) dI_max = lI;
    }
    out[0] = du_max; out[1] = dw_max; out[2] = dI_max;
}
"""

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        tag = hashlib.sha1(_C_SRC.encode()).hexdigest()[:12]
        d = Path(tempfile.gettempdir()) / "cim_adc_mc_kernel"
        d.mkdir(exist_ok=True)
        so, src = d / f"crossbar_ir_{tag}.so", d / f"crossbar_ir_{tag}.c"
        if not so.exists():
            src.write_text(_C_SRC)
            subprocess.check_call(["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
                                   str(src), "-o", str(so)])
        lib = ctypes.CDLL(str(so))
        P = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        L, D = ctypes.c_long, ctypes.c_double
        lib.ir_factor.argtypes = [P, L, D, P, P]
        lib.ir_factor.restype = None
        lib.ir_sor_iter.argtypes = [P, P, P, P, P, P, P, L, L, D, D, P]
        lib.ir_sor_iter.restype = None
        _LIB = lib
    return _LIB


class IRSolver:
    """单个阵列（固定 G，Ω/段线电阻 r）在多组输入电压下的节点方程求解。"""

    def __init__(self, G: np.ndarray, r: float):
        self.G = np.ascontiguousarray(G, dtype=np.float64)
        self.N = self.G.shape[0]
        self.r = float(r)
        self.mwl = np.empty_like(self.G)
        self.mbl = np.empty_like(self.G)
        _lib().ir_factor(self.G, self.N, self.r, self.mwl, self.mbl)
        # 2-cyclic 系统 Young 最优松弛因子（以平均电导估计 block-Jacobi 谱半径）
        lam = (2.0 / r) * (1.0 - np.cos(np.pi / (2 * self.N + 1)))
        g = float(self.G.mean())
        mu2 = g * g / (g + lam) ** 2
        self.omega = 2.0 / (1.0 + np.sqrt(max(1.0 - mu2, 0.0)))

    def solve(self, V: np.ndarray, tol: float = 1e-6, max_iter: int = 500, fields: bool = False) -> dict:
        """V: (N,K)。返回 I (K,N) 输出列电流、I_wl (K,N) 行驱动电流、iters；fields=True 时附带 u, w。"""
        N, r = self.N, self.r
        V = np.ascontiguousarray(V, dtype=np.float64)
        K = V.shape[1]
        u = np.repeat(V[:, None, :], N, axis=1)
        w = np.zeros((N, N, K))
        tmp = np.empty((N, N, K))
        out = np.zeros(3)
        vmax = max(float(np.abs(V).max()), 1e-30)
        lib = _lib()
        it = 0
        for it in range(1, max_iter + 1):
            lib.ir_sor_iter(self.G, self.mwl, self.mbl, V, u, w, tmp, N, K, r, self.omega, out)
            imax = max(float(np.abs(w[N - 1]).max()) / r, 1e-30)
            if max(out[0], out[1]) < tol * vmax and out[2] < tol * imax:
                break
        else:
            print(f"  [warn] IR solver not converged in {max_iter} iters (N={N})", flush=True)
        del tmp
        res = dict(I=(w[N - 1] / r).T.copy(), I_wl=((V - u[:, 0, :]) / r).T.copy(), iters=it)
        if fields:
            res["u"], res["w"] = u, w
        return res


def exact_solve(G: np.ndarray, V: np.ndarray, r: float) -> np.ndarray:
    """稀疏直接法求解完整 2N² 节点方程（自检参考解）。V: (N,K)，返回 (K,N)。"""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    N = G.shape[0]
    gr = 1.0 / r
    iu = np.arange(N * N).reshape(N, N)
    iw = N * N + iu
    rows, cols, vals = [], [], []

    def add(a, b, v):
        rows.append(a.ravel()); cols.append(b.ravel()); vals.append(np.broadcast_to(v, a.shape).ravel())

    add(iu, iu, G); add(iw, iw, G); add(iu, iw, -G); add(iw, iu, -G)
    add(iu[:, 0], iu[:, 0], gr)
    a, b = iu[:, :-1], iu[:, 1:]
    add(a, a, gr); add(b, b, gr); add(a, b, -gr); add(b, a, -gr)
    a, b = iw[:-1, :], iw[1:, :]
    add(a, a, gr); add(b, b, gr); add(a, b, -gr); add(b, a, -gr)
    add(iw[-1, :], iw[-1, :], gr)
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * N * N, 2 * N * N))
    rhs = np.zeros((2 * N * N, V.shape[1]))
    rhs[iu[:, 0]] = V * gr
    x = spla.splu(A).solve(rhs)
    return (x[iw[-1, :]] * gr).T


# ---------------------------------------------------------------------------
# IR drop 标定：r_eff(N) 使参考工况平均衰减 = A(N)
# ---------------------------------------------------------------------------
def target_atten(n: int, cfg: Config) -> float:
    alpha = np.log(cfg.ir_atten_hi / cfg.ir_atten_lo) / np.log(cfg.ir_n_hi / cfg.ir_n_lo)
    return cfg.ir_atten_lo * (n / cfg.ir_n_lo) ** alpha


def _ref_setup(n: int, cfg: Config):
    rng = np.random.default_rng([cfg.seed, 777, n])
    b = 4
    L = 2 ** b - 1
    W = rng.integers(0, L + 1, size=(n, n))
    G = cfg.g_min + W * (cfg.g_max - cfg.g_min) / L
    V = rng.integers(0, L + 1, size=(n, 3)) * (cfg.v_read / L)
    return G, V


def measure_atten(G, V, r, cfg: Config) -> float:
    I = IRSolver(G, r).solve(V, tol=cfg.ir_tol, max_iter=cfg.ir_max_iter)["I"]
    return 1.0 - I.sum() / (V.T @ G).sum()


def calibrate_r_wire(n: int, cfg: Config) -> dict:
    if cfg.r_wire == 0:
        return dict(N=n, r_eff_ohm=0.0, target_atten=0.0, ref_atten=0.0)
    G, V = _ref_setup(n, cfg)
    if cfg.r_wire > 0:
        r = cfg.r_wire
        return dict(N=n, r_eff_ohm=r, target_atten=np.nan, ref_atten=measure_atten(G, V, r, cfg))
    tgt = target_atten(n, cfg)
    # 小压降区衰减 ≈ 线性于 r：比例迭代 + 对数空间割线法
    g = float(G.mean())
    r = tgt / (g * n * n / 1.5)
    hist = []
    for _ in range(30):
        a = measure_atten(G, V, r, cfg)
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


# ---------------------------------------------------------------------------
# ADC 能量表 / 输入分布（同 v2）
# ---------------------------------------------------------------------------
def load_adc_energy(path: str | Path) -> dict[str, dict[int, float]]:
    df = pd.read_csv(path)
    table = {}
    for tier in ENERGY_TIERS:
        col = f"E_{tier}_J"
        if col not in df.columns:
            raise KeyError(f"能量表缺少列 {col}")
        table[tier] = dict(zip(df["bits"].astype(int), df[col].astype(float)))
    return table


def sample_inputs(rng: np.random.Generator, dist: str, b: int, shape, cfg: Config) -> np.ndarray:
    levels = 2 ** b - 1
    if dist == "uniform":
        return rng.integers(0, levels + 1, size=shape).astype(np.float64)
    if dist == "sparse":
        x = rng.integers(1, levels + 1, size=shape).astype(np.float64)
        x[rng.random(shape) < cfg.sparsity] = 0.0
        return x
    if dist == "skewed":
        return np.rint(rng.beta(cfg.skew_a, cfg.skew_b, size=shape) * levels)
    raise ValueError(f"未知输入分布: {dist}")


def input_moments(dist: str, b: int, cfg: Config) -> tuple[float, float]:
    """输入整数 X 的解析均值与二阶矩 (E[X], E[X²])。"""
    L = 2 ** b - 1
    k = np.arange(L + 1, dtype=np.float64)
    if dist == "uniform":
        p = np.full(L + 1, 1.0 / (L + 1))
    elif dist == "sparse":
        p = np.r_[cfg.sparsity, np.full(L, (1 - cfg.sparsity) / L)]
    elif dist == "skewed":
        from scipy.stats import beta
        edges = np.clip((np.arange(L + 2) - 0.5) / L, 0, 1)
        p = np.diff(beta.cdf(edges, cfg.skew_a, cfg.skew_b))
    else:
        raise ValueError(dist)
    return float((p * k).sum()), float((p * k * k).sum())


# ---------------------------------------------------------------------------
# 阵列生成（与 v2 相同的随机流与分块顺序）
# ---------------------------------------------------------------------------
def iter_arrays(rng_arr: np.random.Generator, trials: int, n: int, b: int, sigma: float, cfg: Config):
    """逐块生成 (W, G)，与 v2 simulate_analog 的抽样顺序完全一致。yield (s, e, W, G)。"""
    levels = 2 ** b - 1
    dG = (cfg.g_max - cfg.g_min) / levels
    step = max(1, cfg.chunk_elems // (n * n))
    for s in range(0, trials, step):
        e = min(trials, s + step)
        W = rng_arr.integers(0, levels + 1, size=(e - s, n, n)).astype(np.float64)
        G = W * dG
        G += cfg.g_min
        if sigma > 0:
            eps = rng_arr.standard_normal((e - s, n, n), dtype=np.float32)
            eps *= np.float32(sigma)
            eps += np.float32(1.0)
            G *= eps
            del eps
            np.maximum(G, 0.0, out=G)
        yield s, e, W, G


def noise_std(cfg: Config, I_col: np.ndarray, cell_sq: np.ndarray, G_col: np.ndarray) -> np.ndarray:
    """列电流噪声标准差。I_col, cell_sq: (..., N)；G_col: (N,)。"""
    df = 1.0 / (2.0 * cfg.t_read)
    var = (cfg.read_noise_rel ** 2) * cell_sq \
        + 4 * K_BOLTZ * cfg.temp_K * G_col * df \
        + 2 * Q_E * np.abs(I_col) * df \
        + (cfg.tia_en ** 2) * G_col ** 2 * df + (cfg.tia_in ** 2) * df
    return np.sqrt(var)


def simulate_analog(rng_arr, rng_noise, X: np.ndarray, b: int, sigma: float, r: float, cfg: Config) -> dict:
    """
    X: (trials, K, N)。每个 trial 一个独立阵列。
    返回：
        I        (trials,K,N)  ADC 输入电流（IR drop + 噪声）
        I_prog   (trials,K,N)  仅器件编程误差的理想 crossbar 电流（v2 的模拟电流）
        I_ir     (trials,K,N)  含 IR drop、无噪声
        noise_sd (trials,K,N)
        Y_true, E_array (trials,K), iters (trials,)
    """
    trials, K, n = X.shape
    levels = 2 ** b - 1
    V = X * (cfg.v_read / levels)
    A = np.concatenate([V, V * V], axis=1)
    I_prog = np.empty((trials, K, n))
    I_ir = np.empty((trials, K, n))
    sd = np.zeros((trials, K, n))
    E_array = np.empty((trials, K))
    Y_true = np.empty((trials, K, n))
    iters = np.zeros(trials, dtype=int)
    gain_chip = np.ones((trials, n))
    for s, e, W, G in iter_arrays(rng_arr, trials, n, b, sigma, cfg):
        Y_true[s:e] = X[s:e] @ W
        R = A[s:e] @ G
        I_prog[s:e] = R[:, :K]
        if r == 0:
            I_ir[s:e] = R[:, :K]
            E_array[s:e] = cfg.t_read * R[:, K:].sum(axis=2)
            if cfg.noise:
                cell_sq = (V[s:e] ** 2) @ (G ** 2)
                for t in range(e - s):
                    sd[s + t] = noise_std(cfg, I_ir[s + t], cell_sq[t], G[t].sum(axis=0))
        else:
            for t in range(e - s):
                Gt = G[t]
                sol = IRSolver(Gt, r)
                res = sol.solve(V[s + t].T, tol=cfg.ir_tol, max_iter=cfg.ir_max_iter, fields=cfg.noise)
                if cfg.chip_cal:        # 本阵列全 1 输入的 IR 响应 / 理想列电导和（在片测量可得）
                    gain_chip[s + t] = sol.solve(np.ones((n, 1)), tol=cfg.ir_tol,
                                                 max_iter=cfg.ir_max_iter)["I"][0] / Gt.sum(axis=0)
                I_ir[s + t] = res["I"]
                iters[s + t] = res["iters"]
                E_array[s + t] = cfg.t_read * (V[s + t] * res["I_wl"]).sum(axis=1)
                if cfg.noise:
                    c = res["u"]
                    c -= res["w"]
                    c *= Gt[:, :, None]
                    c *= c
                    cell_sq = c.sum(axis=0).T            # (K, N)
                    del c
                    sd[s + t] = noise_std(cfg, res["I"], cell_sq, Gt.sum(axis=0))
                del res
        del W, G, R
    I = I_ir.copy()
    if cfg.noise:
        z = rng_noise.standard_normal(I.shape)
        I += z * sd
    return dict(I=I, I_prog=I_prog, I_ir=I_ir, noise_sd=sd, Y_true=Y_true, E_array=E_array, iters=iters,
                gain_chip=gain_chip)


def calibration_stats(rng_cal_arr, b: int, sigma: float, r: float, n: int, dists: list, cfg: Config) -> dict:
    """
    半解析 stat 量程校准：对每个校准阵列求有效传输矩阵 T 的列矩
        S1_j = Σ_i T_ij，S2_j = Σ_i T_ij²（单位 A/V，含 IR drop 与器件误差）
    输入 V_i i.i.d.（均值 m、方差 v）⇒ 列电流 μ_j = m·S1_j，Var_j = v·S2_j + 噪声方差。
    跨列、跨阵列合并：μ = ⟨μ_j⟩，σ² = ⟨Var_j⟩ + Var(μ_j)。返回 {dist: (μ, σ)}。
    """
    levels = 2 ** b - 1
    S1s, S2s, Gcols = [], [], []
    for s, e, W, G in iter_arrays(rng_cal_arr, cfg.n_cal_arrays, n, b, sigma, cfg):
        for t in range(e - s):
            Gt = G[t]
            if r == 0:
                S1, S2 = Gt.sum(axis=0), (Gt ** 2).sum(axis=0)
            else:
                sol = IRSolver(Gt, r)
                ones = sol.solve(np.ones((n, 1)), tol=cfg.ir_tol, max_iter=cfg.ir_max_iter)["I"][0]
                if n <= cfg.cal_exact_max:        # 精确 T：单位向量输入
                    T = np.empty((n, n))
                    for c0 in range(0, n, 64):
                        c1 = min(n, c0 + 64)
                        Vb = np.zeros((n, c1 - c0))
                        Vb[np.arange(c0, c1), np.arange(c1 - c0)] = 1.0
                        T[c0:c1] = sol.solve(Vb, tol=cfg.ir_tol, max_iter=cfg.ir_max_iter)["I"]
                    S2 = (T ** 2).sum(axis=0)
                else:                             # Rademacher 探针：E[(zᵀT)_j²] = Σ_i T_ij²
                    prng = np.random.default_rng([cfg.seed, 4242, n, b, int(round(sigma * 1e6)), s + t])
                    acc = np.zeros(n)
                    kb = 3
                    for p0 in range(0, cfg.cal_probes, kb):
                        kk = min(kb, cfg.cal_probes - p0)
                        Z = prng.choice([-1.0, 1.0], size=(n, kk))
                        acc += (sol.solve(Z, tol=cfg.ir_tol, max_iter=cfg.ir_max_iter)["I"] ** 2).sum(axis=0)
                    S2 = acc / cfg.cal_probes
                S1 = ones
            S1s.append(S1); S2s.append(S2); Gcols.append(Gt.sum(axis=0))
        del W, G
    S1, S2, Gc = np.array(S1s), np.array(S2s), np.array(Gcols)
    # 逐列位置增益（设计时校准）：校准阵列上 全 1 输入的 IR 响应 / 理想列电导和，跨校准阵列平均
    out = {"_gain": (S1 / np.maximum(Gc, 1e-30)).mean(axis=0)}
    for d in dists:
        ex, ex2 = input_moments(d, b, cfg)
        k = cfg.v_read / levels
        m, m2 = ex * k, ex2 * k * k
        v = m2 - m * m
        mu_j = m * S1
        var_j = v * S2
        if cfg.noise:
            var_j = var_j + noise_std(cfg, mu_j, m2 * S2, Gc) ** 2
        mu = float(mu_j.mean())
        sd = float(np.sqrt(var_j.mean() + mu_j.var()))
        out[d] = (mu, sd)
    return out


def calibration_sampled(rng_cal_arr, rng_cal_in, b, sigma, n, dists, cfg):
    """v2 的抽样校准（仅 IR 关闭时）。"""
    Xc = np.stack([sample_inputs(rng_cal_in, d, b, (cfg.n_cal_arrays, cfg.n_cal_inputs, n), cfg)
                   for d in dists], axis=2).reshape(cfg.n_cal_arrays, cfg.n_cal_inputs * len(dists), n)
    levels = 2 ** b - 1
    V = Xc * (cfg.v_read / levels)
    Is = []
    for s, e, W, G in iter_arrays(rng_cal_arr, cfg.n_cal_arrays, n, b, sigma, cfg):
        Is.append(V[s:e] @ G)
    I = np.concatenate(Is).reshape(cfg.n_cal_arrays, cfg.n_cal_inputs, len(dists), n)
    return {d: (float(I[:, :, i].mean()), float(I[:, :, i].std())) for i, d in enumerate(dists)}


def current_to_mac(I: np.ndarray, sumX: np.ndarray, b: int, cfg: Config) -> np.ndarray:
    levels = 2 ** b - 1
    k = cfg.v_read / levels
    dG = (cfg.g_max - cfg.g_min) / levels
    return (I / k - cfg.g_min * sumX[..., None]) / dG


# ---------------------------------------------------------------------------
# ADC（同 v2）
# ---------------------------------------------------------------------------
def adc_range(mode: str, n: int, cfg: Config, cal: tuple | None) -> tuple[float, float]:
    if mode == "max":
        return 0.0, n * cfg.v_read * cfg.g_max
    if mode == "stat":
        mu, sd = cal
        return max(0.0, mu - cfg.stat_k * sd), mu + cfg.stat_k * sd
    raise ValueError(f"未知 fs_mode: {mode}")


def ideal_adc(I: np.ndarray, lo: float, hi: float, bits: int):
    n_codes = 2 ** bits
    lsb = (hi - lo) / (n_codes - 1)
    raw = np.rint((I - lo) / lsb)
    sat = (raw < 0) | (raw > n_codes - 1)
    code = np.clip(raw, 0, n_codes - 1)
    return lo + code * lsb, sat


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def _rms(x):
    return np.sqrt(np.mean(x ** 2, axis=1))


def trial_metrics(Y_true, Y_prog, Y_ir, Y_analog, Y_adc, sat, y_fullscale):
    e = Y_adc - Y_true
    rmse = _rms(e)
    eff_bits = np.log2(y_fullscale / (np.maximum(rmse, 1e-12) * np.sqrt(12)))
    sig_std = np.std(Y_true, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_db = 20 * np.log10(np.maximum(sig_std, 1e-12) / np.maximum(rmse, 1e-12))
    return dict(
        rmse=rmse,
        nrmse=rmse / y_fullscale,
        max_abs_err=np.max(np.abs(e), axis=1),
        mean_err=np.mean(e, axis=1),
        rmse_device=_rms(Y_analog - Y_true),      # ADC 之前的全部模拟误差（编程误差 + IR + 噪声）
        rmse_prog=_rms(Y_prog - Y_true),          # 仅编程误差
        rmse_ir=_rms(Y_ir - Y_prog),              # IR drop 增量
        rmse_noise=_rms(Y_analog - Y_ir),         # 读出噪声增量
        rmse_adc=_rms(Y_adc - Y_analog),
        exact_rate=np.mean(np.rint(Y_adc) == Y_true, axis=1),
        sat_rate=np.mean(sat, axis=1),
        eff_bits=eff_bits,
        snr_db=snr_db,
        eff_bits_sig=(snr_db - 1.76) / 6.02,
        sig_std=sig_std,
    )


def cal_metrics(Y_true, Y_prog, Y_ir_cal, Y_adc_cal, y_fullscale, sfx="cal"):
    """逐列数字增益校正后的指标（校正系数来自独立校准阵列，见 calibration_stats）。"""
    e = Y_adc_cal - Y_true
    rmse = _rms(e)
    sig_std = np.std(Y_true, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_db = 20 * np.log10(np.maximum(sig_std, 1e-12) / np.maximum(rmse, 1e-12))
    return {f"rmse_{sfx}": rmse,
            f"eff_bits_{sfx}": np.log2(y_fullscale / (np.maximum(rmse, 1e-12) * np.sqrt(12))),
            f"snr_db_{sfx}": snr_db, f"eff_bits_sig_{sfx}": (snr_db - 1.76) / 6.02,
            f"rmse_ir_{sfx}": _rms(Y_ir_cal - Y_prog)}


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def run(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    energy = load_adc_energy(cfg.energy_csv)
    for B in cfg.adc_bits:
        if B not in energy["typical"]:
            raise ValueError(f"能量表中没有 {B} bit ADC 的数据")
    if cfg.cal_mode == "sampled" and cfg.r_wire != 0:
        raise ValueError("cal_mode=sampled 仅在 --r-wire 0 时可用")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    t0 = time.time()
    irc = pd.DataFrame([calibrate_r_wire(n, cfg) for n in cfg.sizes])
    irc["ref_atten_1ohm"] = [measure_atten(*_ref_setup(n, cfg), 1.0, cfg) if n <= 1024 else np.nan
                             for n in cfg.sizes] if cfg.r_wire != 0 else np.nan
    irc.to_csv(out / "ir_calibration.csv", index=False)
    print(irc.to_string(index=False), f"\n(IR calibration {time.time() - t0:.1f}s)", flush=True)
    r_of = dict(zip(irc.N, irc.r_eff_ohm))

    dists = list(cfg.input_dists)
    blocks = []
    grid = list(itertools.product(cfg.sizes, cfg.mem_bits, cfg.mem_sigma))
    for idx, (n, b, sigma) in enumerate(grid, 1):
        ss = np.random.SeedSequence([cfg.seed, b, int(round(sigma * 1e6)), n])
        rng_arr, rng_in, rng_cal_arr, rng_cal_in = (np.random.default_rng(s) for s in ss.spawn(4))
        rng_noise = np.random.default_rng(ss.spawn(1)[0])
        r = float(r_of[n])

        X = np.stack([sample_inputs(rng_in, d, b, (cfg.n_mc, n), cfg) for d in dists], axis=1)
        sim = simulate_analog(rng_arr, rng_noise, X, b, sigma, r, cfg)
        sumX = X.sum(axis=2)
        Y_analog = current_to_mac(sim["I"], sumX, b, cfg)
        Y_prog = current_to_mac(sim["I_prog"], sumX, b, cfg)
        Y_ir = current_to_mac(sim["I_ir"], sumX, b, cfg)
        y_fs = n * (2 ** b - 1) ** 2

        cal = None
        if "stat" in cfg.fs_modes:
            if cfg.cal_mode == "sampled":
                cal = calibration_sampled(rng_cal_arr, rng_cal_in, b, sigma, n, dists, cfg)
            else:
                cal = calibration_stats(rng_cal_arr, b, sigma, r, n, dists, cfg)

        gain = np.ones(n) if (cal is None or "_gain" not in cal) else cal["_gain"]
        for di, dist in enumerate(dists):
            I_d, Yt_d, sX_d = sim["I"][:, di], sim["Y_true"][:, di], sumX[:, di]
            E_arr = sim["E_array"][:, di]
            ideal_sum = sim["I_prog"][:, di].sum(axis=1)
            ir_atten = 1.0 - sim["I_ir"][:, di].sum(axis=1) / np.maximum(ideal_sum, 1e-30)
            noise_rms = np.sqrt(np.mean(sim["noise_sd"][:, di] ** 2, axis=1))
            for mode in cfg.fs_modes:
                lo, hi = adc_range(mode, n, cfg, None if cal is None else cal[dist])
                for B in cfg.adc_bits:
                    I_q, sat = ideal_adc(I_d, lo, hi, B)
                    m = trial_metrics(Yt_d, Y_prog[:, di], Y_ir[:, di], Y_analog[:, di],
                                      current_to_mac(I_q, sX_d, b, cfg), sat, y_fs)
                    m.update(cal_metrics(Yt_d, Y_prog[:, di], current_to_mac(sim["I_ir"][:, di] / gain, sX_d, b, cfg),
                                         current_to_mac(I_q / gain, sX_d, b, cfg), y_fs))
                    if cfg.chip_cal:
                        gc = sim["gain_chip"]
                        m.update(cal_metrics(Yt_d, Y_prog[:, di],
                                             current_to_mac(sim["I_ir"][:, di] / gc, sX_d, b, cfg),
                                             current_to_mac(I_q / gc, sX_d, b, cfg), y_fs, sfx="chip"))
                    df = pd.DataFrame(m)
                    df.insert(0, "trial", np.arange(cfg.n_mc))
                    for k, v in dict(input_dist=dist, mem_bits=b, mem_sigma=sigma, adc_bits=B,
                                     N=n, fs_mode=mode).items():
                        df.insert(0, k, v)
                    df["adc_lo_A"], df["adc_hi_A"], df["adc_lsb_A"] = lo, hi, (hi - lo) / (2 ** B - 1)
                    df["I_col_mean_A"] = I_d.mean(axis=1)
                    df["input_mean_frac"] = X[:, di].mean(axis=1) / (2 ** b - 1)
                    df["r_eff_ohm"] = r
                    df["ir_atten"] = ir_atten
                    df["noise_rms_A"] = noise_rms
                    df["ir_iters"] = sim["iters"]
                    df["col_gain_mean"] = float(gain.mean())
                    df["E_array_J"] = E_arr
                    for tier in ENERGY_TIERS:
                        df[f"E_adc_{tier}_J"] = n * energy[tier][B]
                        df[f"E_total_{tier}_J"] = df[f"E_adc_{tier}_J"] + E_arr
                    df["E_adc_typical_per_MAC_J"] = df["E_adc_typical_J"] / (n * n)
                    df["E_total_typical_per_MAC_J"] = df["E_total_typical_J"] / (n * n)
                    df["adc_energy_share_typical"] = df["E_adc_typical_J"] / df["E_total_typical_J"]
                    blocks.append(df)

        print(f"[{idx:>3}/{len(grid)}] N={n:<5} b={b} σ={sigma:.3f}  r={r:.3g}Ω  "
              f"iters≈{sim['iters'].mean():.1f}  elapsed {time.time() - t0:7.1f}s", flush=True)
        del sim, X, Y_analog, Y_prog, Y_ir

    raw = pd.concat(blocks, ignore_index=True)
    raw.to_csv(out / "raw_trials.csv.gz", index=False, float_format="%.6g")
    summary = summarize(raw)
    summary.to_csv(out / "summary.csv", index=False)
    knee_table(summary, cfg).to_csv(out / "knee.csv", index=False)
    return raw, summary


KEYS = ["input_dist", "mem_bits", "mem_sigma", "adc_bits", "N", "fs_mode"]


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """同 v2；另含 IR / 噪声相关列。"""
    raw = raw.copy()
    if "sig_std" not in raw.columns:
        raw["sig_std"] = np.maximum(raw["rmse"], 1e-12) * 10 ** (raw["snr_db"] / 20)
    raw["_mse"] = raw["rmse"] ** 2
    raw["_sigvar"] = raw["sig_std"] ** 2
    metrics = ["rmse", "nrmse", "max_abs_err", "mean_err", "rmse_device", "rmse_adc",
               "exact_rate", "sat_rate", "eff_bits", "snr_db", "eff_bits_sig",
               "input_mean_frac", "E_array_J"] + \
              [f"E_total_{t}_J" for t in ENERGY_TIERS] + \
              ["E_total_typical_per_MAC_J", "adc_energy_share_typical"]
    metrics += [c for c in ["rmse_prog", "rmse_ir", "rmse_noise", "ir_atten", "noise_rms_A", "ir_iters",
                            "rmse_cal", "eff_bits_cal", "eff_bits_sig_cal", "rmse_ir_cal", "col_gain_mean",
                            "rmse_chip", "eff_bits_chip", "eff_bits_sig_chip", "rmse_ir_chip"]
                if c in raw.columns]
    for sfx in ("cal", "chip"):
        if f"rmse_{sfx}" in raw.columns:
            raw[f"_mse_{sfx}"] = raw[f"rmse_{sfx}"] ** 2
    g = raw.groupby(KEYS, sort=True)
    agg = g[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}_{s}" for a, s in agg.columns]
    agg["nrmse_p95"] = g["nrmse"].quantile(0.95)
    agg["max_abs_err_p95"] = g["max_abs_err"].quantile(0.95)
    mse = g["_mse"].mean()
    y_fs = g["N"].first() * (2 ** g["mem_bits"].first() - 1) ** 2
    agg["eff_bits_pooled"] = np.log2(y_fs / (np.sqrt(np.maximum(mse, 1e-24)) * np.sqrt(12)))
    snr_pooled = 10 * np.log10(np.maximum(g["_sigvar"].mean(), 1e-24) / np.maximum(mse, 1e-24))
    agg["snr_db_pooled"] = snr_pooled
    agg["eff_bits_sig_pooled"] = (snr_pooled - 1.76) / 6.02
    for sfx in ("cal", "chip"):
        if f"_mse_{sfx}" not in raw.columns:
            continue
        mse_c = g[f"_mse_{sfx}"].mean()
        agg[f"eff_bits_{sfx}_pooled"] = np.log2(y_fs / (np.sqrt(np.maximum(mse_c, 1e-24)) * np.sqrt(12)))
        snr_c = 10 * np.log10(np.maximum(g["_sigvar"].mean(), 1e-24) / np.maximum(mse_c, 1e-24))
        agg[f"snr_db_{sfx}_pooled"] = snr_c
        agg[f"eff_bits_sig_{sfx}_pooled"] = (snr_c - 1.76) / 6.02
    const_cols = [f"E_adc_{t}_J" for t in ENERGY_TIERS] + ["E_adc_typical_per_MAC_J", "adc_lsb_A"]
    const_cols += [c for c in ["r_eff_ohm"] if c in raw.columns]
    const = g[const_cols].first()
    return agg.join(const).reset_index()


def knee_table(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """拐点：有效位已达到最大 ADC 位宽结果 − tol 的最小 ADC 位宽。有校正列时另给 *_cal 版本。"""
    rows = []
    has_cal = "eff_bits_cal_pooled" in summary.columns
    for key, s in summary.groupby(["input_dist", "mem_bits", "mem_sigma", "N", "fs_mode"]):
        s = s.sort_values("adc_bits")
        ceiling = s["eff_bits_pooled"].iloc[-1]
        k = s[s["eff_bits_pooled"] >= ceiling - cfg.knee_tol].iloc[0]
        row = dict(zip(["input_dist", "mem_bits", "mem_sigma", "N", "fs_mode"], key),
                   knee_adc_bits=int(k.adc_bits), eff_bits_at_knee=k.eff_bits_pooled,
                   eff_bits_ceiling=ceiling, eff_bits_sig_at_knee=k.eff_bits_sig_pooled,
                   E_total_typical_per_MAC_at_knee_J=k.E_total_typical_per_MAC_J_mean,
                   adc_energy_share_at_knee=k.adc_energy_share_typical_mean)
        if has_cal:
            cc = s["eff_bits_cal_pooled"].iloc[-1]
            kc = s[s["eff_bits_cal_pooled"] >= cc - cfg.knee_tol].iloc[0]
            row.update(knee_adc_bits_cal=int(kc.adc_bits), eff_bits_cal_at_knee=kc.eff_bits_cal_pooled,
                       eff_bits_cal_ceiling=cc, eff_bits_sig_cal_at_knee=kc.eff_bits_sig_cal_pooled,
                       E_total_typical_per_MAC_at_knee_cal_J=kc.E_total_typical_per_MAC_J_mean)
        if "eff_bits_chip_pooled" in s.columns:
            cc = s["eff_bits_chip_pooled"].iloc[-1]
            kc = s[s["eff_bits_chip_pooled"] >= cc - cfg.knee_tol].iloc[0]
            row.update(knee_adc_bits_chip=int(kc.adc_bits), eff_bits_chip_ceiling=cc,
                       eff_bits_sig_chip_at_knee=kc.eff_bits_sig_chip_pooled)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def self_check(cfg: Config) -> None:
    rng = np.random.default_rng(0)
    c = Config(**{**asdict(cfg), "chunk_elems": 64 * 64, "noise": False})
    # 1. 无 IR、无噪声：MAC 反推精确（同 v2）
    for b in (1, 4, 8):
        X = np.stack([sample_inputs(rng, d, b, (4, 64), c) for d in INPUT_DISTS], axis=1)
        sim = simulate_analog(rng, rng, X, b, 0.0, 0.0, c)
        err = np.max(np.abs(current_to_mac(sim["I"], X.sum(axis=2), b, c) - sim["Y_true"]))
        assert err < 1e-6, f"自检失败 b={b}: 最大误差 {err}"
    # 2. IR 求解器 vs 稀疏直接法
    for n, r in ((1, 1.0), (7, 2.0), (48, 1.0), (96, 0.5)):
        G = rng.uniform(1e-6, 100e-6, (n, n))
        V = rng.uniform(-0.2, 0.2, (n, 3))
        ex = exact_solve(G, V, r)
        so = IRSolver(G, r).solve(V, tol=1e-12, max_iter=2000)["I"]
        rel = np.abs(so - ex).max() / np.abs(ex).max()
        assert rel < 1e-8, f"IR 求解器自检失败 N={n}: {rel:.2e}"
        # 能量守恒：驱动器功率 = 单元 + 导线耗散，用输出/输入电流做 KCL 检查
        res = IRSolver(G, r).solve(V, tol=1e-12, max_iter=2000)
        kcl = np.abs(res["I_wl"].sum(axis=1) - res["I"].sum(axis=1)).max() / np.abs(res["I"]).max()
        assert kcl < 1e-8, f"KCL 自检失败 N={n}: {kcl:.2e}"
    # 3. ADC / 分布（同 v2）
    I = np.linspace(0, 1, 1001)
    I_q, sat = ideal_adc(I, 0.0, 1.0, 4)
    assert not sat.any() and np.max(np.abs(I_q - I)) <= 0.5 / 15 + 1e-12, "ADC 自检失败"
    x = sample_inputs(rng, "sparse", 4, (200000,), c)
    assert abs((x == 0).mean() - c.sparsity) < 0.01, "稀疏分布自检失败"
    for d in INPUT_DISTS:
        xs = sample_inputs(rng, d, 4, (400000,), c)
        m, m2 = input_moments(d, 4, c)
        assert abs(xs.mean() / m - 1) < 0.01 and abs((xs ** 2).mean() / m2 - 1) < 0.01, f"矩自检失败 {d}"
    print("self-check passed", flush=True)


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------
_BLUE_ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
_CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#8a6fd1", "#d4a017"]


def _ordinal_colors(k: int) -> list[str]:
    if k <= 1:
        return [_BLUE_ORDINAL[3]]
    idx = np.linspace(0, len(_BLUE_ORDINAL) - 1, k).round().astype(int)
    return [_BLUE_ORDINAL[i] for i in idx]


def _style(ax):
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e", labelsize=8)


def _label(ax, x=None, y=None):
    if x:
        ax.set_xlabel(x, fontsize=8, color="#52514e")
    if y:
        ax.set_ylabel(y, fontsize=8, color="#52514e")


def make_plots(summary: pd.DataFrame, cfg: Config, cal: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    out = Path(cfg.out_dir)
    sig = min(cfg.mem_sigma, key=lambda s: abs(s - cfg.plot_sigma))
    d = summary[summary["mem_sigma"] == sig]
    EB, EBS = ("eff_bits_cal_pooled", "eff_bits_sig_cal_pooled") if cal else ("eff_bits_pooled", "eff_bits_sig_pooled")
    KN = "knee_adc_bits_cal" if cal else "knee_adc_bits"
    sfx, tag = ("_cal", ", column-gain calibrated") if cal else ("", ", uncalibrated")
    dists = [x for x in cfg.input_dists if x in d["input_dist"].unique()]
    sizes = [n for n in cfg.plot_sizes if n in d["N"].unique()] or sorted(d["N"].unique())
    mb = sorted(d["mem_bits"].unique())
    mcol = dict(zip(mb, _ordinal_colors(len(mb))))
    dcol = dict(zip(dists, _CATEGORICAL))

    for mode in cfg.fs_modes:
        fig, axes = plt.subplots(len(dists), len(sizes), figsize=(3.1 * len(sizes), 2.6 * len(dists)),
                                 sharex=True, sharey=True, squeeze=False)
        for r, dist in enumerate(dists):
            for c, n in enumerate(sizes):
                ax = axes[r][c]
                _style(ax)
                for b in mb:
                    s = d[(d.input_dist == dist) & (d.fs_mode == mode) & (d.N == n) & (d.mem_bits == b)]
                    s = s.sort_values("adc_bits")
                    ax.plot(s.adc_bits, s[EBS], color=mcol[b], lw=2, marker="o", ms=4, label=f"{b}-bit")
                ax.set_title(f"{dist}, N={n}", fontsize=9, color="#0b0b0b")
                _label(ax, "ADC bits" if r == len(dists) - 1 else None,
                       "signal-referred eff. bits" if c == 0 else None)
        axes[0][-1].legend(title="memristor bits", fontsize=7, title_fontsize=7, frameon=False, loc="lower right")
        fig.suptitle(f"Signal-referred effective bits vs ADC resolution  (FS={mode}, σ = {sig:.0%}, IR drop + noise{tag})",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"fig_effbits_vs_adc_{mode}{sfx}.png", dpi=150)
        plt.close(fig)

    mode = "stat" if "stat" in cfg.fs_modes else cfg.fs_modes[0]
    fig, axes = plt.subplots(len(dists), len(sizes), figsize=(3.1 * len(sizes), 2.7 * len(dists)),
                             sharey=True, squeeze=False)
    for r, dist in enumerate(dists):
        for c, n in enumerate(sizes):
            ax = axes[r][c]
            _style(ax)
            s = d[(d.input_dist == dist) & (d.fs_mode == mode) & (d.N == n)]
            for b in mb:
                sb = s[s.mem_bits == b].sort_values("adc_bits")
                ax.plot(sb.E_total_typical_per_MAC_J_mean * 1e15, sb[EBS], color=mcol[b], lw=2,
                        marker="o", ms=4, label=f"{b}-bit", zorder=3)
            if c == 0:
                sb = s[s.mem_bits == mb[-1]].sort_values("adc_bits")
                for _, r_ in sb.iterrows():
                    ax.annotate(f"{int(r_.adc_bits)}b", (r_.E_total_typical_per_MAC_J_mean * 1e15,
                                r_[EBS]), textcoords="offset points", xytext=(0, 6),
                                fontsize=6, color="#52514e", ha="center")
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(f"{dist}, N={n}", fontsize=9, color="#0b0b0b")
            _label(ax, "total energy per MAC (fJ, typical)" if r == len(dists) - 1 else None,
                   "signal-referred eff. bits" if c == 0 else None)
    axes[0][-1].legend(title="memristor bits", fontsize=7, title_fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"Accuracy vs total energy (ADC + array), FS={mode}, σ = {sig:.0%}, IR drop + noise{tag}; "
                 f"points along each line = ADC 4→12 bit", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"fig_pareto_total{sfx}.png", dpi=150)
    plt.close(fig)

    b0 = cfg.plot_mem_bits if cfg.plot_mem_bits in mb else mb[len(mb) // 2]
    B_ref = 8 if 8 in cfg.adc_bits else cfg.adc_bits[len(cfg.adc_bits) // 2]
    knee = knee_table(summary, cfg)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    axes = axes.ravel()
    for dist in dists:
        ss12 = d[(d.input_dist == dist) & (d.mem_bits == b0) & (d.adc_bits == B_ref) & (d.fs_mode == mode)]
        ss12 = ss12.sort_values("N")
        axes[3].plot(ss12.N, ss12[EB], color=dcol[dist], lw=2, marker="o", ms=5, label=f"{dist} (FS-ref)")
        axes[3].plot(ss12.N, ss12[EBS], color=dcol[dist], lw=2, ls=":", marker="s", ms=4,
                     label=f"{dist} (signal-ref)")
        s = d[(d.input_dist == dist) & (d.mem_bits == b0) & (d.adc_bits == B_ref)]
        if {"max", "stat"} <= set(cfg.fs_modes):
            piv = s.pivot(index="N", columns="fs_mode", values=EB)
            axes[0].plot(piv.index, piv["stat"] - piv["max"], color=dcol[dist], lw=2, marker="o", ms=5, label=dist)
        k = knee[(knee.input_dist == dist) & (knee.mem_bits == b0) & (knee.mem_sigma == sig) & (knee.fs_mode == mode)]
        axes[1].plot(k.N, k[KN], color=dcol[dist], lw=2, marker="o", ms=5, label=dist)
        ss = s[s.fs_mode == mode].sort_values("N")
        axes[2].plot(ss.N, ss.adc_energy_share_typical_mean * 100, color=dcol[dist], lw=2, marker="o", ms=5,
                     label=dist)
    titles = [f"stat − max gain @ {B_ref}-bit ADC", f"knee ADC bits (FS={mode})",
              f"ADC share of total energy @ {B_ref}-bit (%)", f"effective bits @ {B_ref}-bit ADC, FS={mode}"]
    ylabels = ["Δ effective bits", "ADC bits", "ADC energy / total (%)", "effective bits"]
    for ax, t, yl in zip(axes, titles, ylabels):
        _style(ax)
        ax.set_xscale("log", base=2)
        ax.set_title(t, fontsize=9, color="#0b0b0b")
        _label(ax, "array size N", yl)
    axes[0].legend(title="input", fontsize=7, title_fontsize=7, frameon=False)
    axes[3].legend(fontsize=6, frameon=False, ncol=1, loc="best")
    fig.suptitle(f"Scaling trends ({b0}-bit memristor, σ = {sig:.0%}, IR drop + noise{tag})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"fig_scaling_trends{sfx}.png", dpi=150)
    plt.close(fig)

    # ---- 图 4（v3 新增）：非理想量随规模的变化与误差分解 ----
    if "rmse_ir_mean" in summary.columns and not cal:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        base = d[(d.mem_bits == b0) & (d.adc_bits == B_ref) & (d.fs_mode == mode)]
        for dist in dists:
            s = base[base.input_dist == dist].sort_values("N")
            axes[0].plot(s.N, s.ir_atten_mean * 100, color=dcol[dist], lw=2, marker="o", ms=5, label=dist)
            fs_I = s.N * cfg.v_read * cfg.g_max
            axes[1].plot(s.N, s.noise_rms_A_mean / fs_I * 1e6, color=dcol[dist], lw=2, marker="o", ms=5, label=dist)
        s = base[base.input_dist == "uniform"].sort_values("N") if "uniform" in dists else \
            base[base.input_dist == dists[0]].sort_values("N")
        comps = [("rmse_prog_mean", "programming σ"), ("rmse_ir_mean", "IR drop (raw)"),
                 ("rmse_ir_cal_mean", "IR drop (after col-gain cal)"),
                 ("rmse_noise_mean", "read noise"), ("rmse_adc_mean", f"ADC ({B_ref}-bit)")]
        for (col, lab), colr in zip(comps, _CATEGORICAL):
            y_fs = s.N * (2 ** b0 - 1) ** 2
            axes[2].plot(s.N, s[col] / y_fs, color=colr, lw=2, marker="o", ms=5, label=lab)
        axes[2].set_yscale("log")
        for ax, t, yl in zip(axes, ["mean IR-drop signal attenuation", "read-noise rms / FS current",
                                    f"error breakdown ({s.input_dist.iloc[0]})"],
                             ["attenuation (%)", "ppm of FS", "rmse / MAC full scale"]):
            _style(ax)
            ax.set_xscale("log", base=2)
            ax.set_title(t, fontsize=9, color="#0b0b0b")
            _label(ax, "array size N", yl)
        axes[0].legend(title="input", fontsize=7, title_fontsize=7, frameon=False)
        axes[2].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Circuit non-idealities vs array size ({b0}-bit memristor, σ = {sig:.0%})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "fig_nonideal_scaling.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> tuple[Config, tuple]:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mem-bits", type=int, nargs="+", default=d.mem_bits)
    p.add_argument("--mem-sigma", type=float, nargs="+", default=d.mem_sigma, help="相对电导误差 σ（0.02 = 2%%）")
    p.add_argument("--adc-bits", type=int, nargs="+", default=d.adc_bits)
    p.add_argument("--sizes", type=int, nargs="+", default=d.sizes)
    p.add_argument("--fs-modes", nargs="+", default=d.fs_modes, choices=["max", "stat"])
    p.add_argument("--input-dists", nargs="+", default=d.input_dists, choices=list(INPUT_DISTS))
    p.add_argument("--n-mc", type=int, default=d.n_mc)
    p.add_argument("--sparsity", type=float, default=d.sparsity)
    p.add_argument("--skew-a", type=float, default=d.skew_a)
    p.add_argument("--skew-b", type=float, default=d.skew_b)
    p.add_argument("--g-min", type=float, default=d.g_min)
    p.add_argument("--g-max", type=float, default=d.g_max)
    p.add_argument("--v-read", type=float, default=d.v_read)
    p.add_argument("--t-read", type=float, default=d.t_read)
    p.add_argument("--r-wire", type=float, default=d.r_wire,
                   help="<0：按 A(N) 标定等效线电阻（默认）；0：关闭 IR drop；>0：固定线电阻 Ω/段")
    p.add_argument("--ir-atten-lo", type=float, default=d.ir_atten_lo)
    p.add_argument("--ir-atten-hi", type=float, default=d.ir_atten_hi)
    p.add_argument("--ir-n-lo", type=int, default=d.ir_n_lo)
    p.add_argument("--ir-n-hi", type=int, default=d.ir_n_hi)
    p.add_argument("--ir-tol", type=float, default=d.ir_tol)
    p.add_argument("--ir-max-iter", type=int, default=d.ir_max_iter)
    p.add_argument("--no-noise", dest="noise", action="store_false")
    p.add_argument("--chip-cal", action="store_true", help="另算逐芯片逐列增益校正指标（*_chip）")
    p.add_argument("--read-noise-rel", type=float, default=d.read_noise_rel)
    p.add_argument("--temp-K", type=float, default=d.temp_K)
    p.add_argument("--tia-en", type=float, default=d.tia_en)
    p.add_argument("--tia-in", type=float, default=d.tia_in)
    p.add_argument("--stat-k", type=float, default=d.stat_k)
    p.add_argument("--n-cal-arrays", type=int, default=d.n_cal_arrays)
    p.add_argument("--n-cal-inputs", type=int, default=d.n_cal_inputs)
    p.add_argument("--cal-mode", default=d.cal_mode, choices=["analytic", "sampled"])
    p.add_argument("--cal-probes", type=int, default=d.cal_probes)
    p.add_argument("--cal-exact-max", type=int, default=d.cal_exact_max)
    p.add_argument("--energy-csv", default=d.energy_csv)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--out", dest="out_dir", default=d.out_dir)
    p.add_argument("--chunk-elems", type=int, default=d.chunk_elems)
    p.add_argument("--plot-sigma", type=float, default=d.plot_sigma)
    p.add_argument("--plot-sizes", type=int, nargs="+", default=d.plot_sizes)
    p.add_argument("--plot-mem-bits", type=int, default=d.plot_mem_bits)
    p.add_argument("--knee-tol", type=float, default=d.knee_tol)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-self-check", action="store_true")
    p.add_argument("--resummarize", action="store_true",
                   help="不重跑仿真：从已有 raw_trials.csv.gz 重新生成 summary / knee / 图")
    a = vars(p.parse_args())
    flags = (a.pop("no_plots"), a.pop("resummarize"), a.pop("no_self_check"))
    return Config(**a), flags


if __name__ == "__main__":
    cfg, (no_plots, resum, no_check) = parse_args()
    if resum:
        out = Path(cfg.out_dir)
        summary = summarize(pd.read_csv(out / "raw_trials.csv.gz"))
        summary.to_csv(out / "summary.csv", index=False)
        knee_table(summary, cfg).to_csv(out / "knee.csv", index=False)
        if not no_plots:
            make_plots(summary, cfg)
            if "eff_bits_cal_pooled" in summary.columns:
                make_plots(summary, cfg, cal=True)
        print(f"re-summarized → {out.resolve()}")
    else:
        if not no_check:
            self_check(cfg)
        raw, summary = run(cfg)
        if not no_plots:
            make_plots(summary, cfg)
            if "eff_bits_cal_pooled" in summary.columns:
                make_plots(summary, cfg, cal=True)
        print(f"done → {Path(cfg.out_dir).resolve()}  ({len(raw)} trials, {len(summary)} configs)")
