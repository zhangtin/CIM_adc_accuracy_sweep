# -*- coding: utf-8 -*-
"""
带线电阻 crossbar 的节点方程求解。

    IRSolver     迭代求解器（block-SOR + 逐线精确三对角求解，C 内核）
    exact_solve  稀疏直接法（参考解，仅用于自检，N ≲ 256）

约定：阵列为 N×N 方阵；输入电压 V 形状 (N, K)，即 K 组输入并行求解；
输出电流按 (K, N) 返回（与 MAC 结果的 [输入组, 列] 排列一致）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernel import load_kernel


@dataclass
class IRSolution:
    I: np.ndarray                 # (K, N) 各列输出电流 (A)（流入 TIA 虚地）
    I_wl: np.ndarray              # (K, N) 各行驱动器输出电流 (A)
    iters: int                    # 实际迭代次数
    u: np.ndarray | None = None   # (N, N, K) WL 节点电压（return_fields=True 时）
    w: np.ndarray | None = None   # (N, N, K) BL 节点电压（return_fields=True 时）


class IRSolver:
    """
    单个阵列（固定电导 G、每段线电阻 r_wire）在多组输入下的求解器。
    构造时完成三对角分解与松弛因子估计，之后可对不同 V 反复调用 solve()。

    Args:
        G:      (N, N) 实际电导 (S)
        r_wire: 每段（相邻单元之间）线电阻 (Ω)，必须 > 0
    """

    def __init__(self, G: np.ndarray, r_wire: float):
        if r_wire <= 0:
            raise ValueError("IRSolver 需要 r_wire > 0；r_wire = 0 时直接用 V @ G")
        self.G = np.ascontiguousarray(G, dtype=np.float64)
        self.N = self.G.shape[0]
        self.r = float(r_wire)
        self.mwl = np.empty_like(self.G)
        self.mbl = np.empty_like(self.G)
        load_kernel().ir_factor(self.G, self.N, self.r, self.mwl, self.mbl)
        # 2-cyclic 系统 Young 最优松弛因子（以平均电导估计 block-Jacobi 谱半径）
        lam = (2.0 / self.r) * (1.0 - np.cos(np.pi / (2 * self.N + 1)))
        g = float(self.G.mean())
        mu2 = g * g / (g + lam) ** 2
        self.omega = 2.0 / (1.0 + np.sqrt(max(1.0 - mu2, 0.0)))

    def solve(self, V: np.ndarray, tol: float = 1e-6, max_iter: int = 500,
              return_fields: bool = False) -> IRSolution:
        """
        Args:
            V:             (N, K) 行驱动电压 (V)
            tol:           相对收敛判据（节点电压相对 max|V|，输出电流相对 max|I|）
            max_iter:      最大迭代次数（不收敛时打印警告并返回当前解）
            return_fields: 是否返回全部节点电压 u、w（读噪声计算需要单元两端电压）

        Returns:
            IRSolution
        """
        N, r = self.N, self.r
        V = np.ascontiguousarray(V, dtype=np.float64)
        K = V.shape[1]
        u = np.repeat(V[:, None, :], N, axis=1)
        w = np.zeros((N, N, K))
        tmp = np.empty((N, N, K))
        out = np.zeros(3)
        vmax = max(float(np.abs(V).max()), 1e-30)
        lib = load_kernel()
        it = 0
        for it in range(1, max_iter + 1):
            lib.ir_sor_iter(self.G, self.mwl, self.mbl, V, u, w, tmp, N, K, r, self.omega, out)
            imax = max(float(np.abs(w[N - 1]).max()) / r, 1e-30)
            if max(out[0], out[1]) < tol * vmax and out[2] < tol * imax:
                break
        else:
            print(f"  [warn] IR solver not converged in {max_iter} iters (N={N})", flush=True)
        del tmp
        sol = IRSolution(I=(w[N - 1] / r).T.copy(), I_wl=((V - u[:, 0, :]) / r).T.copy(), iters=it)
        if return_fields:
            sol.u, sol.w = u, w
        return sol

    def column_gain(self, tol: float = 1e-6, max_iter: int = 500) -> np.ndarray:
        """
        逐列增益：全 1 V 输入时的 IR 响应 / 理想列电导和，(N,)。
        （也就是传输矩阵的列和 S1 除以 ΣG）
        """
        I1 = self.solve(np.ones((self.N, 1)), tol=tol, max_iter=max_iter).I[0]
        return I1 / self.G.sum(axis=0)


def solve_crossbar(G: np.ndarray, V: np.ndarray, r_wire: float, tol: float = 1e-6,
                   max_iter: int = 500) -> np.ndarray:
    """
    便捷函数：返回列电流 (K, N)。r_wire = 0 时为理想 Vᵀ·G。

    Args:
        G:      (N, N) 电导 (S)
        V:      (N, K) 行驱动电压 (V)
        r_wire: 每段线电阻 (Ω)
    """
    if r_wire == 0:
        return V.T @ G
    return IRSolver(G, r_wire).solve(V, tol=tol, max_iter=max_iter).I


def exact_solve(G: np.ndarray, V: np.ndarray, r_wire: float) -> np.ndarray:
    """
    稀疏直接法求解完整 2N² 节点方程（参考解）。

    Args:
        G: (N, N) 电导 (S)；V: (N, K) 驱动电压 (V)；r_wire: 每段线电阻 (Ω)

    Returns:
        (K, N) 列输出电流 (A)
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    N = G.shape[0]
    gr = 1.0 / r_wire
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
