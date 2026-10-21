# -*- coding: utf-8 -*-
"""
Nodal solution for RECTANGULAR crossbars (R rows x C columns) with wire resistance.

    RectIRSolver      iterative solver (block-SOR with exact tridiagonal line solves, C kernel)
    exact_solve_rect  sparse direct solve of the full 2·R·C node system (reference, small arrays)
    make_ir_solver    factory: IRSolver for square arrays (bit-identical to the original path),
                      RectIRSolver otherwise

The circuit topology and the convergence criterion are identical to ir_drop.solver.IRSolver:
row drivers at column 0 of each word line, bit lines terminated at the bottom (row R-1) by a TIA
virtual ground, every wire segment of resistance r_wire. Inputs V have shape (R, K); output
currents are returned as (K, C).
"""
from __future__ import annotations

import numpy as np

from .kernel import load_rect_kernel
from .solver import IRSolution, IRSolver


class RectIRSolver:
    """
    Solver for one rectangular array with fixed conductances and per-segment wire resistance.

    Args:
        G:      (R, C) programmed conductances (S)
        r_wire: wire resistance per segment between adjacent cells (Ohm), must be > 0
    """

    def __init__(self, G: np.ndarray, r_wire: float):
        if r_wire <= 0:
            raise ValueError("RectIRSolver requires r_wire > 0; use V @ G for ideal interconnect")
        self.G = np.ascontiguousarray(G, dtype=np.float64)
        if self.G.ndim != 2:
            raise ValueError(f"G must be 2-D, got shape {self.G.shape}")
        self.R, self.C = self.G.shape
        self.r = float(r_wire)
        self.mwl = np.empty_like(self.G)
        self.mbl = np.empty_like(self.G)
        load_rect_kernel().ir_factor_rect(self.G, self.R, self.C, self.r, self.mwl, self.mbl)
        # Relaxation factor: same Young estimate as IRSolver, using the longer line length.
        # It only affects the convergence rate; accuracy is set by the tolerance.
        n_eff = max(self.R, self.C)
        lam = (2.0 / self.r) * (1.0 - np.cos(np.pi / (2 * n_eff + 1)))
        g = float(self.G.mean())
        mu2 = g * g / (g + lam) ** 2
        self.omega = 2.0 / (1.0 + np.sqrt(max(1.0 - mu2, 0.0)))

    def solve(self, V: np.ndarray, tol: float = 1e-6, max_iter: int = 500,
              return_fields: bool = False) -> IRSolution:
        """
        Args:
            V:             (R, K) row drive voltages (V)
            tol:           relative convergence criterion (node voltages vs. max|V|, outputs vs. max|I|)
            max_iter:      iteration limit (a warning is printed if it is reached)
            return_fields: also return all word-line / bit-line node voltages u, w of shape (R, C, K)

        Returns:
            IRSolution with I (K, C), I_wl (K, R), iters and optionally u, w
        """
        R, C, r = self.R, self.C, self.r
        V = np.ascontiguousarray(V, dtype=np.float64)
        if V.ndim != 2 or V.shape[0] != R:
            raise ValueError(f"V must have shape ({R}, K), got {V.shape}")
        K = V.shape[1]
        u = np.repeat(V[:, None, :], C, axis=1)
        w = np.zeros((R, C, K))
        tmp = np.empty((R, C, K))
        out = np.zeros(3)
        vmax = max(float(np.abs(V).max()), 1e-30)
        lib = load_rect_kernel()
        it = 0
        for it in range(1, max_iter + 1):
            lib.ir_sor_iter_rect(self.G, self.mwl, self.mbl, V, u, w, tmp, R, C, K, r, self.omega, out)
            imax = max(float(np.abs(w[R - 1]).max()) / r, 1e-30)
            if max(out[0], out[1]) < tol * vmax and out[2] < tol * imax:
                break
        else:
            print(f"  [warn] rectangular IR solver not converged in {max_iter} iters (R={R}, C={C})", flush=True)
        del tmp
        sol = IRSolution(I=(w[R - 1] / r).T.copy(), I_wl=((V - u[:, 0, :]) / r).T.copy(), iters=it)
        if return_fields:
            sol.u, sol.w = u, w
        return sol

    def column_gain(self, tol: float = 1e-6, max_iter: int = 500) -> np.ndarray:
        """Per-column gain: IR response to an all-ones (1 V) input divided by the ideal column sum, (C,)."""
        I1 = self.solve(np.ones((self.R, 1)), tol=tol, max_iter=max_iter).I[0]
        return I1 / self.G.sum(axis=0)


def make_ir_solver(G: np.ndarray, r_wire: float):
    """
    Return the nodal solver for an array of any shape.

    Square arrays use the original IRSolver (bit-identical to all earlier results); rectangular
    arrays use RectIRSolver. Both expose solve() and column_gain() with the same signatures.
    """
    G = np.asarray(G)
    if G.ndim == 2 and G.shape[0] == G.shape[1]:
        return IRSolver(G, r_wire)
    return RectIRSolver(G, r_wire)


def exact_solve_rect(G: np.ndarray, V: np.ndarray, r_wire: float) -> np.ndarray:
    """
    Sparse direct solve of the complete 2·R·C node equations (reference solution).

    Args:
        G: (R, C) conductances (S); V: (R, K) drive voltages (V); r_wire: wire resistance per segment (Ohm)

    Returns:
        (K, C) column output currents (A)
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    R, C = G.shape
    gr = 1.0 / r_wire
    iu = np.arange(R * C).reshape(R, C)
    iw = R * C + iu
    rows, cols, vals = [], [], []

    def add(a, b, v):
        rows.append(a.ravel()); cols.append(b.ravel()); vals.append(np.broadcast_to(v, a.shape).ravel())

    add(iu, iu, G); add(iw, iw, G); add(iu, iw, -G); add(iw, iu, -G)       # cells
    add(iu[:, 0], iu[:, 0], gr)                                              # driver segment
    a, b = iu[:, :-1], iu[:, 1:]                                             # word-line segments
    add(a, a, gr); add(b, b, gr); add(a, b, -gr); add(b, a, -gr)
    a, b = iw[:-1, :], iw[1:, :]                                             # bit-line segments
    add(a, a, gr); add(b, b, gr); add(a, b, -gr); add(b, a, -gr)
    add(iw[-1, :], iw[-1, :], gr)                                            # TIA segment
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * R * C, 2 * R * C))
    rhs = np.zeros((2 * R * C, V.shape[1]))
    rhs[iu[:, 0]] = V * gr
    x = spla.splu(A).solve(rhs)
    return (x[iw[-1, :]] * gr).T
