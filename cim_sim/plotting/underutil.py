# -*- coding: utf-8 -*-
"""
under-utilization 实验的图（原 plot_underutil.py）。读取 <res>/summary.csv，图写到 <res>/fig/。

    python -m cim_sim.plotting.underutil res_b4
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..experiments.underutil import UnderutilConfig, calibrate_size, simulate_active
from ..inputs import input_moments
from ..weights import conductance_lut
from .style import AXIS, GRID, INK, INK2, use_agg

CAT = {"stat_matched": "#2a78d6", "stat_hw": "#eb6834", "max_hw": "#1baf7a", "stat_hw_dummy": "#8a6fd1",
       "max_matched": "#d4a017"}
MARK = {"stat_matched": "o", "stat_hw": "s", "max_hw": "^", "stat_hw_dummy": "D", "max_matched": "v"}
LAB = {"stat_matched": "stat 按实际规模 n 校准", "stat_hw": "stat 按硬件规模 N_hw 校准",
       "max_hw": "max 按硬件规模", "stat_hw_dummy": "stat_hw + 空闲行填充", "max_matched": "max 按实际规模"}
DLAB = {"uniform": "uniform（均匀）", "sparse": "sparse（80% 为 0）", "skewed": "skewed（偏小值）"}
DISTS = ["uniform", "sparse", "skewed"]


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)


def _setup_fonts(plt):
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def fig_window_shift(out: Path, cfg: UnderutilConfig, bits: int = 4, sigma: float = 0.02,
                     n_hw: int = 64, n: int = 32, n_mc: int = 100) -> None:
    """图 1：只用 n 行时列电流分布左移，而按 N_hw 设计的量程不动。"""
    _, plt = use_agg()
    _setup_fonts(plt)
    k = cfg.stat_k
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
    cal_hw_all = calibrate_size(bits, sigma, n_hw, DISTS, cfg)
    cal_n_all = calibrate_size(bits, sigma, n, DISTS, cfg)
    sims = {m: simulate_active(bits, sigma, m, n_mc, DISTS, cfg) for m in (n_hw, n)}
    for ax, dist in zip(axes, DISTS):
        _style(ax)
        di = DISTS.index(dist)
        for m, col, lab in ((n_hw, "#86b6ef", f"{n_hw} 行全部参与"), (n, "#184f95", f"只有 {n} 行参与")):
            I = sims[m]["I"][:, di].ravel() * 1e6
            ax.hist(I, bins=80, density=True, color=col, alpha=0.75, label=lab, edgecolor="none")
        ymax = ax.get_ylim()[1]
        for (mu, sd), col, lab, yy in ((cal_hw_all[dist], CAT["stat_hw"], f"按 {n_hw} 行校准的 ADC 量程", 1.12),
                                       (cal_n_all[dist], CAT["stat_matched"], f"按 {n} 行校准的 ADC 量程", 1.02)):
            lo, hi = max(0, mu - k * sd) * 1e6, (mu + k * sd) * 1e6
            ax.plot([lo, hi], [ymax * yy] * 2, color=col, lw=4, solid_capstyle="butt", label=lab)
            ax.plot([lo, lo], [0, ymax * yy], color=col, lw=1, ls="--")
            ax.plot([hi, hi], [0, ymax * yy], color=col, lw=1, ls="--")
        ax.set_ylim(0, ymax * 1.22)
        ax.set_yticks([])
        ax.set_title(DLAB[dist], fontsize=10, color=INK)
        ax.set_xlabel("列电流 (µA)", fontsize=8, color=INK2)
        ax.set_xlim(left=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"图 1  {n_hw}×{n_hw} 硬件只用 {n} 行时，列电流分布整体左移，而按 {n_hw} 行设计的 ADC 量程没动"
                 f"（{bits}-bit 忆阻器，σ={sigma:.0%}）", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig1_window_shift.png", dpi=160)
    plt.close(fig)


def fig_n_sweep(df: pd.DataFrame, out: Path, n_hw: int = 64, bits: int = 4, sigma: float = 0.02,
                adc_bits: int = 8) -> None:
    """图 2：固定硬件规模，参与行数 n 变化时各策略的精度。"""
    _, plt = use_agg()
    _setup_fonts(plt)
    strats = ["stat_matched", "stat_hw", "stat_hw_dummy", "max_hw"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    d = df[(df.N_hw == n_hw) & (df.mem_sigma == sigma) & (df.mem_bits == bits) & (df.adc_bits == adc_bits)]
    for ax, dist in zip(axes, DISTS):
        _style(ax)
        for s in strats:
            x = d[(d.input_dist == dist) & (d.strategy == s)].sort_values("n")
            ax.plot(x.n, x.eff_bits_sig, color=CAT[s], marker=MARK[s], ms=5, lw=2, label=LAB[s])
        ax.axvline(n_hw // 2, color=AXIS, lw=1, ls=":")
        ax.set_title(DLAB[dist], fontsize=10, color=INK)
        ax.set_xlabel(f"实际参与运算的行数 n（硬件 {n_hw} 行）", fontsize=8, color=INK2)
        ax.set_ylim(-4.5, 5)
        ax.set_xticks(list(range(n_hw // 8, n_hw + 1, n_hw // 8)))
    axes[0].set_ylabel("有效位（信号参考，bit）", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"图 2  {n_hw}×{n_hw} 硬件、{adc_bits}-bit ADC：参与行数减少时各量程策略的精度"
                 f"（{bits}-bit 忆阻器，σ={sigma:.0%}）", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig2_n64_sweep.png", dpi=160)
    plt.close(fig)


def fig_loss_heatmap(df: pd.DataFrame, out: Path, cfg: UnderutilConfig, bits: int = 4, sigma: float = 0.02,
                     adc_bits: int = 8,
                     utils: tuple = (1.0, 0.96875, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125),
                     hw_sizes: tuple = (64, 128, 256, 512, 1024, 2048, 4096)) -> None:
    """图 3：stat_hw 相对 stat_matched 的有效位损失热图，附解析临界线。"""
    _, plt = use_agg()
    _setup_fonts(plt)
    from matplotlib.colors import LinearSegmentedColormap
    U, NS = list(utils), list(hw_sizes)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#f6c3a9", "#eb6834", "#a8401b", "#5c2008"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    d = df[(df.mem_sigma == sigma) & (df.mem_bits == bits) & (df.adc_bits == adc_bits)]
    G = conductance_lut(bits, cfg.g_min, cfg.g_max)
    for ax, dist in zip(axes, DISTS):
        M = np.full((len(U), len(NS)), np.nan)
        for i, u in enumerate(U):
            for j, N in enumerate(NS):
                x = d[(d.input_dist == dist) & (d.N_hw == N) & (d.util_r == round(u, 5))]
                if len(x):
                    hw = x[x.strategy == "stat_hw"].eff_bits_sig.iloc[0]
                    mt = x[x.strategy == "stat_matched"].eff_bits_sig.iloc[0]
                    M[i, j] = max(mt - hw, 0)
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=8, aspect="auto")
        for i in range(len(U)):
            for j in range(len(NS)):
                if not np.isnan(M[i, j]):
                    v = M[i, j]
                    ax.text(j, i, f"{v:.1f}" if v >= 0.05 else "0", ha="center", va="center", fontsize=7,
                            color="white" if v > 3.5 else INK)
        # 解析临界线：列电流均值恰好落到量程下沿 (1 − u) = k·c/√N
        ex, ex2 = input_moments(dist, bits, **cfg.inputs.as_kwargs())
        c = np.sqrt(ex2 * (G ** 2).mean() / (ex * ex * G.mean() ** 2) - 1)
        ucrit = 1 - cfg.stat_k * c / np.sqrt(np.array(NS, float))
        yy = np.interp(ucrit, U[::-1], np.arange(len(U))[::-1])
        ok = ucrit > U[-1]
        ax.plot(np.arange(len(NS))[ok], yy[ok], color=INK, lw=1.5, ls="--")
        ax.set_xticks(range(len(NS)))
        ax.set_xticklabels(NS, fontsize=8)
        ax.set_yticks(range(len(U)))
        ax.set_yticklabels([f"{u:.0%}" if u * 100 == int(u * 100) else f"{u:.1%}" for u in U], fontsize=8)
        ax.set_xlabel("硬件规模 N_hw", fontsize=8, color=INK2)
        ax.set_title(DLAB[dist], fontsize=10, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK2, length=0)
    axes[0].set_ylabel("行利用率 n / N_hw", fontsize=8, color=INK2)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("比按实际规模校准少的有效位 (bit)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"图 3  stat 量程按硬件规模设计造成的有效位损失（{adc_bits}-bit ADC，{bits}-bit 忆阻器，σ={sigma:.0%}）；"
                 "虚线 = 解析临界线（列电流均值恰好落到量程下沿）", fontsize=10, color=INK)
    fig.savefig(out / "fig3_loss_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_vs_adc_bits(df: pd.DataFrame, out: Path, bits: int = 4, sigma: float = 0.02,
                    cases: tuple = ((64, 0.5, "uniform"), (1024, 0.875, "uniform"), (1024, 0.5, "sparse"))) -> None:
    """图 4：增加 ADC 位宽能否弥补。"""
    _, plt = use_agg()
    _setup_fonts(plt)
    strats = ["stat_matched", "stat_hw", "stat_hw_dummy", "max_hw"]
    fig, axes = plt.subplots(1, len(cases), figsize=(4 * len(cases), 3.5), sharey=True, squeeze=False)
    axes = axes[0]
    d = df[(df.mem_sigma == sigma) & (df.mem_bits == bits)]
    for ax, (N, u, dist) in zip(axes, cases):
        _style(ax)
        for s in strats:
            x = d[(d.input_dist == dist) & (d.N_hw == N) & (d.util_r == round(u, 5)) & (d.strategy == s)]
            x = x.sort_values("adc_bits")
            ax.plot(x.adc_bits, x.eff_bits_sig, color=CAT[s], marker=MARK[s], ms=5, lw=2, label=LAB[s])
        ax.set_title(f"N_hw={N}，n={int(N * u)}，{DLAB[dist]}", fontsize=9, color=INK)
        ax.set_xlabel("ADC 位宽 (bit)", fontsize=8, color=INK2)
        ax.set_xticks(sorted(d.adc_bits.unique()))
    axes[0].set_ylabel("有效位（信号参考，bit）", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"图 4  增加 ADC 位宽能否弥补？（{bits}-bit 忆阻器，σ={sigma:.0%}）", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig4_vs_adc_bits.png", dpi=160)
    plt.close(fig)


def make_underutil_plots(res_dir: str | Path, cfg: UnderutilConfig | None = None) -> Path:
    """读取 res_dir/summary.csv，生成全部 4 张图到 res_dir/fig/，返回图目录。"""
    cfg = cfg or UnderutilConfig()
    res = Path(res_dir)
    out = res / "fig"
    out.mkdir(exist_ok=True)
    df = pd.read_csv(res / "summary.csv")
    df["util_r"] = df.util.round(5)
    fig_window_shift(out, cfg)
    fig_n_sweep(df, out)
    fig_loss_heatmap(df, out, cfg)
    fig_vs_adc_bits(df, out)
    return out


def main(argv: list | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    out = make_underutil_plots(argv[0] if argv else "res_b4")
    print("figures →", out.resolve())


if __name__ == "__main__":
    main()
