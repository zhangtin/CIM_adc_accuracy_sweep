# -*- coding: utf-8 -*-
"""主扫描的图（fig_effbits_vs_adc_*, fig_pareto_total*, fig_scaling_trends*, fig_nonideal_scaling）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import knee_table
from ..config import SweepConfig
from .style import CATEGORICAL, label_axes, ordinal_colors, style_axes, use_agg


def make_plots(summary: pd.DataFrame, cfg: SweepConfig, cal: bool = False) -> None:
    """
    Args:
        summary: analysis.summarize() 的输出
        cfg:     SweepConfig（用到 out_dir、mem_sigma、plot_*、fs_modes、input_dists、v_read、g_max、knee_tol）
        cal:     True 时画逐列增益校正后的版本（文件名带 _cal）
    """
    matplotlib, plt = use_agg()
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
    mcol = dict(zip(mb, ordinal_colors(len(mb))))
    dcol = dict(zip(dists, CATEGORICAL))

    # ---- 图 1：有效位 vs ADC 位宽 ----
    for mode in cfg.fs_modes:
        fig, axes = plt.subplots(len(dists), len(sizes), figsize=(3.1 * len(sizes), 2.6 * len(dists)),
                                 sharex=True, sharey=True, squeeze=False)
        for r, dist in enumerate(dists):
            for c, n in enumerate(sizes):
                ax = axes[r][c]
                style_axes(ax)
                for b in mb:
                    s = d[(d.input_dist == dist) & (d.fs_mode == mode) & (d.N == n) & (d.mem_bits == b)]
                    s = s.sort_values("adc_bits")
                    ax.plot(s.adc_bits, s[EBS], color=mcol[b], lw=2, marker="o", ms=4, label=f"{b}-bit")
                ax.set_title(f"{dist}, N={n}", fontsize=9, color="#0b0b0b")
                label_axes(ax, "ADC bits" if r == len(dists) - 1 else None,
                           "signal-referred eff. bits" if c == 0 else None)
        axes[0][-1].legend(title="memristor bits", fontsize=7, title_fontsize=7, frameon=False, loc="lower right")
        fig.suptitle(f"Signal-referred effective bits vs ADC resolution  (FS={mode}, σ = {sig:.0%}, IR drop + noise{tag})",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"fig_effbits_vs_adc_{mode}{sfx}.png", dpi=150)
        plt.close(fig)

    # ---- 图 2：精度 vs 总能量 ----
    mode = "stat" if "stat" in cfg.fs_modes else cfg.fs_modes[0]
    fig, axes = plt.subplots(len(dists), len(sizes), figsize=(3.1 * len(sizes), 2.7 * len(dists)),
                             sharey=True, squeeze=False)
    for r, dist in enumerate(dists):
        for c, n in enumerate(sizes):
            ax = axes[r][c]
            style_axes(ax)
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
            label_axes(ax, "total energy per MAC (fJ, typical)" if r == len(dists) - 1 else None,
                       "signal-referred eff. bits" if c == 0 else None)
    axes[0][-1].legend(title="memristor bits", fontsize=7, title_fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"Accuracy vs total energy (ADC + array), FS={mode}, σ = {sig:.0%}, IR drop + noise{tag}; "
                 f"points along each line = ADC 4→12 bit", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"fig_pareto_total{sfx}.png", dpi=150)
    plt.close(fig)

    # ---- 图 3：规模趋势 ----
    b0 = cfg.plot_mem_bits if cfg.plot_mem_bits in mb else mb[len(mb) // 2]
    B_ref = 8 if 8 in cfg.adc_bits else cfg.adc_bits[len(cfg.adc_bits) // 2]
    knee = knee_table(summary, cfg.knee_tol)
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
        style_axes(ax)
        ax.set_xscale("log", base=2)
        ax.set_title(t, fontsize=9, color="#0b0b0b")
        label_axes(ax, "array size N", yl)
    axes[0].legend(title="input", fontsize=7, title_fontsize=7, frameon=False)
    axes[3].legend(fontsize=6, frameon=False, ncol=1, loc="best")
    fig.suptitle(f"Scaling trends ({b0}-bit memristor, σ = {sig:.0%}, IR drop + noise{tag})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"fig_scaling_trends{sfx}.png", dpi=150)
    plt.close(fig)

    # ---- 图 4：非理想量随规模的变化与误差分解 ----
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
        for (col, lab), colr in zip(comps, CATEGORICAL):
            y_fs = s.N * (2 ** b0 - 1) ** 2
            axes[2].plot(s.N, s[col] / y_fs, color=colr, lw=2, marker="o", ms=5, label=lab)
        axes[2].set_yscale("log")
        for ax, t, yl in zip(axes, ["mean IR-drop signal attenuation", "read-noise rms / FS current",
                                    f"error breakdown ({s.input_dist.iloc[0]})"],
                             ["attenuation (%)", "ppm of FS", "rmse / MAC full scale"]):
            style_axes(ax)
            ax.set_xscale("log", base=2)
            ax.set_title(t, fontsize=9, color="#0b0b0b")
            label_axes(ax, "array size N", yl)
        axes[0].legend(title="input", fontsize=7, title_fontsize=7, frameon=False)
        axes[2].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Circuit non-idealities vs array size ({b0}-bit memristor, σ = {sig:.0%})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "fig_nonideal_scaling.png", dpi=150)
        plt.close(fig)
