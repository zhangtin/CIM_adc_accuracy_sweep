# -*- coding: utf-8 -*-
"""作图共用的配色与坐标轴样式。"""
from __future__ import annotations

import numpy as np

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e4e3df", "#b5b4ae"
BLUE_ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#8a6fd1", "#d4a017"]


def ordinal_colors(k: int) -> list[str]:
    """k 个由浅到深的蓝色（有序变量，如忆阻器位宽）。"""
    if k <= 1:
        return [BLUE_ORDINAL[3]]
    idx = np.linspace(0, len(BLUE_ORDINAL) - 1, k).round().astype(int)
    return [BLUE_ORDINAL[i] for i in idx]


def style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)


def label_axes(ax, x: str | None = None, y: str | None = None) -> None:
    if x:
        ax.set_xlabel(x, fontsize=8, color=INK2)
    if y:
        ax.set_ylabel(y, fontsize=8, color=INK2)


def use_agg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt
