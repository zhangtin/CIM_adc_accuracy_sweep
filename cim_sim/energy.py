# -*- coding: utf-8 -*-
"""ADC 能量表读取与能量指标。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import ENERGY_TIERS


def load_adc_energy(path: str | Path) -> dict[str, dict[int, float]]:
    """
    读取 ADC 单次转换能量表（列 bits, E_best_J, E_typical_J, E_conservative_J）。

    Returns:
        {tier: {bits: J}}
    """
    df = pd.read_csv(path)
    table = {}
    for tier in ENERGY_TIERS:
        col = f"E_{tier}_J"
        if col not in df.columns:
            raise KeyError(f"能量表缺少列 {col}")
        table[tier] = dict(zip(df["bits"].astype(int), df[col].astype(float)))
    return table


def check_adc_bits(table: dict, adc_bits: list) -> None:
    """确认能量表覆盖所有要扫描的 ADC 位宽。"""
    for B in adc_bits:
        if B not in table["typical"]:
            raise ValueError(f"能量表中没有 {B} bit ADC 的数据")


def add_energy_columns(df: pd.DataFrame, n: int, adc_bits: int, table: dict, E_array) -> pd.DataFrame:
    """
    给逐次结果表加能量列（就地修改并返回 df）。

    一次 N×N MVM：N 个 ADC 各转换一次 + 阵列读能量。
        E_adc_<tier>_J, E_total_<tier>_J, E_adc_typical_per_MAC_J, E_total_typical_per_MAC_J,
        adc_energy_share_typical

    Args:
        df:       每行一个 trial 的 DataFrame
        n:        阵列规模（列数 = ADC 个数，MAC 数 = n²）
        adc_bits: ADC 位宽
        table:    load_adc_energy() 的输出
        E_array:  (trials,) 阵列读能量 (J)
    """
    df["E_array_J"] = E_array
    for tier in ENERGY_TIERS:
        df[f"E_adc_{tier}_J"] = n * table[tier][adc_bits]
        df[f"E_total_{tier}_J"] = df[f"E_adc_{tier}_J"] + E_array
    df["E_adc_typical_per_MAC_J"] = df["E_adc_typical_J"] / (n * n)
    df["E_total_typical_per_MAC_J"] = df["E_total_typical_J"] / (n * n)
    df["adc_energy_share_typical"] = df["E_adc_typical_J"] / df["E_total_typical_J"]
    return df
