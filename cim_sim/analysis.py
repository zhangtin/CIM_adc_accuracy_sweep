# -*- coding: utf-8 -*-
"""
结果汇总：逐 trial 表 → 每个配置的统计量（summarize），以及拐点 ADC 位宽（knee）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ENERGY_TIERS

SWEEP_KEYS = ["input_dist", "mem_bits", "mem_sigma", "adc_bits", "N", "fs_mode"]
KNEE_KEYS = ["input_dist", "mem_bits", "mem_sigma", "N", "fs_mode"]
UNDERUTIL_KNEE_KEYS = ["input_dist", "mem_bits", "mem_sigma", "N_hw", "n"]


def summarize(raw: pd.DataFrame, keys: list = SWEEP_KEYS) -> pd.DataFrame:
    """
    主扫描逐 trial 表（raw_trials.csv.gz）→ summary.csv。

    每个配置给出各指标的 mean / std、nrmse 与 max_abs_err 的 95 分位，
    以及按 MSE 合并（pooled）的有效位与 SNR（*_pooled，比逐 trial 平均更稳健）。
    """
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
    g = raw.groupby(keys, sort=True)
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


def knee_table(summary: pd.DataFrame, knee_tol: float = 0.25) -> pd.DataFrame:
    """
    拐点：满量程参考有效位达到（最大 ADC 位宽结果 − knee_tol）的最小 ADC 位宽。
    summary 中有 *_cal / *_chip 列时另给对应版本。
    """
    rows = []
    has_cal = "eff_bits_cal_pooled" in summary.columns
    for key, s in summary.groupby(KNEE_KEYS):
        s = s.sort_values("adc_bits")
        ceiling = s["eff_bits_pooled"].iloc[-1]
        k = s[s["eff_bits_pooled"] >= ceiling - knee_tol].iloc[0]
        row = dict(zip(KNEE_KEYS, key),
                   knee_adc_bits=int(k.adc_bits), eff_bits_at_knee=k.eff_bits_pooled,
                   eff_bits_ceiling=ceiling, eff_bits_sig_at_knee=k.eff_bits_sig_pooled,
                   E_total_typical_per_MAC_at_knee_J=k.E_total_typical_per_MAC_J_mean,
                   adc_energy_share_at_knee=k.adc_energy_share_typical_mean)
        if has_cal:
            cc = s["eff_bits_cal_pooled"].iloc[-1]
            kc = s[s["eff_bits_cal_pooled"] >= cc - knee_tol].iloc[0]
            row.update(knee_adc_bits_cal=int(kc.adc_bits), eff_bits_cal_at_knee=kc.eff_bits_cal_pooled,
                       eff_bits_cal_ceiling=cc, eff_bits_sig_cal_at_knee=kc.eff_bits_sig_cal_pooled,
                       E_total_typical_per_MAC_at_knee_cal_J=kc.E_total_typical_per_MAC_J_mean)
        if "eff_bits_chip_pooled" in s.columns:
            cc = s["eff_bits_chip_pooled"].iloc[-1]
            kc = s[s["eff_bits_chip_pooled"] >= cc - knee_tol].iloc[0]
            row.update(knee_adc_bits_chip=int(kc.adc_bits), eff_bits_chip_ceiling=cc,
                       eff_bits_sig_chip_at_knee=kc.eff_bits_sig_chip_pooled)
        rows.append(row)
    return pd.DataFrame(rows)


def knee_underutil(df: pd.DataFrame, tol: float = 0.25, ref_strategy: str = "stat_matched") -> pd.DataFrame:
    """
    under-utilization 实验的拐点：信号参考有效位达到（ref_strategy 在最大 ADC 位宽下的结果 − tol）
    的最小 ADC 位宽；达不到记为 NaN。
    """
    out = []
    for k, s in df.groupby(UNDERUTIL_KNEE_KEYS):
        ref = s[(s.strategy == ref_strategy)].sort_values("adc_bits")["eff_bits_sig"].iloc[-1]
        for strat, ss in s.groupby("strategy"):
            ss = ss.sort_values("adc_bits")
            ok = ss[ss.eff_bits_sig >= ref - tol]
            out.append(dict(zip(UNDERUTIL_KNEE_KEYS, k), strategy=strat, util=ss.util.iloc[0], ref_eff_bits=ref,
                            knee_adc_bits=int(ok.adc_bits.iloc[0]) if len(ok) else np.nan,
                            eff_bits_at_12=ss.eff_bits_sig.iloc[-1]))
    return pd.DataFrame(out)
