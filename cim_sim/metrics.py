# -*- coding: utf-8 -*-
"""
精度指标。输入一般为 (trials, N) 的 MAC 结果，按 trial 逐行统计。

    eff_bits      满量程参考有效位  log2(Y_FS / (rmse·√12))
    eff_bits_sig  信号参考有效位    (SNR_dB − 1.76) / 6.02，SNR = std(Y_true) / rmse
"""
from __future__ import annotations

import numpy as np


def rms(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.sqrt(np.mean(x ** 2, axis=axis))


def fs_effective_bits(rmse, y_fullscale):
    """满量程参考有效位。"""
    return np.log2(y_fullscale / (np.maximum(rmse, 1e-12) * np.sqrt(12)))


def snr_to_bits(snr_db):
    """SNR (dB) → 有效位。"""
    return (snr_db - 1.76) / 6.02


def mac_fullscale(n: int, bits: int) -> int:
    """MAC 结果满量程 N·(2^b − 1)²。"""
    return n * (2 ** bits - 1) ** 2


def trial_metrics(Y_true, Y_prog, Y_ir, Y_analog, Y_adc, sat, y_fullscale) -> dict:
    """
    逐 trial 指标及误差分解。

    Args:
        Y_true:   (T, N) 理想整数 MAC
        Y_prog:   (T, N) 仅写入误差
        Y_ir:     (T, N) 写入误差 + IR drop
        Y_analog: (T, N) ADC 之前（写入误差 + IR + 噪声）
        Y_adc:    (T, N) ADC 之后
        sat:      (T, N) 削顶掩码
        y_fullscale: MAC 满量程（mac_fullscale）
    """
    e = Y_adc - Y_true
    rmse = rms(e)
    eff_bits = fs_effective_bits(rmse, y_fullscale)
    sig_std = np.std(Y_true, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_db = 20 * np.log10(np.maximum(sig_std, 1e-12) / np.maximum(rmse, 1e-12))
    return dict(
        rmse=rmse,
        nrmse=rmse / y_fullscale,
        max_abs_err=np.max(np.abs(e), axis=1),
        mean_err=np.mean(e, axis=1),
        rmse_device=rms(Y_analog - Y_true),     # ADC 之前的全部模拟误差
        rmse_prog=rms(Y_prog - Y_true),         # 仅写入误差
        rmse_ir=rms(Y_ir - Y_prog),             # IR drop 增量
        rmse_noise=rms(Y_analog - Y_ir),        # 读出噪声增量
        rmse_adc=rms(Y_adc - Y_analog),
        exact_rate=np.mean(np.rint(Y_adc) == Y_true, axis=1),
        sat_rate=np.mean(sat, axis=1),
        eff_bits=eff_bits,
        snr_db=snr_db,
        eff_bits_sig=snr_to_bits(snr_db),
        sig_std=sig_std,
    )


def corrected_metrics(Y_true, Y_prog, Y_ir_corr, Y_adc_corr, y_fullscale, suffix: str = "cal") -> dict:
    """
    数字增益校正后的指标（列名带后缀 _cal / _chip）。

    Args:
        Y_ir_corr:  校正后的（写入误差 + IR）MAC
        Y_adc_corr: 校正后的 ADC 输出 MAC
    """
    e = Y_adc_corr - Y_true
    rmse = rms(e)
    sig_std = np.std(Y_true, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_db = 20 * np.log10(np.maximum(sig_std, 1e-12) / np.maximum(rmse, 1e-12))
    return {f"rmse_{suffix}": rmse,
            f"eff_bits_{suffix}": fs_effective_bits(rmse, y_fullscale),
            f"snr_db_{suffix}": snr_db, f"eff_bits_sig_{suffix}": snr_to_bits(snr_db),
            f"rmse_ir_{suffix}": rms(Y_ir_corr - Y_prog)}


def pooled_metrics(Y_true: np.ndarray, Y_hat: np.ndarray) -> tuple[float, float, float]:
    """
    跨所有 trial 合并的 (MSE, 平均信号方差, SNR_dB)。

    Args:
        Y_true, Y_hat: (T, N)
    """
    mse = np.mean((Y_hat - Y_true) ** 2)
    sigvar = np.mean(np.var(Y_true, axis=1))
    snr = 10 * np.log10(max(sigvar, 1e-24) / max(mse, 1e-24))
    return mse, sigvar, snr
