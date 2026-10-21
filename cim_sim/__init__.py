# -*- coding: utf-8 -*-
"""
cim_sim — 忆阻器存算一体 crossbar + ADC 位宽仿真（模块化版）。

信号链与模块对应：
    weights/        理想权重 LUT → 写入噪声 → 实际阵列赋值
    inputs          输入分布、DAC
    ir_drop/        线电阻网络求解与等效线电阻标定
    read_noise      读出噪声
    crossbar        模拟 MAC（组合以上各项）
    calibration     ADC 满量程 stat 校准
    adc             量程窗口与量化
    readout         电流 → 整数 MAC、数字增益校正
    metrics / energy / analysis   指标、能量、汇总与拐点
    experiments/    主扫描与 under-utilization 实验驱动
    plotting/       作图
"""
from .config import (CalibParams, Config, DeviceParams, InputParams, IRParams, NoiseParams,
                     SweepConfig)
from .constants import ENERGY_TIERS, FS_MODES, INPUT_DISTS

__version__ = "4.0.0"
__all__ = ["CalibParams", "Config", "DeviceParams", "InputParams", "IRParams", "NoiseParams", "SweepConfig",
           "ENERGY_TIERS", "FS_MODES", "INPUT_DISTS"]
