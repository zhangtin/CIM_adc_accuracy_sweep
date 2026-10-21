# -*- coding: utf-8 -*-
"""
参数定义。

两层结构：
  * 参数组（frozen dataclass）——按物理含义分组，供模块函数直接接收：
        DeviceParams   器件 / 读出电路常数
        InputParams    输入分布参数
        IRParams       线电阻 / IR drop 求解参数
        NoiseParams    读出噪声参数
        CalibParams    ADC 满量程 stat 校准参数
  * SweepConfig ——主扫描（原 cim_adc_mc.py v3）的完整配置，字段名与原 Config 一致，
    因此原来的 config.json 可以直接载入；通过 .device / .inputs / .ir / .noise / .calib
    取出对应参数组。

所有参数组都有 .as_kwargs()，返回可直接 ** 展开给底层函数的字典。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .constants import FS_MODES, INPUT_DISTS


class _Params:
    def as_kwargs(self, exclude: tuple[str, ...] = ()) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in exclude}


@dataclass(frozen=True)
class DeviceParams(_Params):
    g_min: float = 1e-6        # 最小电导 (S)，对应权重 0
    g_max: float = 100e-6      # 最大电导 (S)，对应权重 2^b − 1
    v_read: float = 0.2        # DAC 满量程读电压 (V)
    t_read: float = 10e-9      # 单次读出时间 (s)


@dataclass(frozen=True)
class InputParams(_Params):
    sparsity: float = 0.8      # sparse 分布中 0 的比例
    skew_a: float = 0.5        # skewed 分布 Beta(a, b) 参数
    skew_b: float = 3.0


@dataclass(frozen=True)
class IRParams(_Params):
    r_wire: float = -1.0       # <0：按 A(N) 标定等效线电阻；0：关闭 IR drop；>0：固定线电阻 Ω/段
    atten_lo: float = 0.01     # A(n_lo)
    atten_hi: float = 0.10     # A(n_hi)
    n_lo: int = 32
    n_hi: int = 4096
    tol: float = 1e-6          # SOR 相对收敛判据
    max_iter: int = 500


@dataclass(frozen=True)
class NoiseParams(_Params):
    enabled: bool = True
    read_noise_rel: float = 0.01   # 单元读噪声（相对单元电流）
    temp_K: float = 300.0
    tia_en: float = 10e-9          # TIA 输入电压噪声 V/√Hz
    tia_in: float = 1e-12          # TIA 输入电流噪声 A/√Hz

    def model_kwargs(self) -> dict:
        """read_noise.read_noise_std 需要的噪声模型参数（不含 enabled）。"""
        return self.as_kwargs(exclude=("enabled",))


@dataclass(frozen=True)
class CalibParams(_Params):
    stat_k: float = 4.0            # stat 量程 = μ ± k·σ
    n_cal_arrays: int = 4          # 独立校准阵列数
    n_cal_inputs: int = 100        # 仅 mode = "sampled" 使用
    mode: str = "analytic"         # "analytic"（半解析）/ "sampled"（抽样，仅 IR 关闭时）
    probes: int = 32               # N > exact_max 时 Rademacher 探针数
    exact_max: int = 256           # N ≤ exact_max 时精确求传输矩阵


@dataclass
class SweepConfig:
    """主扫描配置（字段名与原 cim_adc_mc.py v3 的 Config 完全一致）。"""
    # ---- 扫描变量 ----
    mem_bits: list = field(default_factory=lambda: [1, 2, 3, 4, 6, 8])
    mem_sigma: list = field(default_factory=lambda: [0.0, 0.01, 0.02, 0.05])
    adc_bits: list = field(default_factory=lambda: [4, 6, 8, 10, 12])
    sizes: list = field(default_factory=lambda: [32, 64, 128, 256, 512, 1024, 2048, 4096])
    fs_modes: list = field(default_factory=lambda: list(FS_MODES))
    input_dists: list = field(default_factory=lambda: list(INPUT_DISTS))
    n_mc: int = 100

    # ---- 输入分布 ----
    sparsity: float = 0.8
    skew_a: float = 0.5
    skew_b: float = 3.0

    # ---- 器件 / 电路 ----
    g_min: float = 1e-6
    g_max: float = 100e-6
    v_read: float = 0.2
    t_read: float = 10e-9

    # ---- IR drop ----
    r_wire: float = -1.0
    ir_atten_lo: float = 0.01
    ir_atten_hi: float = 0.10
    ir_n_lo: int = 32
    ir_n_hi: int = 4096
    ir_tol: float = 1e-6
    ir_max_iter: int = 500

    # ---- 噪声 ----
    chip_cal: bool = False
    noise: bool = True
    read_noise_rel: float = 0.01
    temp_K: float = 300.0
    tia_en: float = 10e-9
    tia_in: float = 1e-12

    # ---- ADC 满量程统计校准 ----
    stat_k: float = 4.0
    n_cal_arrays: int = 4
    n_cal_inputs: int = 100
    cal_mode: str = "analytic"
    cal_probes: int = 32
    cal_exact_max: int = 256

    # ---- 其他 ----
    energy_csv: str = "ADC_22nm_energy_vs_bits.csv"
    seed: int = 20260929
    out_dir: str = "results"
    chunk_elems: int = 2 ** 24
    plot_sigma: float = 0.02
    plot_sizes: list = field(default_factory=lambda: [32, 256, 1024, 4096])
    plot_mem_bits: int = 4
    knee_tol: float = 0.25

    # ---- 参数组视图 ----
    @property
    def device(self) -> DeviceParams:
        return DeviceParams(self.g_min, self.g_max, self.v_read, self.t_read)

    @property
    def inputs(self) -> InputParams:
        return InputParams(self.sparsity, self.skew_a, self.skew_b)

    @property
    def ir(self) -> IRParams:
        return IRParams(self.r_wire, self.ir_atten_lo, self.ir_atten_hi, self.ir_n_lo, self.ir_n_hi,
                        self.ir_tol, self.ir_max_iter)

    @property
    def noise_params(self) -> NoiseParams:
        return NoiseParams(self.noise, self.read_noise_rel, self.temp_K, self.tia_en, self.tia_in)

    @property
    def calib(self) -> CalibParams:
        return CalibParams(self.stat_k, self.n_cal_arrays, self.n_cal_inputs, self.cal_mode,
                           self.cal_probes, self.cal_exact_max)

    # ---- 读写 ----
    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def from_json(cls, path: str | Path) -> "SweepConfig":
        d = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# 向后兼容：原脚本里叫 Config
Config = SweepConfig
