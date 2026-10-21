# -*- coding: utf-8 -*-
"""矩阵权重赋值：理想权重 LUT（lut）→ 写入噪声（write_noise）→ 实际阵列赋值（program）。"""
from .lut import (conductance_lut, conductance_step, conductance_to_levels, levels_to_conductance,
                  max_level, n_levels)
from .program import ArrayBatch, iter_programmed_arrays, program_arrays, sample_weight_levels
from .write_noise import apply_write_noise, sample_write_noise

__all__ = ["conductance_lut", "conductance_step", "conductance_to_levels", "levels_to_conductance",
           "max_level", "n_levels", "ArrayBatch", "iter_programmed_arrays", "program_arrays",
           "sample_weight_levels", "apply_write_noise", "sample_write_noise"]
