# -*- coding: utf-8 -*-
"""物理常数与全局枚举。"""

K_BOLTZ = 1.380649e-23          # 玻尔兹曼常数 J/K
Q_E = 1.602176634e-19           # 电子电荷 C

INPUT_DISTS = ("uniform", "sparse", "skewed")          # 输入分布
ENERGY_TIERS = ("best", "typical", "conservative")     # ADC 能量表的三档
FS_MODES = ("max", "stat")                             # ADC 满量程策略（主扫描）
