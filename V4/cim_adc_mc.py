#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主扫描入口（命令行参数与原 cim_adc_mc.py v3 相同）。实现见 cim_sim/experiments/sweep.py。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cim_sim.experiments.sweep import main  # noqa: E402

if __name__ == "__main__":
    main()
