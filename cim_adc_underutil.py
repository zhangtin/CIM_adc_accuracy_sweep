#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""under-utilization 实验入口（命令行参数与原 cim_adc_underutil.py 相同）。实现见 cim_sim/experiments/underutil.py。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cim_sim.experiments.underutil import main  # noqa: E402

if __name__ == "__main__":
    main()
