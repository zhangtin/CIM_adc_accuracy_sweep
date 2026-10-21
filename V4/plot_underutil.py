#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""under-utilization 作图入口：python plot_underutil.py res_b4。实现见 cim_sim/plotting/underutil.py。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cim_sim.plotting.underutil import main  # noqa: E402

if __name__ == "__main__":
    main()
