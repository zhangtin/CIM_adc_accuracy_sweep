# -*- coding: utf-8 -*-
"""
Regression tests against the original scripts, with portable floating-point summary comparisons.

Run:  python -m unittest tests.test_regression -v      (from the repository root)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
sys.path.insert(0, str(ROOT))

from cim_sim import SweepConfig  # noqa: E402
from cim_sim.analysis import knee_underutil  # noqa: E402
from cim_sim.experiments.sweep import run_sweep  # noqa: E402
from cim_sim.experiments.underutil import UnderutilConfig, run_underutil  # noqa: E402
from cim_sim.selfcheck import self_check  # noqa: E402

ENERGY_CSV = str(ROOT / "ADC_22nm_energy_vs_bits.csv")
SWEEP_FILES = ("raw_trials.csv.gz", "summary.csv", "knee.csv", "ir_calibration.csv")


class RegressionTest(unittest.TestCase):

    def _assert_same_csv(self, golden: Path, new: Path):
        a, b = pd.read_csv(golden), pd.read_csv(new)
        self.assertEqual(list(a.columns), list(b.columns), f"column mismatch in {golden.name}")
        self.assertEqual(a.shape, b.shape, f"shape mismatch in {golden.name}")
        # Trial CSVs and under-utilization outputs stay exact.
        # Full-precision sweep aggregates/calibration vary in their last digits across BLAS,
        # CPU/compiler and pandas versions even when the serialized trials match.
        full_precision = golden.parent.name in {"sweep_ir_noise", "sweep_sampled"} and golden.name in {
            "summary.csv", "knee.csv", "ir_calibration.csv"}
        if not full_precision or os.environ.get("CIM_SIM_STRICT_REGRESSION") == "1":
            pd.testing.assert_frame_equal(a, b, check_exact=True, obj=str(golden))
            return
        for col in a.columns:
            if pd.api.types.is_float_dtype(a[col]):
                finite = np.abs(a[col].to_numpy()[np.isfinite(a[col].to_numpy())])
                scale = float(finite.max()) if finite.size else 0.0
                # Scale the absolute allowance per column, so femtojoule energy
                # values cannot disappear under a unit-independent tolerance.
                pd.testing.assert_series_equal(a[col], b[col], check_exact=False,
                                               rtol=1e-10, atol=1e-12 * scale, obj=f"{golden}:{col}")
            else:
                pd.testing.assert_series_equal(a[col], b[col], check_exact=True, obj=f"{golden}:{col}")

    def _run_sweep(self, golden_dir: str, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            run_sweep(SweepConfig(energy_csv=ENERGY_CSV, out_dir=tmp, **kw))
            for f in SWEEP_FILES:
                self._assert_same_csv(GOLDEN / golden_dir / f, Path(tmp) / f)

    def test_selfcheck(self):
        self_check()

    def test_sweep_ir_noise(self):
        """IR drop + read noise + chip calibration, exact and probe-based stat calibration."""
        self._run_sweep("sweep_ir_noise", sizes=[16, 32], mem_bits=[2, 4], mem_sigma=[0.0, 0.02], n_mc=10,
                        adc_bits=[4, 8], chip_cal=True, cal_exact_max=16)

    def test_sweep_sampled_calibration(self):
        """Ideal interconnect, sampled calibration, chunked array programming."""
        self._run_sweep("sweep_sampled", sizes=[16, 32], mem_bits=[3], mem_sigma=[0.05], n_mc=10,
                        adc_bits=[6, 10], r_wire=0.0, cal_mode="sampled", chunk_elems=3000)

    def test_underutil(self):
        cfg = UnderutilConfig(hw_sizes=[32, 64], utils=[1.0, 0.5], mem_bits=[4], mem_sigma=[0.02], n_mc=20,
                              extra_pairs=[(32, 20)], adc_bits=[4, 8, 12])
        df = run_underutil(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            df.to_csv(Path(tmp) / "summary.csv", index=False, float_format="%.6g")
            knee_underutil(df, cfg.knee_tol).to_csv(Path(tmp) / "knee.csv", index=False, float_format="%.6g")
            for f in ("summary.csv", "knee.csv"):
                self._assert_same_csv(GOLDEN / "underutil" / f, Path(tmp) / f)

    def test_main_walkthrough_output(self):
        out = subprocess.run([sys.executable, str(ROOT / "main.py")], capture_output=True, text=True,
                             cwd=ROOT, check=True).stdout
        self.assertEqual(out, (GOLDEN / "main_default.txt").read_text())


if __name__ == "__main__":
    unittest.main()
