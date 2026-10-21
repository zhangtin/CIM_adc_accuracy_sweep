"""Guard the portable golden comparison against masking numerical regressions."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from tests import test_regression


class GoldenComparisonTest(unittest.TestCase):
    def compare(self, actual, *, name='summary.csv', strict=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            golden = root / 'sweep_sampled' / name
            golden.parent.mkdir()
            expected = pd.DataFrame({'N': [16, 32], 'energy_J': [1e-15, 2e-15],
                                     'rmse': [1.0, 2.0], 'dist': ['uniform', 'sparse']})
            expected.to_csv(golden, index=False)
            new = root / name
            actual(expected.copy()).to_csv(new, index=False)
            with patch.dict(os.environ, {'CIM_SIM_STRICT_REGRESSION': '1' if strict else '0'}):
                test_regression.RegressionTest()._assert_same_csv(golden, new)

    @staticmethod
    def roundoff(df):
        df.loc[0, 'rmse'] += 1e-14
        return df

    def test_summary_accepts_roundoff(self):
        for name in ("summary.csv", "knee.csv", "ir_calibration.csv"):
            with self.subTest(file=name):
                self.compare(self.roundoff, name=name)

    def test_strict_mode_and_trial_files_reject_roundoff(self):
        with self.assertRaises(AssertionError):
            self.compare(self.roundoff, strict=True)
        with self.assertRaises(AssertionError):
            self.compare(self.roundoff, name='raw_trials.csv.gz')

    def test_material_energy_metric_and_discrete_changes_fail(self):
        for col, value in [('energy_J', 1.01e-15), ('rmse', 1.001), ('N', 17), ('dist', 'skewed')]:
            with self.subTest(column=col), self.assertRaises(AssertionError):
                def change(df):
                    df.loc[0, col] = value
                    return df
                self.compare(change)
