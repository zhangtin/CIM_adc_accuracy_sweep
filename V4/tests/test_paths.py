"""Exercise V4 entry points from different working directories using real outputs."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

V4 = Path(__file__).resolve().parents[1]
ROOT = V4.parent
SWEEP_ARGS = [
    '--sizes', '8', '--mem-bits', '4', '--mem-sigma', '0.02',
    '--adc-bits', '4', '8', '--n-mc', '6', '--r-wire', '0',
    '--no-noise', '--no-self-check', '--no-plots',
]


class PathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cim-v4-')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def run_script(self, script, *args, cwd=ROOT):
        # No inherited PYTHONPATH: entry points must locate their own package.
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                   MPLBACKEND='Agg', OMP_NUM_THREADS='2')
        env.pop('PYTHONPATH', None)
        result = subprocess.run(
            [sys.executable, str(V4 / script), *map(str, args)], cwd=cwd,
            env=env, capture_output=True, text=True, encoding='utf-8', timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def sweep(self, script, cwd, output, *extra):
        self.run_script(script, *SWEEP_ARGS, '--out', output, *extra, cwd=cwd)
        target = Path(output) if Path(output).is_absolute() else cwd / output
        for name in ('raw_trials.csv.gz', 'summary.csv', 'knee.csv',
                     'ir_calibration.csv', 'config.json'):
            self.assertTrue((target / name).is_file(), name)
        raw = pd.read_csv(target / 'raw_trials.csv.gz')
        self.assertEqual(len(raw), 72)
        self.assertFalse(raw['E_adc_typical_J'].isna().any())
        return raw

    def test_cli_imports_and_walkthrough(self):
        outputs = []
        for cwd in (ROOT, V4, self.work):
            with self.subTest(cwd=cwd):
                for script in ('main.py', 'cim_adc_mc.py', 'cim_adc_underutil.py'):
                    self.assertIn('usage:', self.run_script(script, '--help', cwd=cwd))
                outputs.append(self.run_script('main.py', '--no-ir', '--no-noise', cwd=cwd))
        self.assertIn('Step 10', outputs[0])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_default_sweep_from_root_v4_and_unrelated_directory(self):
        results = []
        for index, cwd in enumerate((ROOT, V4, self.work)):
            with self.subTest(cwd=cwd):
                results.append(self.sweep('cim_adc_mc.py', cwd, self.work / str(index)))
        for raw in results[1:]:
            pd.testing.assert_frame_equal(results[0], raw, check_exact=True)

    def test_legacy_default_table_from_root_and_unrelated_directory(self):
        root = self.sweep('All-in-one script/cim_adc_mc.py', ROOT, self.work / 'root')
        foreign = self.sweep('All-in-one script/cim_adc_mc.py', self.work, 'legacy')
        pd.testing.assert_frame_equal(root, foreign, check_exact=True)

    def test_explicit_table_and_output_stay_relative_to_caller(self):
        table = pd.read_csv(V4 / 'ADC_22nm_energy_vs_bits.csv')
        table['E_typical_J'] = 0.125
        table.to_csv(self.work / 'custom.csv', index=False)
        for index, script in enumerate(('cim_adc_mc.py', 'All-in-one script/cim_adc_mc.py')):
            with self.subTest(script=script):
                raw = self.sweep(script, self.work, str(index), '--energy-csv', 'custom.csv')
                self.assertTrue((raw['E_adc_typical_J'] == 8 * 0.125).all())
        # An explicit missing path must fail, not silently select bundled data.
        env = dict(os.environ, PYTHONUTF8='1')
        result = subprocess.run(
            [sys.executable, str(V4 / 'cim_adc_mc.py'), *SWEEP_ARGS,
             '--energy-csv', 'missing.csv', '--out', 'missing-output'],
            cwd=self.work, env=env, capture_output=True, timeout=120,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.work / 'missing-output').exists())

    def test_resummarize_uses_requested_output_directory(self):
        self.sweep('cim_adc_mc.py', self.work, 'results')
        summary = self.work / 'results/summary.csv'
        expected = pd.read_csv(summary)
        summary.unlink()
        self.run_script('cim_adc_mc.py', '--resummarize', '--no-plots',
                        '--out', 'results', cwd=self.work)
        actual = pd.read_csv(summary)
        self.assertEqual(list(expected.columns), list(actual.columns))
        self.assertEqual(len(expected), len(actual))

    def test_underutil_and_plot_entrypoints(self):
        self.run_script('cim_adc_underutil.py', '--hw-sizes', '8', '--utils', '0.5', '1',
                        '--n-min', '4', '--mem-bits', '4', '--mem-sigma', '0.02',
                        '--adc-bits', '4', '8', '--n-mc', '6', '--out', 'underutil', cwd=self.work)
        summary = self.work / 'underutil/summary.csv'
        self.assertEqual(len(pd.read_csv(summary)), 72)
        self.run_script('plot_underutil.py', 'underutil', cwd=self.work)
        figures = list((self.work / 'underutil/fig').glob('*.png'))
        self.assertEqual(len(figures), 4)
        self.assertTrue(all(p.stat().st_size > 0 for p in figures))


if __name__ == '__main__':
    unittest.main()
