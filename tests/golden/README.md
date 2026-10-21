# Regression golden data

Generated with the ORIGINAL, pre-refactoring scripts (`scripts/cim_adc_mc.py` v3 and
`scripts/cim_adc_underutil.py` in the project) and with `main.py` of release 4.0.0. The refactored
package is checked against these files by `tests/test_regression.py`.

Serialized trials, under-utilization tables, IR calibration and walkthrough stdout
are compared exactly. Full-precision sweep `summary.csv` and `knee.csv` float
columns use `rtol=1e-10` and `atol=1e-12 * max(abs(reference_column))` to accommodate
roundoff across CPU/compiler, BLAS and pandas versions. Non-floating columns,
column order, shapes, dtypes and NaN positions still have to match. Scaling each
column separately preserves strict checks on tiny energy/current quantities.
Set `CIM_SIM_STRICT_REGRESSION=1` to require exact equality for all CSVs when
reproducing the original numerical environment. The supplied golden data has not
been regenerated or modified.

| Directory / file | Command (original script) |
|---|---|
| `sweep_ir_noise/` | `cim_adc_mc.py --sizes 16 32 --mem-bits 2 4 --mem-sigma 0 0.02 --n-mc 10 --adc-bits 4 8 --chip-cal --cal-exact-max 16` |
| `sweep_sampled/` | `cim_adc_mc.py --sizes 16 32 --mem-bits 3 --mem-sigma 0.05 --n-mc 10 --adc-bits 6 10 --r-wire 0 --cal-mode sampled --chunk-elems 3000` |
| `underutil/` | `cim_adc_underutil.py --hw-sizes 32 64 --utils 1 0.5 --mem-bits 4 --mem-sigma 0.02 --n-mc 20 --extra-pairs 32:20 --adc-bits 4 8 12` |
| `main_default.txt` | `python main.py` (stdout) |

`--cal-exact-max 16` makes the N = 32 point use the Rademacher-probe calibration path.
