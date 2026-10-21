# CIM ADC accuracy sweep

The existing simulation code, data and documentation are in **[V4/](V4/README.md)**.
The repository root is reserved for future development; no V5 implementation has
been added.

## Run V4

Requirements: Python >= 3.10, NumPy, SciPy, pandas, matplotlib for plotting, and
GCC with OpenMP for IR-drop simulations.

```bash
cd V4
python main.py
python -m cim_sim.selfcheck
```

From the repository root, `python V4/main.py` also works. See the
[V4 guide](V4/README.md) for sweeps, plotting, data paths and validation.

## Layout and history

- `V4/`: existing code, ADC energy table, documentation, legacy single-file
  implementation and original result files, preserving their relative structure.
- `.github/workflows/v4.yml`: V4 checks; GitHub workflow configuration remains at
  the repository root.

The migration is a separate, content-preserving Git commit followed by path and
documentation fixes. Existing commits remain in history. To trace a moved file:

```bash
git log --follow -- V4/main.py
```
