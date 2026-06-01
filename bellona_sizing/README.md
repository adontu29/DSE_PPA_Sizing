# Bellona sizing code map

`code_outline.py` is kept as the reference snapshot. The active readable version is split here.

- `models.py`: mission inputs and global first-cut assumptions.
- `common.py`: ISA atmosphere and shared helper equations.
- `phases/phase01_*.py` through `phases/phase16_*.py`: one phase per file.
- `workflows/phase1_to9.py`: fixed-MTOW propulsion/aerodynamics inner loop.
- `workflows/layout.py`: mass/CG envelope, scissor-plot coupling, canard grid search, and wing-position solve.
- `workflows/control.py`: elevon, hover-control, transition-energy, and dynamic-stability helper calls.
- `workflows/sanity.py`: design sanity checks and red flags.
- `workflows/coupled.py`: short fixed-MTOW orchestration that connects the layout and control helper workflows.
- `workflows/mtow.py`: outer Phase 16 MTOW convergence loop.
- `cli.py`: command-line parsing, JSON writing, and printed summary.
- `../bellona_main.py`: small script entrypoint.

Run the refactored workflow with:

```powershell
python bellona_main.py --json results_modular.json
```

The original workflow remains available with:

```powershell
python code_outline.py --json results.json
```
