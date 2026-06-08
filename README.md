# Bellona Platform Sizing

Preliminary sizing code for the Bellona UAV platform. The package runs the
mission, wing, canard, scissor-plot, control, transition, mass, and MTOW
coupling phases from the command line.

## Setup

Use Python 3.10 or newer from the repository root.

```powershell
python -m pip install numpy matplotlib pytest
```

XFOIL is optional. If no executable is available, the code falls back to the
project airfoil table and reports the fallback in the warnings. Use
`--no-use-xfoil` for a deterministic fallback run.

## Quick Runs

Run a fixed-MTOW sizing pass:

```powershell
python -m bellona_sizing --fixed-mtow --mtow-kg 52.78 --wing-area-m2 6.8 --no-use-xfoil
```

Run the same pass and generate a Phase 10 canard scissor plot:

```powershell
python -m bellona_sizing --fixed-mtow --mtow-kg 52.78 --wing-area-m2 6.8 `
  --max-inner-iter 5 --control-closure-max-iter 2 `
  --scissor-plot figures\canard_scissor_phase10.png `
  --json results\scissor_plot_run.json
```

Run the coupled MTOW convergence loop:

```powershell
python -m bellona_sizing --mtow-kg 50 --no-use-xfoil --json results\mtow_run.json
```

Generate the mission-profile figure from a JSON result:

```powershell
python figures\plot_mission_profile.py results\scissor_plot_run.json figures\mission_profile.png
```

## Tests

```powershell
python -m pytest -q
```

## Outputs

Generated JSON and plot outputs are ignored by Git:

- `results/`
- `figures/*.png`
- `tmp_stone_pages/`

Keep source changes in `bellona_sizing/`, tests in `tests/`, and run
instructions in the README files.
