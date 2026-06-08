# Bellona Simplified Sizing Code

This branch contains the report-readable Bellona sizing script. It is meant to
be easy to inspect, present, and explain to aerospace engineers who want the
equations and assumptions in one place.

The full software-style package is archived in `archive/full_package/`.

## Run

From the repository root:

```powershell
python simple_sizing.py
```

The script writes:

- `outputs/summary.csv`
- `outputs/mass_breakdown.csv`
- `outputs/scissor_plot.png`
- `outputs/mission_profile.png`

## Change Inputs

Open `simple_sizing.py` and edit the input block at the top. The main values are:

- mission altitude, range, and hover time
- MTOW estimate
- wing area, aspect ratio, and aerodynamic coefficients
- canard arm and canard area-ratio sweep
- mass coefficients
- output folder

## Code Layout

The script is organized like a sizing calculation:

1. Inputs
2. Atmosphere and aircraft equations
3. Mass and CG equations
4. Canard scissor equations
5. Canard/wing-position iteration
6. Output tables and plots

Comments are kept short and equation-focused. The bottom of the file contains
the workflow that runs the complete sizing calculation.

## Tests

```powershell
python -m pytest -q
```

The tests only check that the simplified calculation runs, selects a feasible
canard area, and creates the expected report outputs.
