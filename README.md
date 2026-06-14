# Bellona Simplified Sizing Code

This branch contains the report-readable Bellona sizing script. It is meant to
be easy to inspect, present, and explain to aerospace engineers who want the
equations and assumptions in one place.

This branch intentionally contains only the `simple_sizing.py` workflow and the
small helper modules it imports.

## Run

From the repository root:

```powershell
python simple_sizing.py
```

The script writes:

- `outputs/summary.csv`
- `outputs/mass_breakdown.csv`
- `outputs/iteration_history.csv`
- `outputs/wing_area_sweep.csv`
- `outputs/aircraft_summary.json`
- `outputs/scissor_plot.png`
- `outputs/mission_profile.png`

## Change Inputs

Open `simple_sizing.py` and edit the input block at the top. The main values are:

- mission altitude, range, and hover time
- mission climb, transition, and power limits
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

`mission_energy_course.py` contains the course-method mission profile model
used by the main script. It uses an altitude-stepped constant-EAS climb,
transition-speed limits, a coarse climb/cruise grid search, segment powers, and
battery sizing from the selected mission segments.

The other helper files are:

- `drag_buildup.py` for parasite-drag buildup
- `scissor_plot.py` for canard sizing and static-stability relations
- `xfoil_wrapper.py` for optional XFOIL airfoil analysis
- `xfoil/` for the local XFOIL executable/data files

Comments are kept short and equation-focused. The bottom of the file contains
the workflow that runs the complete sizing calculation.

## Tests

```powershell
python -m pytest -q
```

The tests only check that the simplified calculation runs, selects a feasible
canard area, and creates the expected report outputs.
