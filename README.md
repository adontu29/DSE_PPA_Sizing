# Bellona Simplified Sizing Code

This branch contains the report-readable Bellona sizing code. It is meant to be
easy to inspect, present, and explain to aerospace engineers who want the
equations and assumptions in one place. The sizing method is split into one short
module per physical step, with a single linear workflow and no optional branches.

## Run

From the repository root:

```powershell
python simple_sizing.py
```

It writes a small, report-oriented output set:

- `outputs/summary.csv` — concise design summary (the values quoted in a report)
- `outputs/mass_breakdown.csv` — component masses and stations
- `outputs/wing_area_sweep.csv` — the wing-area trade table
- `outputs/scissor_plot.png` — canard scissor plot
- `outputs/mission_profile.png` — mission profile and energy sizing
- `outputs/mission_trajectory.png` — 3D / side-view trajectory to interception
- `outputs/wing_area_sweep.png` — wing-area mass/stall/energy trade

## Change Inputs

All knobs live in `sizing/inputs.py`, in three dictionaries:

- `MISSION` — altitude, range, hover time, take-off/spiral assumptions
- `AIRCRAFT` — MTOW estimate, wing/canard planform and aerodynamics, the
  wing-area and canard-area sweeps, drag build-up, transition/stall, propulsion,
  battery, and XFOIL settings
- `MASS` — areal/linear densities, component masses, and layout stations

## Code Layout

`simple_sizing.py` is just the entry point. The method lives in the `sizing/`
package, one module per step (read top to bottom):

1. `inputs.py` — mission, aircraft, and mass assumptions (the only knobs)
2. `atmosphere.py` — ISA density
3. `geometry.py` — wing / canard planforms and the rotor disc
4. `transition.py` — reduced-order tail-sitter transition sim -> max stall speed
5. `mass.py` — component mass build-up and CG
6. `scissor.py` — canard sizing and the static-stability / control CG band
7. `mission.py` — course-method climb energy and battery sizing
8. `airfoil.py` — optional XFOIL Reynolds-feedback refinement of section data
9. `loop.py` — the mass / wing-area sizing loops
10. `report.py` — concise tables and the report figures
11. `workflow.py` — `run_sizing()` and `main()`, the top-level spine

The supporting helper modules (shared, lower-level equations) are:

- `mission_energy_course.py` — course-method mission profile model: an
  altitude-stepped constant-EAS climb (`RC_s = (P_a - P_r)/W`), transition-speed
  limits, a coarse climb grid search, segment powers, and battery sizing
- `drag_buildup.py` — AD2 component parasite-drag build-up
- `scissor_plot.py` — canard sizing and static-stability relations
- `xfoil_wrapper.py` — XFOIL airfoil analysis
- `xfoil/` — the local XFOIL executable/data files

Comments are kept short and equation-focused.

## Tests

```powershell
python -m pytest -q
```

The tests only check that the simplified calculation runs, selects a feasible
canard area, and creates the expected report outputs.
