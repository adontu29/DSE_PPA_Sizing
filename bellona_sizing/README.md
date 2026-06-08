# Bellona sizing code map

The active sizing package is split into readable phase and workflow modules.

- `models.py`: mission inputs and first-cut assumptions.
- `common.py`: ISA atmosphere and shared equations.
- `mission_profile.py`: altitude-stepped mission simulation, optimization, and plotting.
- `phases/phase01_*.py` through `phases/phase16_*.py`: one sizing phase per file.
- `phases/phase04_power.py`: preliminary mission electrical-power validation.
- `workflows/phase1_to9.py`: fixed-MTOW mission/aerodynamic inner loop.
- `workflows/coupled.py`: fixed-MTOW layout, control, stability, and mass coupling.
- `workflows/wing_area.py`: wing-area search that minimizes the coupled mass estimate.
- `workflows/mtow.py`: outer MTOW convergence loop.
- `cli.py`: command-line inputs, JSON output, and console summary.

The mission optimizer uses aerodynamic power demand, declared efficiencies,
fixed preliminary hover and transition powers, and an affordable total-aircraft
electrical-power limit with a default 5% reserve margin. Wing area is optimized
by default; stall speed is a calculated result from the selected area and
XFOIL-derived finite-wing lift limits. The active low-speed sizing gate is a
20 m/s transition-complete speed limit, paired with a 30 degree validity cap
on fixed-wing climb angle and a 3 degree Stone-style cruise stall margin. The
canard scissor check uses the lecture canard equations, Phase 7/8/9 aerodynamic
estimates, the Phase 15 mass CG, and 90% of canard `CLmax` by default. The
package does not contain a propeller map or component-level motor, ESC,
battery, RPM, torque, current, or advance-ratio model.

Run a fixed-MTOW preliminary mission with explicit power assumptions:

```powershell
python -m bellona_sizing --fixed-mtow --mtow-kg 65.7 `
  --wing-area-seed-m2 5.0 --cd0 0.04 --aspect-ratio 7.0 `
  --oswald-efficiency 0.78 --cl-max-guess 1.30 `
  --eta-prop 0.75 --eta-motor 0.90 --eta-esc 0.95 `
  --max-affordable-electrical-power-w 20000 `
  --preliminary-hover-power-w 14000 `
  --preliminary-transition-power-w 8000 `
  --json results/simple_mission.json `
  --mission-profile-plot figures/simple_mission.png `
  --scissor-plot figures/canard_scissor.png
```

Use `--wing-area-m2` to bypass the wing-area optimizer for a fixed-geometry
study. In that mode the workflow still calculates stall EAS, mission-altitude
stall TAS, and transition-altitude stall TAS from the supplied area.

The transition-speed and reserve assumptions can be overridden with
`--max-transition-complete-speed-m-s`, `--max-stall-eas-m-s`,
`--minimum-power-margin-fraction`, and
`--max-fixed-wing-climb-angle-deg`.

See `MISSION_PROFILE.md` for the equations, assumptions, optimizer variables,
constraints, and result fields. `code_outline.py` is retained only as a
historical project snapshot; the modular package is the active implementation.
