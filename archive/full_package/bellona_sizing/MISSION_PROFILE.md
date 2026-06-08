# Bellona preliminary mission-profile optimization

## Purpose

The mission model supports early aircraft sizing before a propeller, motor,
ESC, or battery has been selected. It estimates the power and energy required
by an aerodynamic flight profile. Component-level propulsion validation is
deferred until suitable data exists.

## Inputs

`AircraftPerformance` contains:

- aircraft mass, wing area, `CD0`, aspect ratio, and Oswald efficiency;
- maximum lift coefficient and the permitted fraction of that value.

`MissionPowerAssumptions` contains:

- propulsive, motor, and ESC efficiencies;
- maximum affordable total-aircraft electrical power;
- preliminary total-aircraft hover and transition electrical powers.

`MissionProfileConfig` contains mission altitude, horizontal range, outbound
time budget, hover duration, altitude step, low-altitude takeoff and
transition settings, and the active Stone-inspired speed and reserve limits.

## Wing-borne mechanics

Each climb step uses midpoint ISA atmospheric properties and a parabolic drag
polar:

```text
L      = W cos(gamma)
CL     = 2 W cos(gamma) / (rho V^2 S)
CD     = CD0 + CL^2 / (pi AR e)
D_drag = 0.5 rho V^2 S CD
```

Required thrust follows the discrete mechanical-energy balance:

```text
T_required =
    D_drag
  + [W Delta_h + 0.5 m (V_next^2 - V_current^2)] / Delta_s
```

Constant-speed climb reduces to `T = D + W sin(gamma)`. Constant-speed level
flight reduces to `T = D`. The initial climb schedule holds equivalent
airspeed constant:

```text
V_TAS = V_EAS sqrt(rho_0 / rho)
```

The TAS increase with altitude is included in the kinetic-energy term.

## Preliminary electrical power

Wing-borne power uses explicit assumed efficiencies:

```text
P_propulsive = T_required V
P_shaft      = P_propulsive / eta_prop
P_electrical = P_shaft / (eta_motor eta_ESC)
```

The model rejects a state when `P_electrical` exceeds the affordable power
limit or violates the selected reserve margin. The default reserve is 5% of
the affordable electrical-power cap. This limit expresses the design team's
power budget. It does not prove that a particular propulsion system can
produce the required thrust.

Vertical takeoff and mission hover use `preliminary_hover_power_W`. Transition
uses `preliminary_transition_power_W`. These are total-aircraft electrical
powers and should be updated from test data or a separate hover study.

## Mission and optimizer

The integrated mission contains:

1. Low-altitude vertical takeoff.
2. Speed-based transition at the takeoff altitude.
3. Altitude-stepped wing-borne climb.
4. Level cruise at mission altitude over the remaining horizontal distance.
5. Mission hover.

The inspectable coarse grid optimizes climb EAS, climb angle, and level-cruise
TAS. The default climb-angle grid covers 20 to 30 degrees in 2.5-degree steps.
The 30-degree cap is a model-validity limit for wing-borne climb; steeper
segments require explicit transition-assisted or propeller-borne climb
modeling.
The default speed grids are generated from the current candidate wing area and
finite-wing `CL_max`. The lower climb EAS is set by `CL_allowed`, while the
lower cruise TAS is the larger of the `CL_allowed` speed at mission altitude
and the transition-completion speed. If the optimum lies on an upper speed
boundary, the corresponding grid is expanded and re-evaluated.
`CL_allowed` is the stricter of the configured 0.90 `CLmax` limit and the
Stone-style 3 degree margin below estimated stall angle.
A candidate is feasible when:

- permitted `CL` is not exceeded;
- electrical power remains within the affordable limit and reserve margin;
- cruise speed clears the transition requirement;
- transition completion remains below the selected speed limit;
- fixed-wing climb angle remains below the selected validity cap;
- outbound time remains within the budget.

The top-level sizing workflow treats wing area as a design variable. It
evaluates fixed-area candidates, calculates stall speed from each area, and
selects the feasible candidate with the lowest coupled mass estimate. The
active low-speed sizing gate is the default 20 m/s transition-complete speed
limit, which implies a maximum allowed transition-altitude stall speed through
the current blend-end and cruise-margin fractions. The old stall-speed target
is no longer a sizing constraint; an optional direct stall-EAS cap remains
available for studies.

Mission range is treated as required point-to-point horizontal displacement.
If climb ground-track distance exceeds that displacement, the model assumes an
idealized spiral ending at the target position and sets level-cruise distance
to zero. The spiral currently has no bank-angle, increased-lift, or turning-drag
penalty, so shallow spiral-climb profiles are optimistic.

The optional three-band extension varies EAS and climb angle over three
altitude bands. The optional minimum-time reference selects the fastest
feasible constant-EAS climb under the same power limit.

## Results

The mission result reports optimized speeds and climb angle, integrated time,
range and energy, segment summaries, required thrust, drag, `CL`, propulsive,
shaft and electrical power, power margin, and the complete altitude-stepped
state history. Plots show profile geometry, TAS/EAS, climb rate and angle,
required thrust, power demand and limit, `CL`, power margin, and cumulative
energy.

## Limitations

- Atmosphere is ISA and wind is zero.
- Wing-borne steps are quasi-steady.
- Efficiencies are constant assumptions.
- Hover and transition powers are fixed assumptions.
- Propeller thrust capability, RPM, torque, current, thermal limits, and
  component masses are outside this mission model.
- Phase 1 retains a preliminary disk-loading diameter estimate for geometry
  and mass sizing only.
