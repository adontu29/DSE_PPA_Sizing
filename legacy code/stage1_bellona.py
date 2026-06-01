"""
Bellona PPA — Stage 1: Constraint Diagram
==========================================
Constraint diagram (W/S vs P/W) for the Bellona canard tail-sitter.

IMPORTANT NOTE ON PROPELLER SIZING
-----------------------------------
The Tyan 2017 disc loading regression (DL = 3.23*MTOW + 75) was derived
from 11 sea-level multicopters between 2 and 18 kg. For Bellona it is
used only as a reference — not as the primary sizing method — because:
  1. 50 kg is well outside the regression range (max was 18 kg).
  2. The regression captures sea-level hover choices. At 6 000 m
     (rho = 0.660 vs 1.225 kg/m3) the same disc loading needs ~36%
     MORE power than at sea level.
  3. Tyan sizes dedicated VTOL rotors. On a tail-sitter the same
     props also cruise, which couples disc sizing to forward-flight
     propeller efficiency — something Tyan's model ignores.

The recommended approach here is to set D_PROP_METHOD = 'physical' and
specify D_prop_m directly from the wingspan / clearance constraint.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# INPUTS
# =============================================================================

# --- Mission requirements ---
MTOW_kg      = 50.0    # initial mass estimate [kg]
n_rotors     = 4       # number of rotors (locked)
V_cruise     = 20.0    # cruise speed [m/s]
ROC          = 10.0    # required rate of climb [m/s]
V_stall      = 15.0    # max stall speed at ceiling [m/s]
T_W_design   = 1.30    # design hover T/W  (Stone 1.15 + 15% margin)
h_ceiling    = 6000.0  # design altitude [m]

# --- Aerodynamics (first guess) ---
CD0         = 0.04   # zero-lift drag coefficient
e_oswald    = 0.70   # Oswald efficiency — UAS range 0.65–0.72 (Nichols 2011)
CL_max      = 1.30   # max lift coefficient
AR          = 7.0    # aspect ratio
stall_margin = 0.80  # design W/S = 80% of stall limit

# --- Propulsion ---
FoM       = 0.70  # figure of merit — 0.70–0.80 for electric VTOL (Nichols 2011)
eta_motor = 0.90  # motor efficiency
eta_ESC   = 0.95  # ESC efficiency (brushless at high throttle)
eta_prop  = 0.75  # propeller efficiency in forward flight (cruise/climb)

# --- Propeller sizing method ---
# Choose ONE of three methods by setting D_PROP_METHOD:
#
#   'physical'   — recommended for tail-sitter.
#                  Set D_prop_m to the max diameter allowed by the airframe
#                  (wingspan, clearance, tip-to-tip spacing of the 4 rotors).
#                  Rule of thumb for a 4-rotor X-config on a 6.67 m wing:
#                  leave at least one diameter of clearance between tips, so
#                  D_prop_m < wingspan / (2 * rotors_per_side + 1).
#
#   'tyan_sl'    — Tyan 2017 regression at SEA LEVEL.
#                  Valid range: 2–18 kg at sea level.
#                  Extrapolated here and ignores altitude. Use as a reference only.
#
#   'tyan_alt'   — Tyan regression CORRECTED for altitude.
#                  Scales DL so that hover power fraction at altitude
#                  matches what the Tyan multicopters achieve at sea level:
#                  DL_corrected = DL_SL * (rho_h / rho_SL)
#
D_PROP_METHOD = 'physical'  # <-- change this to switch methods

# Used only when D_PROP_METHOD = 'physical':
# Rough estimate for Bellona: 4 rotors on a 6.67 m wing in X-config.
# With prop centres at roughly ±0.75 m from the fuselage, and a 0.10 m
# tip-to-tip clearance between adjacent props, D_prop_m ≈ 0.75 - 0.05 = 0.70 m.
# NOTE: this is a rough placeholder — it MUST be verified against the CAD
# once the fuselage and prop-mount locations are fixed.
D_prop_user_m = 1.10    # [m] — edit this value


# =============================================================================
# ISA ATMOSPHERE
# =============================================================================

def isa(h):
    """Returns (rho [kg/m3], mu [Pa.s], a [m/s]) at altitude h [m]."""
    g, R, gamma = 9.81, 287.05, 1.4
    T_SL, rho_SL, L = 288.15, 1.225, 0.0065
    if h <= 11000:
        T   = T_SL - L * h
        rho = rho_SL * (T / T_SL) ** (g / (L * R) - 1)
    else:
        T_trop   = T_SL - L * 11000
        rho_trop = rho_SL * (T_trop / T_SL) ** (g / (L * R) - 1)
        T        = 216.65
        rho      = rho_trop * np.exp(-g * (h - 11000) / (R * T))
    a  = np.sqrt(gamma * R * T)
    mu = 1.458e-6 * T**1.5 / (T + 110.4)
    return rho, mu, a


g = 9.81
rho_SL, mu_SL, a_SL = isa(0)
rho_h,  mu_h,  a_h  = isa(h_ceiling)

W = MTOW_kg * g   # weight [N]


# =============================================================================
# PROPELLER / DISC SIZING
# =============================================================================

# Tyan regression at sea level (reference only)
DL_tyan_SL  = 3.2261 * MTOW_kg + 74.991          # sea-level DL [N/m²]

# Altitude-corrected Tyan regression
# Derivation: P_hover ∝ sqrt(DL/rho). To match sea-level hover efficiency:
#   sqrt(DL_h / rho_h) = sqrt(DL_SL / rho_SL)  →  DL_h = DL_SL * (rho_h/rho_SL)
DL_tyan_alt = DL_tyan_SL * (rho_h / rho_SL)

T_rotor = T_W_design * W / n_rotors   # hover thrust per rotor [N]

if D_PROP_METHOD == 'physical':
    D_prop = D_prop_user_m
    A_disc = np.pi * (D_prop / 2)**2
    DL     = T_rotor / A_disc
    method_label = f"Physical constraint ({D_prop:.2f} m specified)"

elif D_PROP_METHOD == 'tyan_sl':
    DL     = DL_tyan_SL
    A_disc = T_rotor / DL
    D_prop = np.sqrt(4 * A_disc / np.pi)
    method_label = "Tyan 2017 sea-level regression (reference only)"

elif D_PROP_METHOD == 'tyan_alt':
    DL     = DL_tyan_alt
    A_disc = T_rotor / DL
    D_prop = np.sqrt(4 * A_disc / np.pi)
    method_label = "Tyan 2017 altitude-corrected regression"

else:
    raise ValueError(f"Unknown D_PROP_METHOD: {D_PROP_METHOD}")

# Tip Mach limit — Lan & Roskam / Nichols: M_tip < 0.72
# NOTE: use speed of sound at altitude, not sea level
M_tip_limit = 0.72
RPM_max     = (M_tip_limit * a_h) / (D_prop / 2) * 60 / (2 * np.pi)

# Advance ratio at cruise (sanity check for dual-mode prop feasibility)
# J = V / (n * D). A value of 0.3–0.6 is good for a cruising propeller.
# J < 0.2 means the prop is very inefficient in forward flight.
n_cruise = RPM_max / 60   # revolutions per second (using max RPM as proxy)
J_cruise  = V_cruise / (n_cruise * D_prop)


# =============================================================================
# CONSTRAINT FUNCTIONS
# =============================================================================

WS = np.linspace(10, 155, 600)   # wing loading sweep [N/m²]


def pw_cruise(WS, rho, V, CD0, AR, e, eta_motor, eta_ESC, eta_prop):
    eta_tot = eta_prop * eta_motor * eta_ESC
    q  = 0.5 * rho * V**2
    CL = WS / q
    CD = CD0 + CL**2 / (np.pi * AR * e)
    return V * (CD / CL) / eta_tot   # [W/N]


def pw_climb(WS, rho, V, CD0, AR, e, ROC, eta_motor, eta_ESC, eta_prop):
    eta_tot  = eta_prop * eta_motor * eta_ESC
    pw_level = pw_cruise(WS, rho, V, CD0, AR, e, eta_motor, eta_ESC, eta_prop)
    return pw_level + ROC / eta_tot   # [W/N]


def pw_hover_value(W, n_rotors, T_W, rho, A_disc, FoM, eta_motor, eta_ESC):
    T_r     = T_W * W / n_rotors
    P_ideal = T_r * np.sqrt(T_r / (2 * rho * A_disc))
    P_shaft = P_ideal / FoM
    P_elec  = P_shaft / (eta_motor * eta_ESC)
    return (P_elec * n_rotors) / W   # [W/N]


# =============================================================================
# STALL CONSTRAINT
# =============================================================================

WS_stall_max = 0.5 * rho_h * V_stall**2 * CL_max   # [N/m²]
WS_design    = stall_margin * WS_stall_max
S_wing       = W / WS_design


# =============================================================================
# EVALUATE AT DESIGN POINT
# =============================================================================

pw_c_val  = pw_cruise(WS_design, rho_h, V_cruise, CD0, AR, e_oswald,
                      eta_motor, eta_ESC, eta_prop)

pw_cl_val = pw_climb(WS_design, rho_h, V_cruise, CD0, AR, e_oswald, ROC,
                     eta_motor, eta_ESC, eta_prop)

pw_h_val  = pw_hover_value(W, n_rotors, T_W_design, rho_h, A_disc,
                            FoM, eta_motor, eta_ESC)

pw_design = max(pw_c_val, pw_cl_val, pw_h_val)

P_cruise_kW = pw_c_val  * W / 1000
P_climb_kW  = pw_cl_val * W / 1000
P_hover_kW  = pw_h_val  * W / 1000
P_shaft_kW  = P_hover_kW * (eta_motor * eta_ESC)

# Wing geometry preview
b     = np.sqrt(S_wing * AR)
c_bar = S_wing / b
Re    = rho_h * V_cruise * c_bar / mu_h


# =============================================================================
# SENSITIVITY: hover power vs propeller diameter (useful to visualise)
# =============================================================================

D_range     = np.linspace(0.5, 1.8, 200)
A_range     = np.pi * (D_range / 2)**2
T_r_fixed   = T_W_design * W / n_rotors
P_hover_kW_vs_D = []
for A in A_range:
    pw = pw_hover_value(W, n_rotors, T_W_design, rho_h, A, FoM, eta_motor, eta_ESC)
    P_hover_kW_vs_D.append(pw * W / 1000)

# Reference lines for Tyan methods
D_tyan_SL  = np.sqrt(4 * (T_rotor / DL_tyan_SL)  / np.pi)
D_tyan_alt = np.sqrt(4 * (T_rotor / DL_tyan_alt) / np.pi)


# =============================================================================
# PLOT — two panels: constraint diagram + hover power sensitivity
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel 1: constraint diagram ---
curve_cruise = pw_cruise(WS, rho_h, V_cruise, CD0, AR, e_oswald,
                         eta_motor, eta_ESC, eta_prop)
curve_climb  = pw_climb(WS, rho_h, V_cruise, CD0, AR, e_oswald, ROC,
                        eta_motor, eta_ESC, eta_prop)
line_hover   = np.full_like(WS, pw_h_val)

ax1.plot(WS, curve_cruise, color='#378ADD', lw=2.2, label='Cruise (level)')
ax1.plot(WS, curve_climb,  color='#1D9E75', lw=2.2, label=f'Climb  ROC={ROC:.0f} m/s')
ax1.plot(WS, line_hover,   color='#D85A30', lw=2.2, label=f'Hover  T/W={T_W_design}')
ax1.axvline(WS_stall_max, color='#888780', lw=1.5, ls='--',
            label=f'Stall limit  {WS_stall_max:.0f} N/m²')
ax1.axvline(WS_design, color='#888780', lw=1.0, ls=':',
            label=f'Design W/S  {WS_design:.0f} N/m²')
pw_envelope = np.maximum.reduce([curve_cruise, curve_climb, line_hover])
ax1.fill_between(WS, 0, pw_envelope, where=(WS <= WS_design),
                 color='#1D9E75', alpha=0.07, label='Feasible region')
ax1.scatter([WS_design], [pw_design], s=120, color='#D85A30', zorder=6)
ax1.annotate(
    f'Design point\nW/S={WS_design:.0f} N/m²\n{P_hover_kW:.1f} kW',
    xy=(WS_design, pw_design),
    xytext=(WS_design + 8, pw_design * 1.10),
    fontsize=8.5, color='#3d3d3a',
    arrowprops=dict(arrowstyle='->', color='#888780', lw=1.0),
)
ax1.set_xlabel('Wing loading  W/S  [N/m²]', fontsize=10)
ax1.set_ylabel('Power-to-weight  P/W  [W/N]', fontsize=10)
ax1.set_title(f'Constraint diagram\nISA {h_ceiling/1000:.0f} km  |  MTOW={MTOW_kg:.0f} kg', fontsize=10)
ax1.set_xlim(10, 155)
ax1.set_ylim(0, pw_h_val * 2.2)
ax1.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.25, lw=0.6)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Panel 2: hover power vs prop diameter ---
ax2.plot(D_range * 100, P_hover_kW_vs_D, color='#D85A30', lw=2.2,
         label=f'Hover power @ ISA {h_ceiling/1000:.0f} km')
ax2.axvline(D_prop * 100, color='#534AB7', lw=2.0, ls='-',
            label=f'Current design  ({D_prop:.2f} m, {D_prop*39.37:.0f} in)\n→ {P_hover_kW:.1f} kW')
ax2.axvline(D_tyan_SL  * 100, color='#888780', lw=1.3, ls='--',
            label=f'Tyan sea-level  ({D_tyan_SL:.2f} m, {D_tyan_SL*39.37:.0f} in)')
ax2.axvline(D_tyan_alt * 100, color='#888780', lw=1.3, ls=':',
            label=f'Tyan alt-corrected  ({D_tyan_alt:.2f} m, {D_tyan_alt*39.37:.0f} in)')
ax2.scatter([D_prop * 100], [P_hover_kW], s=100, color='#534AB7', zorder=5)
ax2.set_xlabel('Propeller diameter  [cm]', fontsize=10)
ax2.set_ylabel('Total hover power  [kW]', fontsize=10)
ax2.set_title(f'Hover power sensitivity to prop size\n(T/W={T_W_design}, n={n_rotors} rotors)', fontsize=10)
ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)
ax2.grid(True, alpha=0.25, lw=0.6)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('stage1_constraint_diagram.png', dpi=150, bbox_inches='tight')
print("Plot saved → stage1_constraint_diagram.png")
plt.show()


# =============================================================================
# RESULTS PRINTOUT
# =============================================================================

names  = ['cruise', 'climb', 'hover']
values = [pw_c_val, pw_cl_val, pw_h_val]
active = names[values.index(max(values))].upper()

print()
print("=" * 62)
print("  STAGE 1 RESULTS — Bellona canard tail-sitter")
print("=" * 62)

print("\n  ATMOSPHERE")
print(f"    Altitude            : {h_ceiling:.0f} m")
print(f"    Density  rho_h      : {rho_h:.4f} kg/m³   (rho_SL = {rho_SL:.4f})")
print(f"    Density ratio       : {rho_h/rho_SL:.3f}   (power penalty factor {np.sqrt(rho_SL/rho_h):.3f}×)")
print(f"    Speed of sound      : {a_h:.1f} m/s")
print(f"    Viscosity           : {mu_h:.3e} Pa·s")

print(f"\n  PROPELLER SIZING  [{method_label}]")
print(f"    Propeller diameter  : {D_prop:.3f} m  ({D_prop*39.37:.1f} in)")
print(f"    Disc area / rotor   : {A_disc:.3f} m²")
print(f"    Disc loading DL     : {DL:.1f} N/m²")
print(f"    Hover thrust/rotor  : {T_rotor:.1f} N  (T/W = {T_W_design})")
print(f"    Max RPM (M_tip<{M_tip_limit}): {RPM_max:.0f} rpm")
print(f"    Cruise advance ratio: J = {J_cruise:.3f}  (target 0.3–0.6 for dual-mode prop)")
if J_cruise < 0.25:
    print(f"    *** WARNING: J = {J_cruise:.3f} is low — prop may be poorly matched for cruise ***")
print(f"\n    Reference disc loadings (for comparison):")
print(f"      Tyan sea-level    : DL = {DL_tyan_SL:.1f} N/m²  →  D = {D_tyan_SL:.2f} m  ({D_tyan_SL*39.37:.0f} in)")
print(f"      Tyan alt-corrected: DL = {DL_tyan_alt:.1f} N/m²  →  D = {D_tyan_alt:.2f} m  ({D_tyan_alt*39.37:.0f} in)")
print(f"      (Alt-corrected = sea-level scaled by rho_h/rho_SL = {rho_h/rho_SL:.3f})")

print("\n  STALL CONSTRAINT")
print(f"    Stall W/S @ 6 km    : {WS_stall_max:.1f} N/m²")
print(f"    Design W/S          : {WS_design:.1f} N/m²  ({int(stall_margin*100)}% of limit)")

print("\n  WING GEOMETRY  (feeds Stage 2 & 3)")
print(f"    Wing area S         : {S_wing:.2f} m²")
print(f"    Wingspan b          : {b:.2f} m  (AR = {AR})")
print(f"    Mean chord c_bar    : {c_bar:.3f} m")
print(f"    Reynolds number     : {Re:.2e}  (@ {h_ceiling/1000:.0f} km cruise)")

print("\n  POWER CONSTRAINTS  (electrical, total aircraft)")
print(f"    P/W cruise          : {pw_c_val:.5f} W/N  →  {P_cruise_kW:.2f} kW")
print(f"    P/W climb           : {pw_cl_val:.5f} W/N  →  {P_climb_kW:.2f} kW")
print(f"    P/W hover           : {pw_h_val:.5f} W/N  →  {P_hover_kW:.2f} kW  ← {active}")
print(f"\n  Active constraint     : {active}")
print(f"  Design P/W            : {pw_design:.5f} W/N")

print("\n  REQUIRED POWER")
print(f"    P_hover (elec.)     : {P_hover_kW:.2f} kW  (total)")
print(f"    P_climb (elec.)     : {P_climb_kW:.2f} kW")
print(f"    P_cruise (elec.)    : {P_cruise_kW:.2f} kW")
print(f"    P_shaft (total)     : {P_shaft_kW:.2f} kW")
print(f"    P_shaft per motor   : {P_shaft_kW/n_rotors:.2f} kW")

print("\n  OUTPUTS TO OTHER DEPARTMENTS")
print(f"    → SCE        P_shaft = {P_shaft_kW:.2f} kW total  /  {P_shaft_kW/n_rotors:.2f} kW per motor")
print(f"    → Structures S = {S_wing:.2f} m²  (wing reference area)")
print("=" * 62)