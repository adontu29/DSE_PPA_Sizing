"""Generate an 8-panel presentation-quality mission profile figure.

Run from the repository root:
    python -m figures.plot_mission_profile
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from bellona_sizing.models import Mission, Assumptions
from bellona_sizing.workflows.phase1_to9 import iterate_phases_1_to_9

# ── Sizing run ────────────────────────────────────────────────────────────────
MTOW_KG    = 50.0
RANGE_M    = 6000.0
ALTITUDE_M = 6000.0

mission     = Mission(altitude_m=ALTITUDE_M, range_m=RANGE_M)
assumptions = Assumptions()

result  = iterate_phases_1_to_9(MTOW_kg=MTOW_KG, mission=mission,
                                  assumptions=assumptions, max_inner_iter=12)
p3      = result["phase3"]
cf      = p3["carry_forward"]
profile = p3["diagnostics"]["energy_critical_profile"]
states  = profile["states"]

# ── Derived arrays ────────────────────────────────────────────────────────────
alt   = np.array([s["altitude_m"]          for s in states])
TAS   = np.array([s["speed_m_s"]           for s in states])
EAS   = np.array([s["EAS_m_s"]             for s in states])
ROC   = np.array([s["rate_of_climb_m_s"]   for s in states])
gamma = np.array([s["climb_angle_deg"]      for s in states])
T_req = np.array([s["required_thrust_N"]   for s in states])
P_pro = np.array([s["propulsive_power_W"]  for s in states]) / 1e3
P_sha = np.array([s["shaft_power_W"]       for s in states]) / 1e3
P_ele = np.array([s["electrical_power_W"]  for s in states]) / 1e3
CL    = np.array([s["CL"]                  for s in states])
CL_lim= CL + np.array([s["CL_margin"]      for s in states])
P_mar = np.array([s["power_margin_W"]      for s in states]) / 1e3
E_cum = np.array([s["cumulative_mission_energy_Wh"] for s in states]) / 1e3

tr_data  = profile["takeoff_transition"]
tr_inner = tr_data["transition"]
tr_dist  = tr_data.get("transition_distance_m", 0.0)
tr_alt   = tr_data.get("transition_altitude_m", 20.0)
cap_kW   = profile["power_assumptions"]["max_affordable_electrical_power_W"] / 1e3

# Horizontal distance along flight path (starting after transition)
x_dist = tr_dist + np.cumsum([s["delta_x_m"] for s in states])

# Summary numbers
P_hover_kW = profile["power_assumptions"]["preliminary_hover_power_W"] / 1e3
E_total_kWh = profile["total_electrical_energy_Wh"] / 1e3
t_out_s     = profile["outbound_time_s"]
V_stall     = result["phase8"]["V_stall"]
ROC_min     = cf["minimum_required_average_ROC_m_s"]

# ── Plot ─────────────────────────────────────────────────────────────────────
BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"
GREY   = "#7f7f7f"

fig, axes = plt.subplots(4, 2, figsize=(14, 18))
fig.suptitle(
    f"Bellona — Mission Profile  |  MTOW {MTOW_KG:.0f} kg  ·  "
    f"Range {RANGE_M/1000:.0f} km  ·  Altitude {ALTITUDE_M/1000:.0f} km",
    fontsize=14, fontweight="bold", y=0.995,
)
ax = axes.ravel()

# ── 1. Trajectory ─────────────────────────────────────────────────────────────
ax[0].fill_between(
    np.concatenate(([0.0, 0.0, tr_dist], x_dist)),
    np.concatenate(([0.0, tr_alt, tr_alt], alt)),
    color=BLUE, alpha=0.12,
)
ax[0].plot(
    np.concatenate(([0.0, 0.0, tr_dist], x_dist)),
    np.concatenate(([0.0, tr_alt, tr_alt], alt)),
    color=BLUE, lw=2,
)
# Phase labels
ax[0].annotate("Takeoff\n& Transition", xy=(tr_dist / 2, tr_alt / 2),
               ha="center", fontsize=8, color=GREY)
ax[0].annotate("Wingborne\nClimb", xy=(x_dist[len(x_dist)//2], alt[len(alt)//2] * 0.6),
               ha="center", fontsize=8, color=GREY)
if profile["cruise"]["distance_m"] > 10:
    ax[0].annotate("Cruise", xy=(x_dist[-1] - profile["cruise"]["distance_m"] / 2,
                                  ALTITUDE_M + 150),
                   ha="center", fontsize=8, color=GREY)
ax[0].set_xlabel("Horizontal distance [m]")
ax[0].set_ylabel("Altitude [m]")
ax[0].set_title("Mission Trajectory")

# ── 2. Airspeed ───────────────────────────────────────────────────────────────
ax[1].plot(alt, TAS, color=BLUE,   lw=2, label="TAS")
ax[1].plot(alt, EAS, color=ORANGE, lw=2, label="EAS", linestyle="--")
ax[1].axhline(V_stall, color=RED, lw=1.2, linestyle=":", label=f"Stall {V_stall:.1f} m/s")
ax[1].legend(fontsize=8)
ax[1].set_xlabel("Altitude [m]")
ax[1].set_ylabel("Airspeed [m/s]")
ax[1].set_title("Airspeed vs Altitude")

# ── 3. Climb performance ──────────────────────────────────────────────────────
ax2a = ax[2]
ax2b = ax2a.twinx()
ax2a.plot(alt, ROC,   color=BLUE,   lw=2, label="ROC [m/s]")
ax2a.axhline(ROC_min, color=BLUE, lw=1.2, linestyle=":",
             label=f"Required ROC {ROC_min:.1f} m/s")
ax2b.plot(alt, gamma, color=ORANGE, lw=2, linestyle="--", label="Climb angle [°]")
ax2a.set_xlabel("Altitude [m]")
ax2a.set_ylabel("Rate of climb [m/s]", color=BLUE)
ax2b.set_ylabel("Climb angle [°]",     color=ORANGE)
lines1, labels1 = ax2a.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2a.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax[2].set_title("Climb Rate & Angle vs Altitude")

# ── 4. Thrust ─────────────────────────────────────────────────────────────────
ax[3].plot(alt, T_req, color=GREEN, lw=2)
ax[3].set_xlabel("Altitude [m]")
ax[3].set_ylabel("Required thrust [N]")
ax[3].set_title("Required Thrust vs Altitude")

# ── 5. Power breakdown ────────────────────────────────────────────────────────
ax[4].plot(alt, P_pro, color=GREEN,  lw=2, label="Propulsive")
ax[4].plot(alt, P_sha, color=ORANGE, lw=2, label="Shaft",      linestyle="--")
ax[4].plot(alt, P_ele, color=BLUE,   lw=2, label="Electrical")
ax[4].axhline(cap_kW, color=RED,  lw=1.5, linestyle="--",
              label=f"Cap {cap_kW:.0f} kW")
ax[4].axhline(P_hover_kW, color=GREY, lw=1.2, linestyle=":",
              label=f"Hover {P_hover_kW:.0f} kW")
ax[4].legend(fontsize=8)
ax[4].set_xlabel("Altitude [m]")
ax[4].set_ylabel("Power [kW]")
ax[4].set_title("Power Breakdown vs Altitude")

# ── 6. Lift coefficient ───────────────────────────────────────────────────────
ax[5].plot(alt, CL,     color=BLUE,  lw=2, label="CL")
ax[5].plot(alt, CL_lim, color=RED,   lw=1.5, linestyle="--", label="Permitted CL")
ax[5].legend(fontsize=8)
ax[5].set_xlabel("Altitude [m]")
ax[5].set_ylabel("Lift coefficient [–]")
ax[5].set_title("Lift Coefficient vs Altitude")

# ── 7. Power margin ───────────────────────────────────────────────────────────
ax[6].plot(alt, P_mar, color=BLUE, lw=2)
ax[6].fill_between(alt, P_mar, 0.0,
                   where=(P_mar >= 0), alpha=0.15, color=GREEN, label="Margin available")
ax[6].fill_between(alt, P_mar, 0.0,
                   where=(P_mar < 0),  alpha=0.25, color=RED,   label="Over budget")
ax[6].axhline(0.0, color="k", lw=1.0, linestyle="--")
ax[6].legend(fontsize=8)
ax[6].set_xlabel("Altitude [m]")
ax[6].set_ylabel("Power margin [kW]")
ax[6].set_title("Electrical Power Margin vs Altitude")

# ── 8. Cumulative energy ──────────────────────────────────────────────────────
# Only wingborne states are in the profile; hover energy is a lump at
# destination and has no altitude progression — show it as an annotation.
E_hover_kWh = E_total_kWh - float(E_cum[-1])
ax[7].plot(alt, E_cum, color=BLUE, lw=2)
ax[7].fill_between(alt, E_cum, alpha=0.12, color=BLUE)
ax[7].axhline(E_total_kWh, color=GREY, lw=1.0, linestyle="--",
              label=f"Total incl. hover {E_total_kWh:.3f} kWh")
ax[7].annotate(
    f"+{E_hover_kWh:.3f} kWh hover\nat destination",
    xy=(alt[-1], float(E_cum[-1])),
    xytext=(alt[-1] * 0.72, E_total_kWh * 0.88),
    fontsize=7.5, color=GREY,
    arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8),
)
ax[7].legend(fontsize=8)
ax[7].set_xlabel("Altitude [m]")
ax[7].set_ylabel("Cumulative energy [kWh]")
ax[7].set_title("Cumulative Electrical Energy vs Altitude")

# ── Shared formatting ─────────────────────────────────────────────────────────
for axis in ax:
    axis.grid(True, alpha=0.3, linewidth=0.6)
    axis.tick_params(labelsize=8)

# Key numbers box in figure margin
textstr = (
    f"Energy-critical case  (design range {RANGE_M/1000:.0f} km)\n"
    f"Total electrical energy : {E_total_kWh:.3f} kWh\n"
    f"Peak electrical power   : {P_ele.max():.2f} kW  (wingborne)\n"
    f"Outbound time           : {t_out_s:.0f} s\n"
    f"Min required ROC        : {ROC_min:.2f} m/s\n"
    f"V_stall (sea level)     : {V_stall:.2f} m/s"
)
fig.text(0.01, 0.005, textstr, fontsize=8, family="monospace",
         verticalalignment="bottom",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

fig.tight_layout(rect=[0, 0.07, 1, 1])

out = Path("figures") / "bellona_mission_profile_50kg_6km.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close(fig)
