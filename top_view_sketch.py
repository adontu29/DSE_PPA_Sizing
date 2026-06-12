from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon


SUMMARY_PATH = Path("outputs") / "aircraft_summary.json"
OUTPUT_PATH = Path("outputs") / "top_view_sketch.png"


def tapered_chords(area_m2, span_m, taper):
    """Return root and tip chord for a simple trapezoid with the given area."""
    root = 2.0 * area_m2 / (span_m * (1.0 + taper))
    return root, taper * root


def add_trapezoid_pair(ax, x_le, span, root_chord, tip_chord, root_half_width, **kwargs):
    """Draw left and right unswept tapered surfaces around the fuselage."""
    half_span = 0.5 * span
    root_y = min(root_half_width, 0.85 * half_span)

    right = [
        (x_le, root_y),
        (x_le, half_span),
        (x_le + tip_chord, half_span),
        (x_le + root_chord, root_y),
    ]
    left = [(x, -y) for x, y in right]
    ax.add_patch(Polygon(right, closed=True, **kwargs))
    ax.add_patch(Polygon(left, closed=True, **kwargs))


DIM_COLOR = "#333333"
EXT_COLOR = "#999999"


def extension_line(ax, p1, p2):
    """Thin dotted witness line linking a feature to its dimension line."""
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=EXT_COLOR, lw=0.6, ls=":", alpha=0.8, zorder=4)


def dimension_line(ax, p1, p2, label, *, vertical=False, text_offset=0.07, text_side=1):
    """Double-headed dimension arrow between p1 and p2 with a centred boxed label.

    text_side = +1 places the label above (horizontal) or right (vertical) of the
    arrow; -1 places it below / left, used to push labels into clear space.
    """
    ax.annotate(
        "",
        xy=p2,
        xytext=p1,
        arrowprops=dict(arrowstyle="<|-|>", color=DIM_COLOR, lw=1.1, shrinkA=0, shrinkB=0),
        zorder=6,
    )
    mid_x = 0.5 * (p1[0] + p2[0])
    mid_y = 0.5 * (p1[1] + p2[1])
    box = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85)
    if vertical:
        ax.text(mid_x + text_side * text_offset, mid_y, label,
                ha="left" if text_side > 0 else "right", va="center", rotation=90,
                color=DIM_COLOR, fontsize=8, bbox=box, zorder=7)
    else:
        ax.text(mid_x, mid_y + text_side * text_offset, label, ha="center",
                va="bottom" if text_side > 0 else "top",
                color=DIM_COLOR, fontsize=8, bbox=box, zorder=7)


def main():
    with open(SUMMARY_PATH, "r", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)

    fuselage = summary["fuselage"]
    wing = summary["wing"]
    canard = summary["canard"]
    cg = summary.get("cg", {})

    fuselage_length = fuselage["length_m"] 
    fuselage_width = fuselage["width_m"]
    fuselage_half_width = 0.5 * fuselage_width

    wing_le = wing["mac_le_x_m_from_nose"]
    wing_span = wing["span_m"]
    wing_root = wing["root_chord_m"]
    wing_tip = wing["tip_chord_m"]

    canard_le = canard["le_m_from_nose"]
    canard_span = canard["span_m"]
    # The summary stores canard area/span but not taper. Match AIRCRAFT
    # ["canard_taper"] so the drawn canard root chord (and thus the canard->wing
    # gap shown here) is consistent with the canard_wing_min_gap_m constraint.
    canard_root, canard_tip = tapered_chords(
        canard["area_m2"], canard_span, taper=0.50
    )

    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    add_trapezoid_pair(
        ax,
        wing_le,
        wing_span,
        wing_root,
        wing_tip,
        fuselage_half_width,
        facecolor="#8fb7d8",
        edgecolor="#24445f",
        linewidth=1.7,
        alpha=0.85,
    )
    add_trapezoid_pair(
        ax,
        canard_le,
        canard_span,
        canard_root,
        canard_tip,
        fuselage_half_width,
        facecolor="#f0c36a",
        edgecolor="#7a5412",
        linewidth=1.5,
        alpha=0.9,
    )

    fuselage_patch = Ellipse(
        (0.5 * fuselage_length, 0.0),
        width=fuselage_length,
        height=fuselage_width,
        facecolor="#d8d8d8",
        edgecolor="#333333",
        linewidth=1.8,
        alpha=0.95,
    )
    ax.add_patch(fuselage_patch)
    ax.plot([0.0, fuselage_length], [0.0, 0.0], color="#555555", linewidth=0.8, alpha=0.5)

    if "x_cg_m_from_mac_le" in cg:
        x_cg = wing_le + cg["x_cg_m_from_mac_le"]
        ax.scatter([x_cg], [0.0], color="#c0392b", s=35, zorder=5)
        ax.text(x_cg, -0.18, "CG", ha="center", va="top", color="#8f2a1f", fontsize=8)

    ax.text(wing_le + 0.45 * wing_root, 0.52 * wing_span, "wing", ha="center", va="bottom")
    ax.text(canard_le + 0.45 * canard_root, 0.52 * canard_span, "canard", ha="center", va="bottom")
    ax.text(0.5 * fuselage_length, 0.0, "fuselage", ha="center", va="center", fontsize=9)

    # --- Dimension annotations -------------------------------------------------
    half_wing = 0.5 * wing_span
    half_canard = 0.5 * canard_span
    wing_root_y = min(fuselage_half_width, 0.85 * half_wing)
    canard_root_y = min(fuselage_half_width, 0.85 * half_canard)
    # l_h is the canard->wing arm the scissor plot uses: the distance between the
    # two surfaces' aerodynamic centres, taken at each MAC quarter-chord.
    wing_ac_x = wing_le + 0.25 * wing["mean_chord_m"]
    canard_ac_x = canard_le + 0.25 * canard["chord_m"]
    lh = wing_ac_x - canard_ac_x

    x_max = max(wing_le + wing_root, canard_le + canard_root, fuselage_length)
    surface_half = 0.5 * max(wing_span, canard_span)

    # Spanwise dimensions, drawn in the side margins with witness lines from the tips.
    x_wing_dim = x_max + 0.30
    extension_line(ax, (wing_le + wing_tip, half_wing), (x_wing_dim, half_wing))
    extension_line(ax, (wing_le + wing_tip, -half_wing), (x_wing_dim, -half_wing))
    dimension_line(ax, (x_wing_dim, -half_wing), (x_wing_dim, half_wing),
                   f"b = {wing_span:.2f} m", vertical=True)

    x_canard_dim = canard_le - 0.30
    extension_line(ax, (canard_le, half_canard), (x_canard_dim, half_canard))
    extension_line(ax, (canard_le, -half_canard), (x_canard_dim, -half_canard))
    dimension_line(ax, (x_canard_dim, -half_canard), (x_canard_dim, half_canard),
                   f"b$_c$ = {canard_span:.2f} m", vertical=True, text_side=-1)

    # Root chords, drawn along each surface root (inboard edge).
    dimension_line(ax, (wing_le, wing_root_y), (wing_le + wing_root, wing_root_y),
                   f"c$_r$ = {wing_root:.2f} m")
    dimension_line(ax, (canard_le, canard_root_y), (canard_le + canard_root, canard_root_y),
                   f"c$_{{r,c}}$ = {canard_root:.2f} m")

    # Aerodynamic-centre markers (MAC quarter-chord of each surface).
    ax.scatter([wing_ac_x, canard_ac_x], [0.0, 0.0], marker="x", color="#1f2d3d",
               s=28, linewidths=1.4, zorder=8)

    # Longitudinal dimensions, stacked in the bottom margin with witness lines.
    # l_h spans the two aerodynamic centres (a.c. to a.c.).
    y_lh = -(surface_half + 0.30)
    extension_line(ax, (canard_ac_x, 0.0), (canard_ac_x, y_lh))
    extension_line(ax, (wing_ac_x, 0.0), (wing_ac_x, y_lh))
    dimension_line(ax, (canard_ac_x, y_lh), (wing_ac_x, y_lh), f"l$_h$ = {lh:.2f} m")

    y_fus = -(surface_half + 0.70)
    extension_line(ax, (0.0, 0.0), (0.0, y_fus))
    extension_line(ax, (fuselage_length, 0.0), (fuselage_length, y_fus))
    dimension_line(ax, (0.0, y_fus), (fuselage_length, y_fus),
                   f"l$_f$ = {fuselage_length:.2f} m")

    ax.set_xlim(x_canard_dim - 0.70, x_wing_dim + 0.55)
    ax.set_ylim(y_fus - 0.30, surface_half + 0.30)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x from nose [m]")
    ax.set_ylabel("spanwise y [m]")
    ax.set_title("Bellona Top-View Geometry Sketch")
    ax.grid(True, alpha=0.2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
