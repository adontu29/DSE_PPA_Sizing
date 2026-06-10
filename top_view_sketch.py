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


def main():
    with open(SUMMARY_PATH, "r", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)

    fuselage = summary["fuselage"]
    wing = summary["wing"]
    canard = summary["canard"]
    cg = summary.get("cg", {})

    fuselage_length = fuselage["length_m"] + 1
    fuselage_width = fuselage["width_m"]
    fuselage_half_width = 0.5 * fuselage_width

    wing_le = wing["mac_le_x_m_from_nose"]
    wing_span = wing["span_m"]
    wing_root = wing["root_chord_m"]
    wing_tip = wing["tip_chord_m"]

    canard_le = canard["le_m_from_nose"]
    canard_span = canard["span_m"]
    # The summary stores canard area/span but not taper. This taper is only for
    # the sketch shape; area and span still come from the JSON.
    canard_root, canard_tip = tapered_chords(
        canard["area_m2"], canard_span, taper=0.55
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

    x_max = max(wing_le + wing_root, canard_le + canard_root, fuselage_length)
    y_limit = 0.5 * max(wing_span, canard_span) + 0.35
    ax.set_xlim(-0.15, x_max + 0.25)
    ax.set_ylim(-y_limit, y_limit)
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
