"""Wing / canard planforms and the rotor disc.

Pure geometry from the area ratios and aspect ratios in the inputs, plus the
nose-referenced longitudinal layout that ties the canard, wing, and tail
together (the fuselage length follows from where the wing sits).
"""

from __future__ import annotations

import math

from sizing.inputs import AIRCRAFT, MASS
from sizing.atmosphere import isa_density


def propeller_disk_estimate(weight_N):
    """Rotor thrust and disc diameter from the disc loading."""
    thrust_total = AIRCRAFT["thrust_to_weight"] * weight_N
    thrust_per_rotor = thrust_total / AIRCRAFT["n_rotors"]
    disk_area = thrust_per_rotor / AIRCRAFT["disc_loading_N_m2"]
    diameter = 2.0 * math.sqrt(disk_area / math.pi)
    return {
        "thrust_total_N": thrust_total,
        "thrust_per_rotor_N": thrust_per_rotor,
        "disk_area_m2": disk_area,
        "propeller_diameter_m": diameter,
    }


def wing_geometry(weight_N, rho_cruise, wing_area=None):
    """Wing planform and the sea-level stall EAS / cruise trim CL."""
    if wing_area is None:
        wing_area = AIRCRAFT["wing_area_m2"]
    aspect_ratio = AIRCRAFT["wing_aspect_ratio"]
    span = math.sqrt(wing_area * aspect_ratio)
    chord = wing_area / span
    root_chord = 2.0 * wing_area / (span * (1.0 + AIRCRAFT["wing_taper"]))
    tip_chord = AIRCRAFT["wing_taper"] * root_chord
    x_ac = 0.25 * chord

    rho_sea_level = isa_density(0.0)
    stall_EAS = math.sqrt(2.0 * weight_N / (rho_sea_level * wing_area * AIRCRAFT["wing_CL_max"]))
    cruise_speed = AIRCRAFT["cruise_true_speed_m_s"]
    q_cruise = 0.5 * rho_cruise * cruise_speed**2
    CL_trim = weight_N / (q_cruise * wing_area)

    return {
        "area_m2": wing_area,
        "span_m": span,
        "chord_m": chord,
        "root_chord_m": root_chord,
        "tip_chord_m": tip_chord,
        "x_ac_m": x_ac,
        "stall_EAS_m_s": stall_EAS,
        "cruise_true_speed_m_s": cruise_speed,
        "CL_trim": CL_trim,
        "CL_alpha_per_rad": AIRCRAFT["wing_CL_alpha_per_rad"],
    }


def canard_geometry(area_ratio, wing):
    """Canard planform from the selected area ratio Sc/Sw.

    Longitudinal position is not set here: the canard is pinned to the nose and
    the wing position (canard->wing arm) is solved in canard_and_wing_iteration,
    so the arm lives in the layout, not the planform.
    """
    area = area_ratio * wing["area_m2"]
    span = math.sqrt(area * AIRCRAFT["canard_aspect_ratio"])
    chord = area / span
    return {
        "area_ratio": area_ratio,
        "area_m2": area,
        "span_m": span,
        "chord_m": chord,
        "CL_alpha_per_rad": AIRCRAFT["canard_CL_alpha_per_rad"],
    }


def longitudinal_layout(wing, canard, wing_le_m):
    """Nose-referenced longitudinal layout for a given wing MAC LE station.

    x is measured aft from the nose. The canard MAC LE is pinned at
    nose_to_canard; the wing MAC LE is at wing_le_m; the fuselage tail is
    wing_te_to_tail behind the wing root trailing edge, so
    L_fus = wing_le_m + c_root_w + wing_te_to_tail and the arm (and fuselage
    length) grow as the wing moves aft.
    """
    c_w = wing["chord_m"]
    c_c = canard["chord_m"]
    nose_to_canard = MASS["nose_to_canard_m"]
    wing_root_te_m = wing_le_m + wing["root_chord_m"]
    L_fus = wing_root_te_m + MASS["wing_te_to_tail_m"]
    wing_quarter_m = wing_le_m + 0.25 * c_w               # wing MAC a.c. (1/4 chord)
    canard_quarter_m = nose_to_canard + 0.25 * c_c        # canard MAC a.c. (1/4 chord)
    return {
        "wing_le_m": wing_le_m,
        "canard_le_m": nose_to_canard,
        "arm_m": wing_le_m - nose_to_canard,                  # canard->wing, > 0
        "L_fus_m": L_fus,
        "wing_root_te_m": wing_root_te_m,
        "wing_quarter_m": wing_quarter_m,                     # mass/AC station
        "canard_quarter_m": canard_quarter_m,
        # Tail arm for the scissor equations: distance between the two surfaces'
        # aerodynamic centres (each MAC quarter-chord), not LE-to-LE, over c_w.
        "lh_over_mac": (canard_quarter_m - wing_quarter_m) / c_w,   # < 0 (canard ahead)
    }
