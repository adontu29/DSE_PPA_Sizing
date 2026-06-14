"""Component mass build-up and CG for a given wing station."""

from __future__ import annotations

from sizing.inputs import AIRCRAFT, MASS
from sizing.geometry import longitudinal_layout


def mass_and_cg(wing, canard, mission, propeller, wing_le_m):
    """Mass build-up and CG for a wing MAC LE station (measured aft of nose).

    Longitudinal layout (x aft of nose), from colleague mass estimates:
      * nose bay (sensors, avionics, net+netgun) ........ nose_bay_x_m
      * canard ........................................... pinned at nose_to_canard
      * fuselage (uniform body) .......................... centroid L_fus/2
      * reel (balloon subsystem, ~at CG) ................. mid-body
      * parachute ........................................ just ahead of wing LE
      * battery .......................................... wing station
    L_fus = wing_le_m + wing root chord + wing_te_to_tail grows with the
    canard->wing arm.
    """
    layout = longitudinal_layout(wing, canard, wing_le_m)
    L_fus = layout["L_fus_m"]
    c_w = wing["chord_m"]
    n_rotors = AIRCRAFT["n_rotors"]

    nose_x = MASS["nose_bay_x_m"]
    mid_x = 0.5 * L_fus
    wing_x = layout["wing_quarter_m"]
    parachute_x = wing_le_m - MASS["parachute_ahead_of_wing_le_m"]
    # Battery is a large CG-trim mass; its station is selectable. "wing" (aft) is
    # tail-heavy for a canard; a forward offset moves the CG forward to help the
    # scissor close.
    battery_x = wing_le_m + AIRCRAFT.get("battery_cg_offset_over_mac", 0.0) * c_w

    # name -> (mass_kg, station_m aft of nose)
    components = {
        "wing":       (MASS["wing_areal_density_kg_m2"] * wing["area_m2"],      wing_x),
        "canard":     (MASS["canard_areal_density_kg_m2"] * canard["area_m2"],  layout["canard_quarter_m"]),
        "fuselage":   (MASS["fuselage_linear_density_kg_m"] * L_fus,            mid_x),
        "motors":     (n_rotors * MASS["motor_mass_each_kg"],                   wing_x),
        "propellers": (n_rotors * MASS["prop_mass_coeff_kg_m2"]
                       * propeller["propeller_diameter_m"] ** 2,                wing_x),
        "battery":    (mission["battery_mass_kg"],                             battery_x),
        "avionics":   (MASS["avionics_mass_kg"],                               nose_x),
        "sensors":    (MASS["sensor_mass_kg"],                                 nose_x),
        "net_gun":    (MASS["net_gun_mass_kg"],                                nose_x),
        "reel":       (MASS["reel_mass_kg"],                                   mid_x),
        "parachute":  (MASS["parachute_mass_kg"],                              parachute_x),
    }
    masses = {name: m for name, (m, _) in components.items()}
    locations = {name: x for name, (_, x) in components.items()}

    # Wiring scales with the powered systems; contingency on the full subtotal.
    masses["wiring"] = MASS["wiring_fraction"] * (
        masses["motors"] + masses["avionics"]
    )
    locations["wiring"] = mid_x
    subtotal = sum(masses.values())
    masses["contingency"] = MASS["contingency_fraction"] * subtotal
    locations["contingency"] = mid_x

    total_mass = sum(masses.values())
    x_cg_nose_m = sum(masses[name] * locations[name] for name in masses) / total_mass
    x_cg_m = x_cg_nose_m - wing_le_m                 # relative to wing MAC LE

    return {
        "total_mass_kg":        total_mass,
        "masses_kg":            masses,
        "locations_fuselage_m": locations,
        "wing_mac_le_x_m":      wing_le_m,
        "arm_m":                layout["arm_m"],
        "fuselage_length_m":    L_fus,
        "x_cg_fuselage_m":      x_cg_nose_m,
        "x_cg_m":               x_cg_m,
        "x_cg_over_mac":        x_cg_m / wing["chord_m"],
    }
