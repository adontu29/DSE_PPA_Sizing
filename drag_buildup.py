"""Component drag build-up for the zero-lift (parasite) drag coefficient.

This follows the Lecture 2 (AircraftDesign2 - Lift & Drag Estimation) workflow:

    CD0 = sum_components (Cf * FF * IF * Swet) / Sref + CD_misc

where, per component (AD2 slides 48-53):
  * Cf  - flat-plate skin-friction coefficient, the weighted average of the
          laminar (1.328/sqrt(Re)) and turbulent (Prandtl-Schlichting) values
          over the assumed laminar-flow fraction (AD2 slides 49-51).
  * FF  - form factor, the pressure drag from viscous separation (AD2 slide 52).
  * IF  - interference factor between components (AD2 slide 53).
  * Swet- component wetted area (AD2 slide 47).
CD_misc here is the excrescence/leakage increment (AD2 slide 60), taken as a
fraction of the clean component sum (5-10% for propeller aircraft).

Reynolds numbers are per-component (Re = unit_Re * characteristic length) and are
capped at the roughness cut-off Reynolds number (AD2 slide 51) so a smoother
finish does not promise unrealistically low friction at high length Reynolds.

The functions take plain numbers / small dicts so the caller (mission_energy_course
and the sizing summary) can evaluate the build-up at any mission state.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Flat-plate skin friction (AD2 slides 50-51)
# ---------------------------------------------------------------------------


def flat_plate_friction_laminar(reynolds):
    """Blasius laminar flat-plate skin-friction coefficient."""
    return 1.328 / math.sqrt(max(reynolds, 1.0))


def flat_plate_friction_turbulent(reynolds, mach):
    """Prandtl-Schlichting turbulent skin friction with compressibility."""
    reynolds = max(reynolds, 1.0)
    return 0.455 / (
        math.log10(reynolds) ** 2.58 * (1.0 + 0.144 * mach**2) ** 0.65
    )


def cutoff_reynolds(length_m, roughness_k_m):
    """Subsonic roughness cut-off Reynolds number (AD2 slide 51).

    Above this length Reynolds number the surface behaves as fully rough, so the
    effective friction Reynolds number is capped here.
    """
    if roughness_k_m <= 0.0 or length_m <= 0.0:
        return math.inf
    return 38.21 * (length_m / roughness_k_m) ** 1.053


def mixed_friction(reynolds, mach, laminar_fraction, length_m=0.0, roughness_k_m=0.0):
    """Weighted laminar/turbulent flat-plate Cf (AD2 slide 51).

    Cf = f_lam * Cf_laminar + (1 - f_lam) * Cf_turbulent, evaluated at the
    roughness-limited Reynolds number.
    """
    effective_re = reynolds
    if length_m > 0.0 and roughness_k_m > 0.0:
        effective_re = min(reynolds, cutoff_reynolds(length_m, roughness_k_m))
    cf_laminar = flat_plate_friction_laminar(effective_re)
    cf_turbulent = flat_plate_friction_turbulent(effective_re, mach)
    fraction = min(max(laminar_fraction, 0.0), 1.0)
    return fraction * cf_laminar + (1.0 - fraction) * cf_turbulent


# ---------------------------------------------------------------------------
# Form factors (AD2 slide 52) and wetted areas (AD2 slide 47)
# ---------------------------------------------------------------------------


def lifting_surface_form_factor(thickness_ratio, max_thickness_x_c, sweep_max_rad, mach):
    """Wing / tail / strut / pylon form factor (AD2 slide 52, Raymer).

    FF = (1 + 0.6/(x/c)_m * t/c + 100 (t/c)^4) * (1.34 M^0.18 (cos L_m)^0.28)

    The compressibility multiplier is floored at 1.0: it is calibrated for the
    transonic regime and drops below one at the very low Mach numbers of this
    UAV, which would unphysically reduce the form below the incompressible value.
    """
    incompressible = (
        1.0
        + 0.6 / max(max_thickness_x_c, 1e-6) * thickness_ratio
        + 100.0 * thickness_ratio**4
    )
    compressibility = 1.34 * mach**0.18 * math.cos(sweep_max_rad) ** 0.28
    return incompressible * max(1.0, compressibility)


def body_form_factor(fineness_ratio):
    """Fuselage / smooth-body form factor (AD2 slide 52, Raymer)."""
    f = max(fineness_ratio, 1e-6)
    return 1.0 + 60.0 / f**3 + f / 400.0


def lifting_surface_wetted_area(exposed_area, thickness_ratio):
    """Wetted area of an exposed lifting surface (AD2 slide 47, Raymer).

    Swet ~= S_exposed * (1.977 + 0.52 t/c) for the thicknesses used here.
    """
    return exposed_area * (1.977 + 0.52 * thickness_ratio)


def body_wetted_area(length_m, diameter_m):
    """Wetted area of a roughly axisymmetric body (AD2 slide 47, Raymer)."""
    f = length_m / max(diameter_m, 1e-6)
    if f <= 2.0:
        return math.pi * diameter_m * length_m
    return (
        math.pi * diameter_m * length_m
        * (1.0 - 2.0 / f) ** (2.0 / 3.0)
        * (1.0 + 1.0 / f**2)
    )


# ---------------------------------------------------------------------------
# Geometry assembly and the build-up itself
# ---------------------------------------------------------------------------


def build_drag_geometry(aircraft, wing, trim):
    """Pre-compute the per-component drag terms for the build-up, or None if missing.

    `trim` is the converged canard/arm descriptor (see trim_drag_descriptor in
    simple_sizing); it carries the canard planform and the fuselage length/width
    needed for the wetted areas. When it (or the fuselage geometry) is absent the
    build-up cannot run and the caller falls back to the fixed CD0.

    Everything that does not depend on the flow state (wetted area, cut-off
    Reynolds number, the constant part of the form factor, and the IF*Swet/Sref
    factor) is baked in here so the per-state evaluation in parasite_drag_buildup
    only has to recompute the skin friction and the Mach-dependent form-factor
    term. The result is a list of prepared components, evaluated in mission loops
    tens of thousands of times, so this hoisting is the main performance lever.
    """
    if trim is None:
        return None
    fuselage_length = trim.get("L_fus")
    fuselage_width = trim.get("fus_width")
    if not fuselage_length or not fuselage_width:
        return None

    s_ref = wing["area_m2"]
    roughness = aircraft.get("surface_roughness_k_m", 0.0)

    wing_root_chord = wing.get("root_chord_m", wing["chord_m"])
    wing_exposed = max(1e-6, s_ref - wing_root_chord * fuselage_width)

    canard_area = trim["S_c"]
    canard_span = trim["b_c"]
    canard_chord = trim.get("c_c", canard_area / max(canard_span, 1e-6))
    canard_root_chord = trim.get("c_c_root", canard_chord)
    canard_exposed = max(1e-6, canard_area - canard_root_chord * fuselage_width)

    def lifting_component(name, length, exposed_area, thickness_ratio,
                          max_thickness_x_c, sweep_deg, interference, laminar):
        swet = lifting_surface_wetted_area(exposed_area, thickness_ratio)
        ff_incompressible = (
            1.0
            + 0.6 / max(max_thickness_x_c, 1e-6) * thickness_ratio
            + 100.0 * thickness_ratio**4
        )
        return {
            "name": name,
            "kind": "lifting",
            "characteristic_length_m": length,
            "laminar_fraction": min(max(laminar, 0.0), 1.0),
            "cutoff_reynolds": cutoff_reynolds(length, roughness),
            "ff_incompressible": ff_incompressible,
            "cos_sweep_pow": math.cos(math.radians(sweep_deg)) ** 0.28,
            "area_factor": interference * swet / s_ref,
        }

    def fixed_ff_component(name, length, swet, form_factor, interference, laminar):
        return {
            "name": name,
            "kind": "fixed_ff",
            "characteristic_length_m": length,
            "laminar_fraction": min(max(laminar, 0.0), 1.0),
            "cutoff_reynolds": cutoff_reynolds(length, roughness),
            "form_factor": form_factor,
            "area_factor": interference * swet / s_ref,
        }

    components = [
        lifting_component(
            "wing", wing["chord_m"], wing_exposed,
            aircraft["wing_thickness_ratio"], aircraft["wing_max_thickness_x_c"],
            aircraft.get("wing_sweep_half_chord_deg", 0.0),
            aircraft["wing_interference_factor"], aircraft["wing_laminar_fraction"],
        ),
        lifting_component(
            "canard", canard_chord, canard_exposed,
            aircraft["canard_thickness_ratio"], aircraft["canard_max_thickness_x_c"],
            aircraft.get("canard_sweep_half_chord_deg", 0.0),
            aircraft["canard_interference_factor"], aircraft["canard_laminar_fraction"],
        ),
        fixed_ff_component(
            "fuselage", fuselage_length,
            body_wetted_area(fuselage_length, fuselage_width),
            body_form_factor(fuselage_length / fuselage_width),
            aircraft["fuselage_interference_factor"],
            aircraft["fuselage_laminar_fraction"],
        ),
    ]
    hardware_swet = aircraft.get("exposed_hardware_wetted_area_m2", 0.0)
    if hardware_swet > 0.0:
        components.append(fixed_ff_component(
            "hardware", max(fuselage_width, 1e-6), hardware_swet,
            aircraft.get("exposed_hardware_form_factor", 1.30),
            aircraft.get("hardware_interference_factor", 1.2),
            aircraft.get("hardware_laminar_fraction", 0.0),
        ))

    return {
        "S_ref": s_ref,
        "components": components,
        "excrescence_fraction": aircraft.get("excrescence_fraction", 0.0),
    }


def parasite_drag_buildup(geom, mach, reynolds_per_m):
    """Component drag build-up CD0 (wing-area referenced) and its breakdown.

    Recomputes only the flow-dependent terms (skin friction Cf and the Mach part
    of the lifting-surface form factor) against the constants prepared in
    build_drag_geometry, then sums and adds the excrescence/leakage increment.
    """
    compressibility = (1.0 + 0.144 * mach**2) ** 0.65
    lifting_compressibility = 1.34 * mach**0.18

    contributions = {}
    clean_sum = 0.0
    for component in geom["components"]:
        reynolds = reynolds_per_m * component["characteristic_length_m"]
        if reynolds > component["cutoff_reynolds"]:
            reynolds = component["cutoff_reynolds"]
        reynolds = max(reynolds, 1.0)
        cf_laminar = 1.328 / math.sqrt(reynolds)
        cf_turbulent = 0.455 / (math.log10(reynolds) ** 2.58 * compressibility)
        laminar = component["laminar_fraction"]
        cf = laminar * cf_laminar + (1.0 - laminar) * cf_turbulent

        if component["kind"] == "lifting":
            form_factor = component["ff_incompressible"] * max(
                1.0, lifting_compressibility * component["cos_sweep_pow"]
            )
        else:
            form_factor = component["form_factor"]

        cd0 = cf * form_factor * component["area_factor"]
        contributions[component["name"]] = cd0
        clean_sum += cd0

    cd0_misc = geom["excrescence_fraction"] * clean_sum
    cd0_component = clean_sum + cd0_misc

    return {
        "CD0_component": cd0_component,
        "CD0_wing": contributions.get("wing", 0.0),
        "CD0_canard": contributions.get("canard", 0.0),
        "CD0_fuselage": contributions.get("fuselage", 0.0),
        "CD0_hardware": contributions.get("hardware", 0.0),
        "CD0_misc": cd0_misc,
        "CD0_clean_sum": clean_sum,
        "mach": mach,
        "reynolds_per_m": reynolds_per_m,
    }
