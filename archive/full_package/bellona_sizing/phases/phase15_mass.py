"""Phase 15: preliminary mass, CG, inertia, and wing-layout datum bookkeeping."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def phase15_mass(S_w, S_c, fuselage_length, P_motor_cont_W, m_battery_kg,
                 mission_equipment_mass_kg: float,
                 n_rotors: int = 4,
                 prop_diameter_m: Optional[float] = None,
                 propeller_mass_total_kg: Optional[float] = None,
                 b_w: Optional[float] = None,
                 c_bar_w: Optional[float] = None,
                 x_ac_w: Optional[float] = None,
                 x_ac_c: Optional[float] = None,
                 wing_mac_le_x_m: float = 0.0,
                 external_tow_load_N: float = 0.0,
                 g: float = 9.80665,
                 wing_areal_density_kg_m2: float = 2.00,
                 canard_areal_density_kg_m2: float = 1.70,
                 fuselage_linear_density_kg_m: float = 2.20,
                 boom_landing_gear_mass_kg: float = 2.50,
                 motor_specific_mass_kg_W: float = 0.00027,
                 esc_specific_mass_kg_W: float = 0.00008,
                 prop_mass_coeff_kg_m2: float = 0.10,
                 avionics_mass_kg: float = 1.50,
                 wiring_fraction: float = 0.06,
                 mass_contingency_fraction: float = 0.08,
                 Ixx_radius_fraction_span: float = 0.20,
                 Iyy_radius_fraction_fuselage: float = 0.35,
                 Izz_radius_fraction_span: float = 0.22) -> Dict:
    """Preliminary mass breakdown using project-level first-cut coefficients."""
    if S_w <= 0.0 or S_c <= 0.0:
        raise ValueError("Wing and canard areas must be positive.")
    if fuselage_length <= 0.0:
        raise ValueError("fuselage_length must be positive.")
    if P_motor_cont_W <= 0.0:
        raise ValueError("P_motor_cont_W must be positive.")
    if m_battery_kg <= 0.0:
        raise ValueError("m_battery_kg must be positive.")
    if mission_equipment_mass_kg < 0.0:
        raise ValueError("mission_equipment_mass_kg must be non-negative.")
    if n_rotors <= 0:
        raise ValueError("n_rotors must be positive.")
    if external_tow_load_N < 0.0:
        raise ValueError("external_tow_load_N must be non-negative.")
    if g <= 0.0:
        raise ValueError("g must be positive.")

    coefficients = [
        wing_areal_density_kg_m2,
        canard_areal_density_kg_m2,
        fuselage_linear_density_kg_m,
        boom_landing_gear_mass_kg,
        motor_specific_mass_kg_W,
        esc_specific_mass_kg_W,
        prop_mass_coeff_kg_m2,
        avionics_mass_kg,
        wiring_fraction,
        mass_contingency_fraction,
        Ixx_radius_fraction_span,
        Iyy_radius_fraction_fuselage,
        Izz_radius_fraction_span,
    ]
    if any(value < 0.0 for value in coefficients):
        raise ValueError("Mass and inertia coefficients must be non-negative.")

    prop_diameter_used = 0.0 if prop_diameter_m is None else float(prop_diameter_m)
    if prop_diameter_used < 0.0:
        raise ValueError("prop_diameter_m must be non-negative when provided.")
    if propeller_mass_total_kg is not None and propeller_mass_total_kg < 0.0:
        raise ValueError("propeller_mass_total_kg must be non-negative when provided.")

    b_w_used = np.sqrt(S_w * 7.0) if b_w is None else float(b_w)
    c_bar_used = S_w / b_w_used if c_bar_w is None else float(c_bar_w)
    if b_w_used <= 0.0 or c_bar_used <= 0.0:
        raise ValueError("Wing span and mean chord must be positive.")

    wing_mac_le_x_m = float(wing_mac_le_x_m)
    x_wing_local = 0.25 * c_bar_used if x_ac_w is None else float(x_ac_w)
    x_canard_local = (
        x_wing_local - 2.5 * c_bar_used
        if x_ac_c is None
        else float(x_ac_c)
    )
    x_reference_fuselage = 0.0
    x_wing_fuselage = wing_mac_le_x_m + x_wing_local
    x_canard_fuselage = wing_mac_le_x_m + x_canard_local

    m_wing = wing_areal_density_kg_m2 * S_w
    m_canard = canard_areal_density_kg_m2 * S_c
    m_fuselage = fuselage_linear_density_kg_m * fuselage_length
    m_boom_landing_gear = boom_landing_gear_mass_kg
    m_motor = motor_specific_mass_kg_W * P_motor_cont_W
    m_ESC = esc_specific_mass_kg_W * P_motor_cont_W
    m_propeller = (
        n_rotors * prop_mass_coeff_kg_m2 * prop_diameter_used**2
        if propeller_mass_total_kg is None
        else float(propeller_mass_total_kg)
    )
    m_avionics = avionics_mass_kg
    m_mission_equipment = mission_equipment_mass_kg

    base_masses = {
        "wing": m_wing,
        "canard": m_canard,
        "fuselage": m_fuselage,
        "boom_landing_gear": m_boom_landing_gear,
        "motors": m_motor,
        "ESCs": m_ESC,
        "propellers": m_propeller,
        "battery": m_battery_kg,
        "avionics": m_avionics,
        "mission_equipment": m_mission_equipment,
    }
    base_mass = sum(base_masses.values())
    m_wiring = wiring_fraction * (m_motor + m_ESC + m_avionics)
    contingency_base = base_mass + m_wiring
    m_contingency = mass_contingency_fraction * contingency_base
    MTOW_estimate_kg = contingency_base + m_contingency

    component_masses = dict(base_masses)
    component_masses["wiring"] = m_wiring
    component_masses["contingency"] = m_contingency

    x_locations_fuselage = {
        "wing": x_wing_fuselage,
        "canard": x_canard_fuselage,
        "fuselage": x_reference_fuselage,
        "boom_landing_gear": x_reference_fuselage,
        "motors": x_reference_fuselage,
        "ESCs": x_reference_fuselage,
        "propellers": x_reference_fuselage,
        "battery": x_reference_fuselage,
        "avionics": x_reference_fuselage,
        "mission_equipment": x_reference_fuselage,
        "wiring": x_reference_fuselage,
        "contingency": x_reference_fuselage,
    }
    x_locations = {
        name: value - wing_mac_le_x_m
        for name, value in x_locations_fuselage.items()
    }
    x_CG_fuselage = (
        sum(
            component_masses[name] * x_locations_fuselage[name]
            for name in component_masses
        )
        / MTOW_estimate_kg
    )
    x_CG = x_CG_fuselage - wing_mac_le_x_m

    Ixx = MTOW_estimate_kg * (Ixx_radius_fraction_span * b_w_used) ** 2
    Iyy = MTOW_estimate_kg * (Iyy_radius_fraction_fuselage * fuselage_length) ** 2
    Izz = MTOW_estimate_kg * (Izz_radius_fraction_span * b_w_used) ** 2

    structure_mass = m_wing + m_canard + m_fuselage + m_boom_landing_gear
    propulsion_mass = m_motor + m_ESC + m_propeller
    fixed_equipment_mass = m_avionics + m_mission_equipment
    external_tow_load_equivalent_kg = external_tow_load_N / g

    warnings = [
        "Phase 15 is a first-cut mass model; replace areal densities, propulsion masses, and CG locations with CAD and datasheet values.",
        "All nonlifting components are placed at a provisional fuselage/equipment datum for the preliminary CG estimate.",
        "wing_mac_le_x_m shifts the wing MAC leading edge relative to that provisional mass datum; the canard arm remains fixed in this first-cut layout solve.",
        "The contingency mass is included in MTOW but has no independent CAD location yet.",
    ]
    if prop_diameter_m is None and propeller_mass_total_kg is None:
        warnings.append(
            "No propeller diameter was provided; propeller mass was set to zero."
        )
    if external_tow_load_N > 0.0:
        warnings.append(
            "external_tow_load_N is reported as an external load and is not included in UAV MTOW."
        )

    return {
        "MTOW_estimate_kg": float(MTOW_estimate_kg),
        "empty_mass_no_battery_no_mission_kg": float(
            MTOW_estimate_kg - m_battery_kg - m_mission_equipment
        ),
        "structure_mass_kg": float(structure_mass),
        "propulsion_mass_kg": float(propulsion_mass),
        "fixed_equipment_mass_kg": float(fixed_equipment_mass),
        "component_masses_kg": {name: float(value) for name, value in component_masses.items()},
        "component_x_locations_m": {name: float(value) for name, value in x_locations.items()},
        "component_x_locations_fuselage_m": {
            name: float(value) for name, value in x_locations_fuselage.items()
        },
        "wing_mac_le_x_m": float(wing_mac_le_x_m),
        "x_CG_fuselage_m": float(x_CG_fuselage),
        "x_CG_m": float(x_CG),
        "x_CG_over_wing_mac": float(x_CG / c_bar_used),
        "I_tensor_kg_m2": {
            "Ixx": float(Ixx),
            "Iyy": float(Iyy),
            "Izz": float(Izz),
            "Ixy": 0.0,
            "Ixz": 0.0,
            "Iyz": 0.0,
        },
        "m_wing_kg": float(m_wing),
        "m_canard_kg": float(m_canard),
        "m_fuselage_kg": float(m_fuselage),
        "m_boom_landing_gear_kg": float(m_boom_landing_gear),
        "m_motor_kg": float(m_motor),
        "m_ESC_kg": float(m_ESC),
        "m_propeller_kg": float(m_propeller),
        "m_battery_kg": float(m_battery_kg),
        "m_avionics_kg": float(m_avionics),
        "m_mission_equipment_kg": float(m_mission_equipment),
        "m_wiring_kg": float(m_wiring),
        "m_contingency_kg": float(m_contingency),
        "external_tow_load_N": float(external_tow_load_N),
        "external_tow_load_equivalent_kg": float(external_tow_load_equivalent_kg),
        "external_tow_load_included_in_MTOW": False,
        "mass_coefficients": {
            "wing_areal_density_kg_m2": float(wing_areal_density_kg_m2),
            "canard_areal_density_kg_m2": float(canard_areal_density_kg_m2),
            "fuselage_linear_density_kg_m": float(fuselage_linear_density_kg_m),
            "motor_specific_mass_kg_W": float(motor_specific_mass_kg_W),
            "esc_specific_mass_kg_W": float(esc_specific_mass_kg_W),
            "prop_mass_coeff_kg_m2": float(prop_mass_coeff_kg_m2),
            "propeller_mass_total_override_kg": (
                None
                if propeller_mass_total_kg is None
                else float(propeller_mass_total_kg)
            ),
            "wiring_fraction": float(wiring_fraction),
            "mass_contingency_fraction": float(mass_contingency_fraction),
        },
        "notes": [
            "m_mission_equipment_kg contains onboard netgun, sensor, and mission hardware.",
            "No separate capture mass is included in this model.",
            "m_motor_kg = motor_specific_mass_kg_W * P_motor_cont_W.",
            "m_ESC_kg = esc_specific_mass_kg_W * P_motor_cont_W.",
            "Propeller mass uses the candidate-map mass when provided; otherwise m_propeller_kg = n_rotors * prop_mass_coeff_kg_m2 * D_prop^2.",
            "Phase 15 reports MTOW_estimate_kg; Phase 16 will decide whether to iterate the sizing MTOW.",
        ],
        "warnings": warnings,
    }
