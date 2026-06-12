"""Course-method canard scissor-plot equations (TU Delft AE3211-I, Lectures 7 & 8).

Pure functions only: given geometry and aerodynamic inputs, return the
stability (aft-CG) and controllability (forward-CG) limits as functions of the
canard area ratio Sc/Sw. `simple_sizing.py` imports these so the sizing
iteration can refine wing position and canard area ratio against the curves.
This module deliberately does NOT import simple_sizing (would be circular).

Canard configuration conventions (Lecture 8, slides 32-34):
  * (Vc/V)^2     ~ 1    -- the canard is forward, in clean air (full dynamic pressure)
  * interference        -- everything the canard does to the flow lands on the WING,
                           which sits downstream in the canard wake (NOT the reverse):
                             - downwash de_da (Slingerland, wing_downwash_gradient()
                               with the canard as the generating surface), and
                             - a wake dynamic-pressure loss (Vw/V)^2,
                           both acting only on the inboard wing fraction within the
                           canard span. They reduce the wing's effective
                           C_L_alpha_{A-h}. de_da = immersed fraction = 0 recovers the
                           clean (no-interference) case.
  * l_h          < 0    -- canard is ahead of the wing aerodynamic centre
  * C_L_h        > 0    -- the canard lifts up (use its usable/max CL, not -1)

All longitudinal lengths are non-dimensionalised by the wing reference chord
c_bar. To stay consistent with the rest of simple_sizing.py, c_bar is the
geometric mean chord S/b (simple_sizing uses that as its MAC reference and
places the CG against it); the fuselage corrections are weak functions of the
exact chord definition.

Circular-fuselage assumption: fuselage height h_f is taken equal to fuselage
width b_f wherever the Torenbeek formulas call for the product b_f * h_f.
"""

import math

GAMMA_AIR = 1.4
R_AIR = 287.05
T0_ISA = 288.15
LAPSE_ISA = 0.0065


def isa_temperature(altitude_m):
    """Troposphere ISA temperature [K]."""
    return T0_ISA - LAPSE_ISA * altitude_m


def mach_number(speed_m_s, altitude_m):
    """Flight Mach number at a given true airspeed and ISA altitude."""
    speed_of_sound = math.sqrt(GAMMA_AIR * R_AIR * isa_temperature(altitude_m))
    return speed_m_s / speed_of_sound


def datcom_lift_slope(aspect_ratio, mach, sweep_half_chord_rad, eta=0.95):
    """DATCOM subsonic lift-curve slope C_L_alpha [1/rad] (Lecture 7 slide 43)."""
    beta = math.sqrt(1.0 - mach**2)
    return (2.0 * math.pi * aspect_ratio) / (
        2.0
        + math.sqrt(
            4.0
            + (aspect_ratio * beta / eta) ** 2
            * (1.0 + math.tan(sweep_half_chord_rad) ** 2 / beta**2)
        )
    )


def aircraft_less_canard_lift_slope(
    cl_alpha_wing, fuselage_width_m, span_m, wing_area_m2, root_chord_m
):
    """C_L_alpha_{A-h}, the aircraft-less-canard lift slope (Lecture 7 slide 44).

    Snet is the wing area outside the fuselage; root_chord_m is used as the
    chord at the fuselage side (a slight overestimate, since the true station
    is where the wing leaves the fuselage rather than the centreline).
    """
    s_net = wing_area_m2 - fuselage_width_m * root_chord_m
    return (
        cl_alpha_wing * (1.0 + 2.15 * fuselage_width_m / span_m) * (s_net / wing_area_m2)
        + math.pi / 2.0 * fuselage_width_m**2 / wing_area_m2
    )


def aerodynamic_centre_over_mac(
    cl_alpha_A_h,
    fuselage_width_m,
    nose_length_m,
    wing_area_m2,
    mac_m,
    span_m,
    taper,
    mean_geo_chord_m,
    sweep_quarter_chord_rad,
):
    """x_ac/MAC of the aircraft-less-canard (Lecture 7 slides 35-39).

    = wing quarter-chord (0.25) + fuselage nose term (destabilising, negative)
      + wing-root lift-loss term (stabilising, positive). Nacelle terms are
      omitted (the rotors are not lifting nacelles for this tailsitter).
    """
    x_ac_wing = 0.25
    # Fuselage contribution 1: nose lift, destabilising -> forward (negative) shift.
    d_fus1 = -(1.8 / cl_alpha_A_h) * (
        fuselage_width_m**2 * nose_length_m
    ) / (wing_area_m2 * mac_m)
    # Fuselage contribution 2: lift loss at the wing-fuselage join, stabilising.
    # Numerator uses the mean geometric chord c_g = S/b (Lecture 7 slide 39).
    d_fus2 = (
        (0.273 / (1.0 + taper))
        * (fuselage_width_m * mean_geo_chord_m * (span_m - fuselage_width_m))
        / (mac_m**2 * (span_m + 2.15 * fuselage_width_m))
        * math.tan(sweep_quarter_chord_rad)
    )
    return x_ac_wing + d_fus1 + d_fus2


def zero_lift_pitching_moment(
    cm0_airfoil,
    aspect_ratio,
    sweep_half_chord_rad,
    cl_alpha_A_h,
    fuselage_width_m,
    fuselage_length_m,
    wing_area_m2,
    mac_m,
    cl0_aircraft,
):
    """C_m_ac of the aircraft-less-canard, clean configuration (Lecture 8 slide 19).

    Wing-airfoil term + fuselage term. Flap and nacelle terms are intentionally
    dropped: this is a tailsitter, so low-speed control is handled by the rotors
    during VTOL and there is no flaps-down approach controllability case.
    """
    cm_ac_wing = cm0_airfoil * (
        aspect_ratio * math.cos(sweep_half_chord_rad) ** 2
        / (aspect_ratio + 2.0 * math.cos(sweep_half_chord_rad))
    )
    cm_ac_fus = (
        -1.8
        * (1.0 - 2.5 * fuselage_width_m / fuselage_length_m)
        * (math.pi * fuselage_width_m**2 * fuselage_length_m)
        / (4.0 * wing_area_m2 * mac_m)
        * cl0_aircraft
        / cl_alpha_A_h
    )
    return cm_ac_wing + cm_ac_fus


def wing_downwash_gradient(
    cl_alpha_wing, aspect_ratio, sweep_quarter_chord_rad, r, m_tv=0.0
):
    """Slingerland downwash gradient de/da behind a lifting surface (Lecture 7 slide 47).

    Generic: pass the GENERATING surface's lift slope, aspect ratio and quarter-
    chord sweep. For this canard layout the generating surface is the canard, so
    the result is the downwash the wing (downstream) experiences.

    Arguments (Λ in radians, evaluated at the quarter chord):
      r    = l_h / (b/2)  -- arm to the downstream surface over the GENERATING
             surface's semi-span; pass a positive magnitude (the 1/r and sqrt
             terms assume r > 0).
      m_tv = vertical offset of the downstream surface above the generating
             surface's root-chord plane, over b/2. m_tv = 0 collapses the
             second {.} factor to 1.

    Returns de/da (dimensionless). Larger r (surfaces further apart) and higher
    aspect ratio both reduce the downwash, as expected.
    """
    k_sweep = (
        (0.1124 + 0.1265 * sweep_quarter_chord_rad + 0.1766 * sweep_quarter_chord_rad**2)
        / r**2
        + 0.1024 / r
        + 2.0
    )
    k_sweep_zero = 0.1124 / r**2 + 0.1024 / r + 2.0

    term1 = (r / (r**2 + m_tv**2)) * 0.4876 / math.sqrt(r**2 + 0.6319 + m_tv**2)
    term2 = (
        1.0 + (r**2 / (r**2 + 0.7915 + 5.0734 * m_tv**2)) ** 0.3113
    ) * (1.0 - math.sqrt(m_tv**2 / (1.0 + m_tv**2)))

    return (k_sweep / k_sweep_zero) * (term1 + term2) * cl_alpha_wing / (math.pi * aspect_ratio)


def scissor_cg_limits(
    area_ratio,
    *,
    x_ac_over_mac,
    static_margin,
    cl_alpha_canard,
    cl_alpha_A_h,
    cl_h_control,
    cl_A_h_control,
    cmac,
    lh_over_mac,
    de_da=0.0,
    wing_immersed_fraction=0.0,
    wing_wake_dynamic_pressure_ratio=1.0,
    canard_speed_ratio_sq=1.0,
):
    """Forward (controllability) and aft (stability) CG limits at one Sc/Sw.

    Returns (x_forward_over_mac, x_aft_over_mac), both as x_cg/MAC measured
    from the wing MAC leading edge. For a canard lh_over_mac < 0, so both
    curves slope down to the left and the controllability curve (steeper) opens
    the feasible band as the area ratio grows.

    Canard-wing interference (the canard is forward, in clean air; the wing sits
    in the canard wake) enters through the WING side, not the canard:
      * canard_speed_ratio_sq = (Vc/V)^2 ~ 1 -- canard sees the freestream.
      * the inboard wing fraction `wing_immersed_fraction` (k) within the canard
        wake loses both lift slope (downwash de_da) and dynamic pressure
        (wing_wake_dynamic_pressure_ratio = (Vw/V)^2); the clean outboard
        fraction is unaffected. So the wing's effective C_L_alpha_{A-h} is scaled
        by f_wing below, which sits in the DENOMINATOR of the canard's stability
        term. de_da = k = 0 recovers the clean (no-interference) case.
    """
    # Effective wing lift-slope factor: clean outboard part + immersed inboard
    # part (reduced by downwash and wake dynamic pressure).
    k = wing_immersed_fraction
    f_wing = (1.0 - k) + k * (1.0 - de_da) * wing_wake_dynamic_pressure_ratio

    # Stability / aft CG limit -- Lecture 7 slide 34, Lecture 8 slides 5 & 34.
    stability_slope = (
        cl_alpha_canard * canard_speed_ratio_sq / (cl_alpha_A_h * f_wing)
        * lh_over_mac
    )
    x_aft = x_ac_over_mac + stability_slope * area_ratio - static_margin

    # Controllability / forward CG limit -- Lecture 8 slides 16 & 33. The canard
    # control surface is in clean air, so it uses (Vc/V)^2 directly.
    control_slope = (cl_h_control / cl_A_h_control) * lh_over_mac * canard_speed_ratio_sq
    x_forward = x_ac_over_mac - cmac / cl_A_h_control + control_slope * area_ratio

    return x_forward, x_aft
