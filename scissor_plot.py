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


def mach_number(altitude_m, speed_m_s):
    """Flight Mach number at a given ISA altitude and true airspeed.

    Argument order matches xfoil_wrapper.mach_number (altitude first, then speed).
    """
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


# ---------------------------------------------------------------------------
# Finite-wing maximum lift (DATCOM / Raymer high-aspect-ratio method)
# AircraftDesign2 (AD2) slides 16-24.
# ---------------------------------------------------------------------------
#
# The 2D section c_lmax overestimates what a finite wing reaches: the wing must
# fly at a higher angle than the airfoil (finite span), and tip-region effects
# trigger stall before every section is at its own maximum. AD2 slide 22 gives
# the Raymer/DATCOM high-aspect-ratio estimate
#
#     CL_max = (CL_max / cl_max) * cl_max + dCL_max(Mach)
#
# where the ratio (CL_max / cl_max) is read from Raymer Fig. 12.8 as a function
# of the leading-edge sharpness parameter dY [% chord] and the leading-edge
# sweep, and dCL_max is a Mach-number correction (Raymer Fig. 12.9) that AD2
# slide 22 states is "not relevant" at the low speeds where CL_max actually
# matters (take-off / landing / stall, M < 0.2). We therefore evaluate the ratio
# from the section c_lmax taken at the right (low-speed) Reynolds number and set
# the Mach term to zero below M = 0.2.
#
# This method is only valid for HIGH aspect-ratio wings (slide 20). Low-AR wings
# develop leading-edge vortices that *raise* CL_max and need the separate slide
# 26 method; that is out of scope here (the wing AR = 7, canard AR = 5 both sit
# well above the high-AR boundary for an unswept planform).

# Raymer Fig. 12.8: (CL_max / cl_max) for high-aspect-ratio wings, digitized on a
# grid of LE sharpness dY [% chord] x leading-edge sweep [deg]. Bilinearly
# interpolated; inputs are clamped to the table range.
_CLMAX_RATIO_DY_PCT = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
_CLMAX_RATIO_SWEEP_DEG = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
_CLMAX_RATIO_TABLE = {
    # sweep_deg: ratios at dY = 1,2,3,4,5,6 % chord
    0.0:  (0.70, 0.84, 0.90, 0.93, 0.95, 0.955),
    10.0: (0.68, 0.81, 0.875, 0.905, 0.925, 0.93),
    20.0: (0.62, 0.75, 0.82, 0.86, 0.885, 0.895),
    30.0: (0.53, 0.66, 0.74, 0.79, 0.825, 0.84),
    40.0: (0.43, 0.555, 0.635, 0.69, 0.73, 0.755),
    50.0: (0.33, 0.44, 0.525, 0.585, 0.625, 0.65),
}


def leading_edge_sweep_deg(quarter_chord_sweep_deg, aspect_ratio, taper):
    """Leading-edge sweep [deg] from the quarter-chord sweep and planform.

    Standard sweep-conversion identity
        tan(L_LE) = tan(L_c/4) + (1/A) * (1 - taper) / (1 + taper).
    For a straight (unswept-c/4) tapered wing this returns the small positive LE
    sweep the taper introduces, which the DATCOM CL_max method needs.
    """
    tan_le = math.tan(math.radians(quarter_chord_sweep_deg)) + (
        (1.0 - taper) / (aspect_ratio * (1.0 + taper))
    )
    return math.degrees(math.atan(tan_le))


def le_sharpness_parameter(thickness_ratio, dy_per_thickness=26.0):
    """LE sharpness parameter dY [% chord] (AD2 slide 19).

    dY is the ordinate difference between 0.15% and 6% chord on the upper
    surface; when the exact airfoil ordinates are not used it is estimated as
    dY = k * (t/c), with k ~= 26 for round-nosed (NACA 4-/5-digit-like) sections
    and ~21.3 for the sharper 6-series. The sharper the nose (smaller dY), the
    weaker the LE vortices and the lower the wing CL_max.
    """
    return dy_per_thickness * thickness_ratio


def _interpolate_1d(xs, ys, x):
    """Linear interpolation with clamping at the table ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            span = xs[i] - xs[i - 1]
            frac = (x - xs[i - 1]) / span if span else 0.0
            return ys[i - 1] + frac * (ys[i] - ys[i - 1])
    return ys[-1]


def datcom_clmax_ratio(delta_y_pct, sweep_le_deg):
    """(CL_max / cl_max) for a high-AR wing (Raymer Fig. 12.8, AD2 slide 22)."""
    # Ratio versus dY at each tabulated sweep, then interpolate across sweep.
    ratios_at_sweeps = [
        _interpolate_1d(_CLMAX_RATIO_DY_PCT, _CLMAX_RATIO_TABLE[sweep], delta_y_pct)
        for sweep in _CLMAX_RATIO_SWEEP_DEG
    ]
    return _interpolate_1d(_CLMAX_RATIO_SWEEP_DEG, ratios_at_sweeps, sweep_le_deg)


def datcom_clmax_mach_correction(mach):
    """dCL_max Mach correction (Raymer Fig. 12.9, AD2 slides 22-23).

    Zero up to M = 0.2 (where CL_max is evaluated for take-off/landing), then a
    mild reduction at higher subsonic Mach. Kept as a small linear fit so the
    method stays valid if a non-low-speed condition is ever passed in.
    """
    if mach <= 0.2:
        return 0.0
    return -0.4 * (mach - 0.2)


def finite_wing_clmax(
    section_clmax,
    thickness_ratio,
    sweep_le_deg=0.0,
    mach=0.0,
    delta_y_pct=None,
    dy_per_thickness=26.0,
):
    """High-AR finite-wing CL_max from the 2D section c_lmax (AD2 slides 16-24).

    section_clmax must be the airfoil c_lmax at the *low-speed* Reynolds number of
    the stall condition (slide 22). Returns a dict with the CL_max and the
    intermediate DATCOM terms for reporting.
    """
    if delta_y_pct is None:
        delta_y_pct = le_sharpness_parameter(thickness_ratio, dy_per_thickness)
    ratio = datcom_clmax_ratio(delta_y_pct, sweep_le_deg)
    mach_term = datcom_clmax_mach_correction(mach)
    cl_max = ratio * section_clmax + mach_term
    return {
        "CL_max": cl_max,
        "section_clmax": section_clmax,
        "clmax_ratio": ratio,
        "delta_y_pct": delta_y_pct,
        "mach_correction": mach_term,
        "sweep_le_deg": sweep_le_deg,
        "mach": mach,
    }


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
