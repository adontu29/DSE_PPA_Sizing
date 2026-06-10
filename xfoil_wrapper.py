"""Small XFOIL wrapper used by the active sizing script.

The wrapper keeps XFOIL-specific subprocess handling out of `simple_sizing.py`.
It returns 2D section data only; finite-wing lift slopes remain handled by the
DATCOM method in `scissor_plot.py`.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path


GAMMA_AIR = 1.4
R_AIR = 287.05
T0_ISA = 288.15
LAPSE_ISA = 0.0065
RHO0_ISA = 1.225
MU0_SUTHERLAND = 1.716e-5
T0_SUTHERLAND = 273.15
SUTHERLAND_C = 110.4

_POLAR_CACHE = {}
_LAST_RESULT_CACHE = {}


def isa_temperature(altitude_m):
    return T0_ISA - LAPSE_ISA * altitude_m


def isa_density(altitude_m):
    temperature = isa_temperature(altitude_m)
    pressure_ratio = (temperature / T0_ISA) ** (
        9.80665 / (LAPSE_ISA * R_AIR)
    )
    return RHO0_ISA * pressure_ratio * T0_ISA / temperature


def dynamic_viscosity_sutherland(temperature_K):
    return (
        MU0_SUTHERLAND
        * (temperature_K / T0_SUTHERLAND) ** 1.5
        * (T0_SUTHERLAND + SUTHERLAND_C)
        / (temperature_K + SUTHERLAND_C)
    )


def mach_number(altitude_m, true_speed_m_s):
    speed_of_sound = math.sqrt(GAMMA_AIR * R_AIR * isa_temperature(altitude_m))
    return true_speed_m_s / speed_of_sound


def reynolds_number(altitude_m, true_speed_m_s, chord_m):
    temperature = isa_temperature(altitude_m)
    density = isa_density(altitude_m)
    viscosity = dynamic_viscosity_sutherland(temperature)
    return density * true_speed_m_s * chord_m / viscosity


def datcom_efficiency_from_section_slope(cl_alpha_per_rad):
    """Convert a 2D lift-curve slope to the DATCOM airfoil-efficiency factor."""
    if cl_alpha_per_rad <= 0.0:
        return 0.95
    return max(0.70, min(1.05, cl_alpha_per_rad / (2.0 * math.pi)))


def _resolve_executable(xfoil_path):
    if not xfoil_path:
        return None
    path = Path(xfoil_path)
    if path.is_file():
        return str(path.resolve())
    return shutil.which(str(xfoil_path))


def _parse_xfoil_polar(polar_path):
    rows = []
    with open(polar_path, "r", encoding="utf-8", errors="ignore") as polar_file:
        for line in polar_file:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                alpha, cl, cd, cdp, cm = [float(value) for value in parts[:5]]
            except ValueError:
                continue
            rows.append({
                "alpha_deg": alpha,
                "cl": cl,
                "cd": cd,
                "cdp": cdp,
                "cm": cm,
            })
    if not rows:
        raise ValueError("No converged polar rows were found.")
    return rows


def _linear_fit(xs, ys):
    n = len(xs)
    if n < 2:
        raise ValueError("At least two points are required for a linear fit.")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0.0:
        raise ValueError("Cannot fit a vertical or degenerate line.")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _interpolate(xs, ys, target):
    pairs = sorted(zip(xs, ys), key=lambda item: item[0])
    for x, y in pairs:
        if abs(x - target) < 1e-12:
            return y
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        lower = min(x0, x1)
        upper = max(x0, x1)
        if lower <= target <= upper and abs(x1 - x0) > 1e-12:
            fraction = (target - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    return min(pairs, key=lambda item: abs(item[0] - target))[1]


def summarize_polar_rows(
    rows,
    *,
    airfoil,
    reynolds,
    mach,
    x_transition,
    source,
):
    clean_rows = [
        row
        for row in rows
        if all(math.isfinite(row[name]) for name in ("alpha_deg", "cl", "cd", "cm"))
        and row["cd"] > 0.0
    ]
    if len(clean_rows) < 3:
        raise ValueError("Too few usable polar rows were found.")

    alpha = [row["alpha_deg"] for row in clean_rows]
    cl = [row["cl"] for row in clean_rows]
    cd = [row["cd"] for row in clean_rows]
    cm = [row["cm"] for row in clean_rows]

    linear_rows = [
        row for row in clean_rows if -2.0 <= row["alpha_deg"] <= 6.0
    ]
    if len(linear_rows) < 3:
        linear_rows = clean_rows
    slope_per_deg, intercept = _linear_fit(
        [row["alpha_deg"] for row in linear_rows],
        [row["cl"] for row in linear_rows],
    )
    cl_alpha_per_rad = slope_per_deg * 180.0 / math.pi
    cl_at_alpha0 = intercept

    positive_rows = [row for row in clean_rows if row["alpha_deg"] >= 0.0]
    if not positive_rows:
        positive_rows = clean_rows
    clmax_row = max(positive_rows, key=lambda row: row["cl"])
    cl_cd = [cl_value / cd_value for cl_value, cd_value in zip(cl, cd)]
    best_ld_index = max(range(len(clean_rows)), key=lambda index: cl_cd[index])

    linear_cl = [row["cl"] for row in linear_rows]
    linear_cm = [row["cm"] for row in linear_rows]
    cm_at_zero_lift = _interpolate(linear_cl, linear_cm, 0.0)

    return {
        "airfoil": airfoil,
        "source": source,
        "reynolds": float(reynolds),
        "mach": float(mach),
        "x_transition": float(x_transition),
        "cl_alpha_per_rad": float(cl_alpha_per_rad),
        "cl_at_alpha0": float(cl_at_alpha0),
        "cl_max": float(clmax_row["cl"]),
        "alpha_cl_max_deg": float(clmax_row["alpha_deg"]),
        "cm_at_alpha0": float(_interpolate(alpha, cm, 0.0)),
        "cm_at_zero_lift": float(cm_at_zero_lift),
        "cd_at_alpha0": float(_interpolate(alpha, cd, 0.0)),
        "best_cl_cd": float(cl_cd[best_ld_index]),
        "best_cl_cd_alpha_deg": float(clean_rows[best_ld_index]["alpha_deg"]),
        "polar_points": len(clean_rows),
    }


def _rounded_key(value, step):
    if step <= 0.0:
        return float(value)
    return float(round(value / step) * step)


def _run_xfoil_airfoil(
    *,
    xfoil_path,
    airfoil,
    reynolds,
    mach,
    x_transition,
    coordinate_file=None,
    alpha_start_deg=-6.0,
    alpha_end_deg=18.0,
    alpha_step_deg=1.0,
    mach_command_min=0.20,
    timeout_s=30.0,
):
    executable = _resolve_executable(xfoil_path)
    if not executable:
        raise FileNotFoundError(f"XFOIL executable not found: {xfoil_path}")

    with tempfile.TemporaryDirectory(prefix="xfoil_") as temp_name:
        temp_dir = Path(temp_name)
        polar_path = temp_dir / "polar.txt"

        if coordinate_file:
            coordinate_path = Path(coordinate_file)
            if not coordinate_path.is_file():
                raise FileNotFoundError(f"Airfoil coordinate file not found: {coordinate_file}")
            local_coordinate = temp_dir / coordinate_path.name
            shutil.copy2(coordinate_path, local_coordinate)
            airfoil_command = f"LOAD {local_coordinate.name}"
        elif airfoil.lower().replace(" ", "") == "naca0012":
            airfoil_command = "NACA 0012"
        else:
            raise ValueError(f"No coordinate source was provided for {airfoil}.")

        xfoil_mach = mach if mach >= mach_command_min else 0.0
        commands = [
            "PLOP",
            "G F",
            "",
            airfoil_command,
            "PANE",
            "OPER",
            f"VISC {reynolds:.0f}",
        ]
        if xfoil_mach > 0.0:
            commands.append(f"MACH {xfoil_mach:.4f}")
        step = abs(alpha_step_deg) or 0.5
        commands.extend([
            "ITER 300",
            "VPAR",
            f"XTR {x_transition:.3f} {x_transition:.3f}",
            "",
            "PACC",
            # XFOIL truncates filenames to ~64 chars; an absolute temp path can
            # overflow that and silently write a truncated name. The polar is
            # written relative to cwd (= temp_dir), so pass the short name.
            polar_path.name,
            "",
        ])
        # Sweep OUTWARD from alpha=0 (which converges easily) instead of cold-
        # starting at a hard negative alpha and marching up: at low Reynolds the
        # negative-alpha points fail and XFOIL halts the whole sequence after a
        # few consecutive failures, leaving an empty polar. Splitting the sweep
        # at 0 means a stall-side failure cannot wipe out the low-alpha points.
        if alpha_end_deg > 0.0:
            commands.append(f"ASEQ 0.00 {alpha_end_deg:.2f} {step:.2f}")
        if alpha_start_deg < 0.0:
            commands.append("INIT")
            commands.append(f"ASEQ {-step:.2f} {alpha_start_deg:.2f} {-step:.2f}")
        commands.extend([
            "PACC",
            "",
            "QUIT",
            "",
        ])
        completed = subprocess.run(
            [executable],
            input="\n".join(commands),
            text=True,
            capture_output=True,
            cwd=temp_dir,
            timeout=timeout_s,
        )

        if not polar_path.exists():
            tail = (completed.stderr.strip() or completed.stdout.strip())[-500:]
            raise RuntimeError(f"XFOIL did not produce a polar for {airfoil}. {tail}")

        rows = _parse_xfoil_polar(polar_path)
        return summarize_polar_rows(
            rows,
            airfoil=airfoil,
            reynolds=reynolds,
            mach=xfoil_mach,
            x_transition=x_transition,
            source="xfoil",
        )


def analyze_airfoil(
    *,
    xfoil_path,
    airfoil,
    reynolds,
    mach,
    x_transition=0.50,
    coordinate_file=None,
    reynolds_rounding=0.0,
    reynolds_update_threshold=5000.0,
    mach_rounding=0.02,
    alpha_start_deg=-6.0,
    alpha_end_deg=18.0,
    alpha_step_deg=1.0,
    mach_command_min=0.20,
    timeout_s=30.0,
):
    """Run or reuse a cached XFOIL polar summary for one airfoil."""
    rounded_re = _rounded_key(reynolds, reynolds_rounding)
    rounded_mach = _rounded_key(mach, mach_rounding)
    xfoil_mach = rounded_mach if rounded_mach >= mach_command_min else 0.0
    base_key = (
        str(Path(xfoil_path).resolve()) if Path(str(xfoil_path)).is_file() else str(xfoil_path),
        airfoil.lower().replace(" ", ""),
        xfoil_mach,
        round(float(x_transition), 3),
        round(float(alpha_start_deg), 2),
        round(float(alpha_end_deg), 2),
        round(float(alpha_step_deg), 2),
        None if coordinate_file is None else str(Path(coordinate_file).resolve()),
    )
    last_result = _LAST_RESULT_CACHE.get(base_key)
    if (
        last_result is not None
        and reynolds_update_threshold > 0.0
        and abs(reynolds - last_result["requested_reynolds"]) < reynolds_update_threshold
    ):
        result = dict(last_result)
        result["cache_hit"] = True
        result["reused_for_reynolds_delta"] = float(reynolds - last_result["requested_reynolds"])
        result["requested_reynolds"] = float(reynolds)
        result["requested_mach"] = float(mach)
        _LAST_RESULT_CACHE[base_key] = dict(result)
        return result

    cache_key = (
        airfoil.lower().replace(" ", ""),
        rounded_re,
        xfoil_mach,
        round(float(x_transition), 3),
        round(float(alpha_start_deg), 2),
        round(float(alpha_end_deg), 2),
        round(float(alpha_step_deg), 2),
        None if coordinate_file is None else str(Path(coordinate_file).resolve()),
    )
    if cache_key in _POLAR_CACHE:
        result = dict(_POLAR_CACHE[cache_key])
        result["cache_hit"] = True
        result["requested_reynolds"] = float(reynolds)
        result["requested_mach"] = float(mach)
        return result

    def run_with_low_mach_retry(candidate_re):
        try:
            return _run_xfoil_airfoil(
                xfoil_path=xfoil_path,
                airfoil=airfoil,
                reynolds=max(1.0, candidate_re),
                mach=max(0.0, xfoil_mach),
                x_transition=x_transition,
                coordinate_file=coordinate_file,
                alpha_start_deg=alpha_start_deg,
                alpha_end_deg=alpha_end_deg,
                alpha_step_deg=alpha_step_deg,
                mach_command_min=mach_command_min,
                timeout_s=timeout_s,
            )
        except (RuntimeError, ValueError) as first_error:
            retry_mach = 0.08 if 0.0 < rounded_mach < mach_command_min else None
            if retry_mach is None:
                raise
            result = _run_xfoil_airfoil(
                xfoil_path=xfoil_path,
                airfoil=airfoil,
                reynolds=max(1.0, candidate_re),
                mach=retry_mach,
                x_transition=x_transition,
                coordinate_file=coordinate_file,
                alpha_start_deg=alpha_start_deg,
                alpha_end_deg=alpha_end_deg,
                alpha_step_deg=alpha_step_deg,
                mach_command_min=0.0,
                timeout_s=timeout_s,
            )
            result["retry_reason"] = str(first_error)
            result["retry_mach"] = retry_mach
            return result

    candidate_reynolds = []

    def add_candidate_reynolds(value):
        if value <= 0.0:
            return
        if not any(abs(value - existing) < 1.0 for existing in candidate_reynolds):
            candidate_reynolds.append(float(value))

    add_candidate_reynolds(rounded_re)
    if reynolds_update_threshold > 0.0:
        threshold_re = _rounded_key(reynolds, reynolds_update_threshold)
        add_candidate_reynolds(threshold_re)
        add_candidate_reynolds(threshold_re + reynolds_update_threshold)
        add_candidate_reynolds(threshold_re - reynolds_update_threshold)
    add_candidate_reynolds(_rounded_key(reynolds, 50000.0))

    errors = []
    result = None
    for candidate_re in candidate_reynolds:
        try:
            result = run_with_low_mach_retry(candidate_re)
            if abs(candidate_re - rounded_re) >= 1.0:
                result["retry_reynolds"] = candidate_re
            break
        except (RuntimeError, ValueError) as error:
            errors.append(f"Re {candidate_re:.0f}: {error}")

    if result is None:
        raise ValueError("; ".join(errors[-3:]))

    result["cache_hit"] = False
    result["requested_reynolds"] = float(reynolds)
    result["requested_mach"] = float(mach)
    _POLAR_CACHE[cache_key] = dict(result)
    _LAST_RESULT_CACHE[base_key] = dict(result)
    return result


def analyze_airfoil_pair(
    *,
    xfoil_path,
    sd7037_file,
    wing_reynolds,
    canard_reynolds,
    mach,
    x_transition=0.50,
    reynolds_rounding=0.0,
    reynolds_update_threshold=5000.0,
    mach_rounding=0.02,
    alpha_start_deg=-6.0,
    alpha_end_deg=18.0,
    alpha_step_deg=1.0,
    mach_command_min=0.20,
    timeout_s=30.0,
):
    warnings = []
    wing = None
    canard = None

    try:
        wing = analyze_airfoil(
            xfoil_path=xfoil_path,
            airfoil="SD7037",
            reynolds=wing_reynolds,
            mach=mach,
            x_transition=x_transition,
            coordinate_file=sd7037_file,
            reynolds_rounding=reynolds_rounding,
            reynolds_update_threshold=reynolds_update_threshold,
            mach_rounding=mach_rounding,
            alpha_start_deg=alpha_start_deg,
            alpha_end_deg=alpha_end_deg,
            alpha_step_deg=alpha_step_deg,
            mach_command_min=mach_command_min,
            timeout_s=timeout_s,
        )
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        warnings.append(f"SD7037 XFOIL analysis failed: {error}")

    try:
        canard = analyze_airfoil(
            xfoil_path=xfoil_path,
            airfoil="NACA 0012",
            reynolds=canard_reynolds,
            mach=mach,
            x_transition=x_transition,
            reynolds_rounding=reynolds_rounding,
            reynolds_update_threshold=reynolds_update_threshold,
            mach_rounding=mach_rounding,
            alpha_start_deg=alpha_start_deg,
            alpha_end_deg=alpha_end_deg,
            alpha_step_deg=alpha_step_deg,
            mach_command_min=mach_command_min,
            timeout_s=timeout_s,
        )
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        warnings.append(f"NACA 0012 XFOIL analysis failed: {error}")

    return {
        "wing": wing,
        "canard": canard,
        "warnings": warnings,
        "xfoil_path": str(xfoil_path),
        "sd7037_file": str(sd7037_file),
    }
