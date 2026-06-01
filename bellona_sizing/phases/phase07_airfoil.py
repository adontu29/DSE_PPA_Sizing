"""Phase 7: optional XFOIL wrapper and fallback airfoil table."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np


def _parse_xfoil_polar(polar_path: Path) -> Dict[str, np.ndarray]:
    """Read the numeric part of an XFOIL polar file."""
    rows = []
    with open(polar_path, "r", encoding="utf-8", errors="ignore") as polar_file:
        for line in polar_file:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                values = [float(value) for value in parts[:5]]
            except ValueError:
                continue
            rows.append(values)

    if not rows:
        raise ValueError("No numeric polar rows were found.")

    data = np.asarray(rows, dtype=float)
    return {
        "alpha_deg": data[:, 0],
        "cl": data[:, 1],
        "cd": data[:, 2],
        "cdp": data[:, 3],
        "cm": data[:, 4],
    }


def _summarise_airfoil_polar(polar: Dict[str, np.ndarray],
                             airfoil: str,
                             reynolds: float,
                             x_transition: float,
                             design_cl: float,
                             source: str) -> Dict:
    """Convert XFOIL polar arrays into the compact Phase 7 output shape."""
    alpha = polar["alpha_deg"]
    cl = polar["cl"]
    cd = polar["cd"]
    cm = polar["cm"]

    finite = np.isfinite(alpha) & np.isfinite(cl) & np.isfinite(cd) & np.isfinite(cm) & (cd > 0.0)
    alpha = alpha[finite]
    cl = cl[finite]
    cd = cd[finite]
    cm = cm[finite]
    if alpha.size < 3:
        raise ValueError("Too few converged polar points to summarize.")

    linear_mask = (alpha >= -2.0) & (alpha <= 6.0)
    if np.count_nonzero(linear_mask) < 3:
        linear_mask = np.ones_like(alpha, dtype=bool)
    slope_per_deg, intercept = np.polyfit(alpha[linear_mask], cl[linear_mask], 1)
    cl_a = slope_per_deg * 180.0 / np.pi

    idx_cl0 = int(np.argmin(np.abs(cl)))
    idx_design = int(np.argmin(np.abs(cl - design_cl)))
    idx_clmax = int(np.argmax(cl))
    cl_cd = np.where(cd > 0.0, cl / cd, np.nan)
    idx_best_ld = int(np.nanargmax(cl_cd))

    stall_char = "xfoil polar; inspect post-stall behavior before final use"
    if idx_clmax < alpha.size - 2 and cl[idx_clmax + 1] < cl[idx_clmax] - 0.05:
        stall_char = "xfoil polar suggests a defined peak; verify stall convergence"

    return {
        "airfoil": airfoil,
        "Re": float(reynolds),
        "x_transition": float(x_transition),
        "cl_a": float(cl_a),
        "cl_alpha_per_rad": float(cl_a),
        "cl_max": float(cl[idx_clmax]),
        "alpha_stall_deg": float(alpha[idx_clmax]),
        "cd0": float(cd[idx_cl0]),
        "cd_at_design": float(cd[idx_design]),
        "cl_design": float(design_cl),
        "cl_cd_at_design": float(cl[idx_design] / cd[idx_design]),
        "cl_cd_max": float(cl_cd[idx_best_ld]),
        "cm0": float(cm[idx_cl0]),
        "cm_at_design": float(cm[idx_design]),
        "stall_char": stall_char,
        "source": source,
        "polar_points": int(alpha.size),
    }


def _run_xfoil_airfoil(xfoil_exe: str,
                       airfoil: str,
                       reynolds: float,
                       x_transition: float,
                       design_cl: float,
                       coordinate_file: Optional[str] = None,
                       alpha_start: float = -4.0,
                       alpha_end: float = 14.0,
                       alpha_step: float = 1.0,
                       timeout_s: float = 30.0) -> Tuple[Optional[Dict], Optional[str]]:
    """Run XFOIL for one airfoil and return (result, warning)."""
    if reynolds <= 0.0:
        raise ValueError("reynolds must be positive.")

    try:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp_dir = Path(tmp_name)
            polar_path = tmp_dir / "polar.txt"

            if coordinate_file:
                source_path = Path(coordinate_file)
                if not source_path.exists():
                    return None, f"Coordinate file not found for {airfoil}: {coordinate_file}"
                local_coord = tmp_dir / source_path.name
                shutil.copy2(source_path, local_coord)
                airfoil_command = f"LOAD {local_coord.name}"
            elif airfoil.lower().replace(" ", "") == "naca0012":
                airfoil_command = "NACA 0012"
            else:
                return None, f"No coordinate file was provided for {airfoil}."

            commands = "\n".join([
                "PLOP",
                "G F",
                "",
                airfoil_command,
                "PANE",
                "OPER",
                f"VISC {reynolds:.0f}",
                "ITER 150",
                "VPAR",
                f"XTR {x_transition:.3f} {x_transition:.3f}",
                "",
                "PACC",
                str(polar_path),
                "",
                f"ASEQ {alpha_start:.2f} {alpha_end:.2f} {alpha_step:.2f}",
                "PACC",
                "",
                "QUIT",
                "",
            ])

            completed = subprocess.run(
                [xfoil_exe],
                input=commands,
                text=True,
                capture_output=True,
                cwd=tmp_dir,
                timeout=timeout_s,
            )
            if not polar_path.exists():
                tail = completed.stderr.strip() or completed.stdout.strip()[-300:]
                return None, f"XFOIL did not produce a polar for {airfoil}. {tail}"

            polar = _parse_xfoil_polar(polar_path)
            return _summarise_airfoil_polar(
                polar,
                airfoil,
                reynolds,
                x_transition,
                design_cl,
                "xfoil",
            ), None
    except subprocess.TimeoutExpired:
        return None, f"XFOIL timed out while analyzing {airfoil}."
    except OSError as exc:
        return None, f"XFOIL could not be started for {airfoil}: {exc}"
    except ValueError as exc:
        return None, f"XFOIL polar could not be parsed for {airfoil}: {exc}"


def _fallback_airfoil_result(role: str, reynolds: float, x_transition: float) -> Dict:
    """Project-table fallback values for SD7037 and NACA 0012."""
    reference_re = 788199.3377311177
    if role == "main":
        data = {
            "airfoil": "SD7037",
            "cl_design": 0.60,
            "cl_max": 1.4257,
            "cd_at_design": 0.00724,
            "cl_cd_at_design": 81.51933701657458,
            "cm0": -0.07579230769230769,
            "cm_at_design": -0.0760173144876325,
            "alpha_stall_deg": 12.0,
            "cl_a": 5.90,
            "stall_char": "low-Re airfoil; verify stall softness with measured or XFOIL polar data",
        }
    elif role == "canard":
        data = {
            "airfoil": "NACA 0012",
            "cl_design": 0.30,
            "cl_max": 1.3177,
            "cd_at_design": 0.00775,
            "cl_cd_at_design": 37.2,
            "cm0": 0.0,
            "cm_at_design": -0.0005575317604355717,
            "alpha_stall_deg": 12.0,
            "cl_a": 6.10,
            "stall_char": "symmetric reference section; verify control-surface performance with hinge and deflection data",
        }
    else:
        raise ValueError("role must be 'main' or 'canard'.")

    reynolds_ratio = reynolds / reference_re
    warnings = []
    if reynolds_ratio < 0.75 or reynolds_ratio > 1.25:
        warnings.append(
            "Requested Reynolds number differs substantially from the project-table reference value."
        )

    result = {
        "airfoil": data["airfoil"],
        "Re": float(reynolds),
        "Re_reference": float(reference_re),
        "x_transition": float(x_transition),
        "cl_a": float(data["cl_a"]),
        "cl_alpha_per_rad": float(data["cl_a"]),
        "cl_max": float(data["cl_max"]),
        "alpha_stall_deg": float(data["alpha_stall_deg"]),
        "cd0": float(data["cd_at_design"]),
        "cd_at_design": float(data["cd_at_design"]),
        "cl_design": float(data["cl_design"]),
        "cl_cd_at_design": float(data["cl_cd_at_design"]),
        "cl_cd_max": float(data["cl_cd_at_design"]),
        "cm0": float(data["cm0"]),
        "cm_at_design": float(data["cm_at_design"]),
        "stall_char": data["stall_char"],
        "source": "fallback_project_xfoil_table",
        "warnings": warnings,
    }
    return result


def phase7_airfoil_xfoil(Re_main: float, Re_canard: float,
                         x_transition: float = 0.50,
                         use_xfoil: bool = False,
                         xfoil_path: Optional[str] = None,
                         airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """XFOIL wrapper: evaluate SD7037 + NACA 0012 + candidates.
    In:  Re_main, Re_canard (Ph3+Ph8 estimate), forced transition x/c (Selig 2003).
    Out: {'main': {'cl_a', 'cl_max', 'cd0', 'cm0', 'stall_char'},
          'canard': {...}}
    Ref: Drela XFOIL User Guide; Selig 2003 VKI LRN lecture notes.
         SD7037 Cl/Cd_max  75 @ Re=2e5,  99 @ Re=5e5 (UIUC database).
    Loop: Re inner loop with Ph8."""
    if Re_main <= 0.0:
        raise ValueError("Re_main must be positive.")
    if Re_canard <= 0.0:
        raise ValueError("Re_canard must be positive.")
    if not 0.0 < x_transition < 1.0:
        raise ValueError("x_transition should be between 0 and 1.")

    airfoil_files = {
        str(key): str(value)
        for key, value in (airfoil_files or {}).items()
    }
    warnings = []
    notes = [
        "Fallback values are anchored to the project XFOIL comparison table at Re=7.88e5 and x/c=0.5.",
        "The fallback cd0 value is a profile-drag proxy taken from cd at the design lift coefficient.",
        "XFOIL near stall can be convergence-sensitive; measured polars or wind-tunnel data should replace these values when available.",
    ]

    xfoil_exe = None
    if xfoil_path:
        xfoil_exe = xfoil_path
        use_xfoil = True
    elif use_xfoil:
        xfoil_exe = shutil.which("xfoil") or shutil.which("xfoil.exe")

    xfoil_used = False
    main = None
    canard = None

    if use_xfoil and xfoil_exe:
        main, warning = _run_xfoil_airfoil(
            xfoil_exe,
            "SD7037",
            Re_main,
            x_transition,
            0.60,
            coordinate_file=airfoil_files.get("main") or airfoil_files.get("sd7037"),
        )
        if warning:
            warnings.append(warning)

        canard, warning = _run_xfoil_airfoil(
            xfoil_exe,
            "NACA 0012",
            Re_canard,
            x_transition,
            0.30,
            coordinate_file=airfoil_files.get("canard") or airfoil_files.get("naca0012"),
        )
        if warning:
            warnings.append(warning)

        xfoil_used = main is not None or canard is not None
    elif use_xfoil and not xfoil_exe:
        warnings.append("XFOIL was requested, but no executable was found on PATH and no xfoil_path was provided.")

    if main is None:
        main = _fallback_airfoil_result("main", Re_main, x_transition)
        warnings.extend(f"main: {warning}" for warning in main.get("warnings", []))
    if canard is None:
        canard = _fallback_airfoil_result("canard", Re_canard, x_transition)
        warnings.extend(f"canard: {warning}" for warning in canard.get("warnings", []))

    if abs(x_transition - 0.50) > 0.05 and (main["source"].startswith("fallback") or canard["source"].startswith("fallback")):
        warnings.append(
            "Fallback values were generated for x/c=0.5 forced transition; rerun XFOIL for other transition locations."
        )

    return {
        "main": main,
        "canard": canard,
        "xfoil": {
            "requested": bool(use_xfoil),
            "used": bool(xfoil_used),
            "executable": xfoil_exe,
            "airfoil_files": dict(airfoil_files),
        },
        "notes": notes,
        "warnings": warnings,
    }
