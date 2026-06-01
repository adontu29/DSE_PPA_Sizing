"""Phase 16: damped MTOW convergence update."""
from __future__ import annotations

from typing import Dict


def phase16_mtow_converge(MTOW_old: float, MTOW_new: float,
                          damping: float = 0.4, tol: float = 0.01
                          ) -> Dict:
    """Outer-loop MTOW convergence with damping (Gundlach 2014 Ch.19).
    Returns a dictionary so the iteration history is easy to inspect."""
    if MTOW_old <= 0.0:
        raise ValueError("MTOW_old must be positive.")
    if MTOW_new <= 0.0:
        raise ValueError("MTOW_new must be positive.")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in the range 0 < damping <= 1.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    delta = MTOW_new - MTOW_old
    MTOW_next = MTOW_old + damping * delta
    relative_error = abs(delta) / MTOW_old
    converged = relative_error < tol
    return {
        "MTOW_old_kg": float(MTOW_old),
        "MTOW_new_kg": float(MTOW_new),
        "MTOW_next_kg": float(MTOW_next),
        "delta_kg": float(delta),
        "relative_error": float(relative_error),
        "damping": float(damping),
        "tol": float(tol),
        "converged": bool(converged),
        "notes": [
            "MTOW_next = MTOW_old + damping*(MTOW_new - MTOW_old).",
            "Convergence is based on abs(MTOW_new - MTOW_old)/MTOW_old < tol.",
        ],
        "warnings": [],
    }
