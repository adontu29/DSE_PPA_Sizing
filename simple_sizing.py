"""Bellona simplified sizing -- entry point.

The sizing method lives in the ``sizing`` package, one module per physical step
(see ``sizing/__init__.py``). This file is just the runnable entry point and a
small compatibility surface (``AIRCRAFT``, ``MISSION``, ``MASS``, ``run_sizing``)
for scripts and tests that import ``simple_sizing`` directly.

Run from the repository root:

    python simple_sizing.py
"""

from __future__ import annotations

from sizing.inputs import AIRCRAFT, MISSION, MASS
from sizing.workflow import OUTPUT_DIR, run_sizing, main

__all__ = ["AIRCRAFT", "MISSION", "MASS", "OUTPUT_DIR", "run_sizing", "main"]


if __name__ == "__main__":
    main()
