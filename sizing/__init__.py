"""Bellona simplified aircraft-sizing package.

The modules follow the physical sizing method, in order:

    inputs       mission, aircraft, and mass-model assumptions (the only knobs)
    atmosphere   ISA density
    geometry     wing / canard planforms and the rotor disc
    transition   reduced-order tail-sitter transition simulation -> stall cap
    mass         component mass build-up and CG
    scissor      canard sizing and the static-stability / control CG band
    mission      course-method climb energy and battery sizing
    airfoil      optional XFOIL Reynolds-feedback refinement of section data
    loop         the mass / wing-area sizing loops
    report       concise tables and report figures
    workflow     run_sizing() and main() -- the top-level method spine

The single entry point is ``simple_sizing.py`` at the repository root, which
just calls ``sizing.workflow.main``.
"""
