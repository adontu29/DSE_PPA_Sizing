"""ISA atmosphere model."""

from __future__ import annotations

from functools import lru_cache

from sizing.inputs import AIRCRAFT


@lru_cache(maxsize=4096)
def isa_density(altitude_m):
    """Troposphere ISA density [kg/m^3].

    Memoized: the climb integration evaluates it (and dV/dH, which calls it twice
    more) on a fixed altitude grid that repeats across every power/EAS candidate
    and mass iteration, so the same handful of altitudes recur hundreds of
    thousands of times. g is constant for a run, so caching on altitude is exact.
    """
    rho0 = 1.225
    temperature0 = 288.15
    lapse = -0.0065
    gas_constant = 287.05
    gravity = AIRCRAFT["g_m_s2"]
    temperature = temperature0 + lapse * altitude_m
    pressure_ratio = (temperature / temperature0) ** (-gravity / (lapse * gas_constant))
    return rho0 * pressure_ratio * temperature0 / temperature
