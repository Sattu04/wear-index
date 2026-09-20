"""
The wear index.

This is deliberately not a failure predictor. It is an exposure model: it says
how much harder than a median driver this car is being worked, per subsystem,
and converts that into a revised service interval.

The distinction matters. "Your pads fail in 3,000 km" is a claim I cannot make
without failure labels nobody has published. "You are wearing pads 2.1x faster
than the fleet, here are the two behaviours doing it, so your 55,000 km interval
is really about 26,000" is a claim that follows from the data I do have.

Every number on the way out can be traced back to a stressor and a mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from obd_features import STRESSORS

# Nominal design life under median use, in km. These are ordinary service-manual
# figures, not model outputs -- the model only moves them.
@dataclass(frozen=True)
class Subsystem:
    key: str
    label: str
    nominal_km: int
    weights: dict[str, float]


SUBSYSTEMS = [
    Subsystem(
        "brakes",
        "Front brake pads",
        55_000,
        {"brake_dwell_frac": 0.45, "harsh_decel_per_100km": 0.40, "stop_start_per_km": 0.15},
    ),
    Subsystem(
        "engine",
        "Engine internals",
        250_000,
        {
            "cold_high_load_events": 0.40,
            "short_trip_frac": 0.25,
            "lugging_frac": 0.20,
            "high_rpm_frac": 0.15,
        },
    ),
    Subsystem(
        "cooling",
        "Cooling system",
        160_000,
        {"thermal_excursions_per_100km": 0.55, "idle_frac": 0.25, "lugging_frac": 0.20},
    ),
    Subsystem(
        "fuel_air",
        "Fuel and air metering",
        120_000,
        {"stft_abs_mean": 0.65, "short_trip_frac": 0.35},
    ),
    Subsystem(
        "transmission",
        "Transmission",
        200_000,
        {"lugging_frac": 0.45, "stop_start_per_km": 0.30, "harsh_decel_per_100km": 0.25},
    ),
    Subsystem(
        "battery",
        "Starter battery",
        70_000,
        {"short_trip_frac": 0.60, "stop_start_per_km": 0.40},
    ),
]

# How strongly one robust standard deviation of a stressor multiplies wear.
#
# These are engineering priors, not fitted values, and that is the single
# biggest limitation in this repo. They are set per stressor rather than as one
# global constant because the underlying mechanisms genuinely differ in how
# sharply they respond: pad glazing is close to a step change once the friction
# surface goes, whereas idling ages a cooling system slowly and forgivingly.
#
# Given real warranty or parts-replacement data, these ten numbers are the first
# thing to fit and the first thing I would expect to be wrong.
SENSITIVITY = {
    "thermal_excursions_per_100km": 0.60,  # overheating is the least forgiving thing here
    "brake_dwell_frac": 0.55,              # glazing is near a step change
    "harsh_decel_per_100km": 0.50,         # pad energy scales with v^2
    "cold_high_load_events": 0.45,         # boundary lubrication, not hydrodynamic
    "short_trip_frac": 0.40,
    "lugging_frac": 0.35,
    "high_rpm_frac": 0.30,
    "stft_abs_mean": 0.30,
    "stop_start_per_km": 0.30,
    "idle_frac": 0.25,
}
Z_FLOOR, Z_CEIL = -1.5, 3.0


def fleet_stats(profiles: list[dict]) -> dict[str, tuple[float, float]]:
    """Median and spread per stressor. Median absolute deviation, because a
    handful of extreme drivers should not define what 'normal' means."""
    stats = {}
    for key in STRESSORS:
        vals = np.array([p[key] for p in profiles], dtype=float)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826
        stats[key] = (med, mad if mad > 1e-9 else max(float(vals.std()), 1e-6))
    return stats


def _z(value: float, stat: tuple[float, float]) -> float:
    med, spread = stat
    return float(np.clip((value - med) / spread, Z_FLOOR, Z_CEIL))


def assess(
    profile: dict,
    stats: dict,
    km_since_service: dict | None = None,
    available: set[str] | None = None,
) -> list[dict]:
    """Return one verdict per subsystem, with its reasons attached.

    `available` restricts the model to the stressors a given data source can
    actually compute. Weights for the survivors are renormalised, which is the
    fairest possible treatment of a degraded sensor set: the model does the best
    it can with what it has rather than silently scoring the gap as zero.
    """
    km_since_service = km_since_service or {}
    out = []

    for sub in SUBSYSTEMS:
        weights = sub.weights
        if available is not None:
            weights = {k: w for k, w in weights.items() if k in available}
            if not weights:
                continue
            total = sum(weights.values())
            weights = {k: w / total for k, w in weights.items()}

        contributions = []
        log_multiplier = 0.0
        for key, weight in weights.items():
            z = _z(profile[key], stats[key])
            term = weight * SENSITIVITY[key] * z
            log_multiplier += term
            contributions.append(
                {
                    "stressor": key,
                    "label": STRESSORS[key].label,
                    "mechanism": STRESSORS[key].mechanism,
                    "units": STRESSORS[key].units,
                    "value": profile[key],
                    "fleet_median": stats[key][0],
                    "z": z,
                    "effect": float(np.exp(term)),
                }
            )

        multiplier = float(np.exp(log_multiplier))
        effective_life = sub.nominal_km / multiplier
        used = float(km_since_service.get(sub.key, 0.0))
        remaining = max(effective_life - used, 0.0)

        contributions.sort(key=lambda c: -abs(c["z"] * weights[c["stressor"]]))
        out.append(
            {
                "key": sub.key,
                "label": sub.label,
                "nominal_km": sub.nominal_km,
                "multiplier": multiplier,
                "effective_life_km": effective_life,
                "km_since_service": used,
                "remaining_km": remaining,
                "reasons": contributions,
            }
        )

    out.sort(key=lambda r: r["remaining_km"])
    return out


def explain(verdict: dict, top_n: int = 2) -> str:
    """One paragraph a driver would actually read."""
    rate = verdict["multiplier"]
    if rate >= 1.15:
        pace = f"{rate:.1f}x faster than the fleet median"
    elif rate <= 0.87:
        pace = f"{1 / rate:.1f}x slower than the fleet median"
    else:
        pace = "at about the fleet median pace"

    lines = [
        f"{verdict['label']}: wearing {pace}. "
        f"Book-interval {verdict['nominal_km']:,} km becomes "
        f"{verdict['effective_life_km']:,.0f} km for this driver."
    ]
    for reason in verdict["reasons"][:top_n]:
        if abs(reason["z"]) < 0.25:
            continue
        direction = "above" if reason["z"] > 0 else "below"
        lines.append(
            f"  - {reason['label']}: {reason['value']:.3g} {reason['units']} "
            f"({direction} the fleet median of {reason['fleet_median']:.3g}). "
            f"Why it matters: {reason['mechanism']}."
        )
    return "\n".join(lines)