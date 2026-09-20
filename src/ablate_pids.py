"""
What are two OBD-II PIDs actually worth?

Public driving datasets log speed, RPM, load, fuel rate and fuel trims. They do
not log engine coolant temperature (Mode 01 PID 0x05) or throttle position
(0x11). Both are standard, and any device sitting in the port already reads
them.

Losing those two channels removes four of the ten stressors:

    coolant  -> thermal_excursions_per_100km, cold_high_load_events, short_trip_frac
    throttle -> brake_dwell_frac

The obvious question is how much that costs. Not in the abstract -- in the only
output a driver ever sees, which is "which part of my car should I worry about
first."

So: score every driver twice, once on all ten stressors and once on the six that
survive, and count how often the answer changes.

Method note: the degraded run renormalises the surviving weights rather than
scoring the missing ones as zero. That is deliberately generous to the degraded
model. Treating absent data as "no stress" would inflate the gap and would be a
dishonest way to make the point.

Run:  python src/ablate_pids.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd

from obd_features import STRESSORS, driver_profile
from synth import make_fleet
from wear_index import SUBSYSTEMS, assess, fleet_stats

COOLANT_ONLY = ["thermal_excursions_per_100km", "cold_high_load_events", "short_trip_frac"]
THROTTLE_ONLY = ["brake_dwell_frac"]
LOST = COOLANT_ONLY + THROTTLE_ONLY

N_DRIVERS = 200
TRIPS_EACH = 12


def top_pick(verdicts: list[dict]) -> str:
    """The one subsystem the driver is told to worry about first."""
    return min(verdicts, key=lambda v: v["effective_life_km"])["label"]


def rates(verdicts: list[dict]) -> dict[str, float]:
    return {v["key"]: v["multiplier"] for v in verdicts}


def compare(profiles, stats, available: set[str], label: str) -> dict:
    flips, rate_err, rank_shift = 0, [], []

    for profile in profiles:
        full = assess(profile, stats)
        part = assess(profile, stats, available=available)

        if top_pick(full) != top_pick(part):
            flips += 1

        a, b = rates(full), rates(part)
        shared = [k for k in a if k in b]
        rate_err += [abs(a[k] - b[k]) / a[k] for k in shared]

        order_full = [v["key"] for v in sorted(full, key=lambda v: v["effective_life_km"])]
        order_part = [v["key"] for v in sorted(part, key=lambda v: v["effective_life_km"])]
        pos = {k: i for i, k in enumerate(order_part)}
        rank_shift += [abs(i - pos[k]) for i, k in enumerate(order_full) if k in pos]

    return {
        "removed": label,
        "stressors_lost": len(STRESSORS) - len(available),
        "top_pick_changes_pct": 100.0 * flips / len(profiles),
        "mean_wear_rate_error_pct": 100.0 * float(np.mean(rate_err)),
        "mean_rank_shift": float(np.mean(rank_shift)),
    }


def main() -> None:
    print(f"fleet: {N_DRIVERS} synthetic drivers x {TRIPS_EACH} trips\n")
    fleet = make_fleet(n_drivers=N_DRIVERS, trips_per_driver=TRIPS_EACH, seed=11)
    profiles = [driver_profile(d["trips"]) for d in fleet]
    stats = fleet_stats(profiles)

    every = set(STRESSORS)
    scenarios = [
        (every - set(THROTTLE_ONLY), "throttle position (0x11)"),
        (every - set(COOLANT_ONLY), "coolant temperature (0x05)"),
        (every - set(LOST), "both PIDs"),
    ]

    results = [compare(profiles, stats, avail, name) for avail, name in scenarios]
    table = pd.DataFrame(results)

    print("Effect of removing OBD-II channels, vs the full ten-stressor model")
    print(
        table.to_string(
            index=False,
            formatters={
                "top_pick_changes_pct": "{:.1f}%".format,
                "mean_wear_rate_error_pct": "{:.1f}%".format,
                "mean_rank_shift": "{:.2f}".format,
            },
        )
    )

    both = results[-1]
    print(
        f"\nWithout both PIDs, the subsystem a driver is told to worry about first "
        f"changes for {both['top_pick_changes_pct']:.0f}% of drivers."
    )
    print(
        "Both are standard Mode 01. The gap is in public data, not in the port."
    )

    # Which subsystems take the damage.
    print("\nper-subsystem wear-rate error without both PIDs:")
    errors = {s.key: [] for s in SUBSYSTEMS}
    for profile in profiles:
        a, b = rates(assess(profile, stats)), rates(
            assess(profile, stats, available=every - set(LOST))
        )
        for k in errors:
            if k in a and k in b:
                errors[k].append(abs(a[k] - b[k]) / a[k])
    for sub in SUBSYSTEMS:
        vals = errors[sub.key]
        if vals:
            print(f"  {sub.label:<22} {100 * np.mean(vals):5.1f}%")

    out = pathlib.Path(__file__).parents[1] / "data" / "pid_ablation.csv"
    table.to_csv(out, index=False)
    print(f"\nwritten to {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
