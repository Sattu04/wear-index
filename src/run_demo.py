"""
End-to-end: trips in, wear verdicts out.

    python src/run_demo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import pandas as pd

from obd_features import driver_profile
from synth import make_fleet
from wear_index import assess, explain, fleet_stats


def main() -> None:
    print("building synthetic fleet (SYNTHETIC -- baseline only, not a finding)\n")
    fleet = make_fleet(n_drivers=40, trips_per_driver=12)

    profiles = [driver_profile(d["trips"]) for d in fleet]
    stats = fleet_stats(profiles)

    print(f"fleet: {len(profiles)} drivers, "
          f"{sum(p['n_trips'] for p in profiles):,} trips, "
          f"{sum(p['total_km'] for p in profiles):,.0f} km\n")

    # Report on one driver from each archetype.
    seen = set()
    for driver, profile in zip(fleet, profiles):
        if driver["archetype"] in seen:
            continue
        seen.add(driver["archetype"])

        print("=" * 72)
        print(f"{driver['driver_id']}  ({driver['archetype']})  "
              f"{profile['total_km']:,.0f} km logged over {profile['n_trips']} trips")
        print("=" * 72)

        km_since = {"brakes": 21_000, "engine": 84_000, "cooling": 84_000,
                    "fuel_air": 61_000, "transmission": 84_000, "battery": 38_000}
        for verdict in assess(profile, stats, km_since)[:3]:
            print(explain(verdict))
            print(f"  -> {verdict['remaining_km']:,.0f} km remaining before service\n")

    rows = []
    for driver, profile in zip(fleet, profiles):
        top = assess(profile, stats)[0]
        rows.append({
            "driver": driver["driver_id"],
            "archetype": driver["archetype"],
            "first_to_wear": top["label"],
            "wear_rate_vs_fleet": round(top["multiplier"], 2),
            "effective_life_km": round(top["effective_life_km"]),
        })
    out = pathlib.Path(__file__).parents[1] / "data" / "fleet_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"fleet summary written to {out}")


if __name__ == "__main__":
    main()
