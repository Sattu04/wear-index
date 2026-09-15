"""
Loader for the University of Michigan Vehicle Energy Dataset (VED).

VED is ~380 vehicles driving around Ann Arbor, logged from the OBD-II port over
a year. It is the closest public thing to what a SPARQ device sees.

Getting it:
    https://github.com/gsoh/VED  ->  Data/VED_DynamicData_Part1.7z (and Part2)
    7z x VED_DynamicData_Part1.7z -o data/ved/

WHAT VED DOES NOT HAVE, AND WHY IT MATTERS
------------------------------------------
VED logs speed, RPM, absolute load, MAF, fuel rate, outside air temperature and
fuel trims. It does not log engine coolant temperature or throttle position.

That kills four of the ten stressors outright:

    coolant temperature  ->  thermal_excursions_per_100km   (cooling system)
                         ->  cold_high_load_events          (engine internals)
                         ->  short_trip_frac                (battery, fuel/air)
    throttle position    ->  brake_dwell_frac               (brakes)

Those four are not marginal. They are the ones that carry the clearest wear
mechanism, and brake wear -- the thing a driver will actually pay to hear about
-- depends entirely on the one inference I cannot make without throttle.

Both PIDs are standard OBD-II Mode 01 (0x05 and 0x11). Any device sitting in the
port already reads them. So the gap here is a gap in public data, not in the
method, and it closes the moment the method runs on logs from real hardware.

This loader therefore runs in degraded mode on VED and says so loudly, rather
than silently imputing the missing channels and reporting a number that looks
complete.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DEGRADED = [
    "thermal_excursions_per_100km",
    "cold_high_load_events",
    "short_trip_frac",
    "brake_dwell_frac",
]

COLUMN_MAP = {
    "Vehicle Speed[km/h]": "speed_kph",
    "Engine RPM[RPM]": "rpm",
    "Absolute Load[%]": "engine_load_pct",
    "OAT[DegC]": "ambient_c",
    "Fuel Rate[L/hr]": "fuel_rate_lph",
    "Short Term Fuel Trim Bank 1[%]": "stft_pct",
}


def load_trips(ved_dir: str | pathlib.Path, max_vehicles: int | None = 40):
    """Yield (vehicle_id, [trip DataFrames]) in the schema obd_features expects."""
    ved_dir = pathlib.Path(ved_dir)
    files = sorted(ved_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSVs under {ved_dir}. Extract VED_DynamicData_Part1.7z there first."
        )

    frames = [pd.read_csv(f, low_memory=False) for f in files]
    data = pd.concat(frames, ignore_index=True)

    present = {k: v for k, v in COLUMN_MAP.items() if k in data.columns}
    missing = set(COLUMN_MAP) - set(present)
    if missing:
        print(f"[ved] columns absent from this extract: {sorted(missing)}")

    data = data.rename(columns=present)
    data["t_s"] = data["Timestamp(ms)"] / 1000.0

    # Channels VED does not carry. Filled with neutral constants so the pipeline
    # runs, which is exactly why the affected stressors must be discarded.
    data["coolant_c"] = np.nan
    data["throttle_pct"] = np.nan

    print(f"[ved] DEGRADED MODE -- these stressors are not computable: {DEGRADED}")

    vehicles = data["VehId"].unique()
    if max_vehicles:
        vehicles = vehicles[:max_vehicles]

    for vid in vehicles:
        block = data[data["VehId"] == vid]
        trips = [
            trip.sort_values("t_s").reset_index(drop=True)
            for _, trip in block.groupby("Trip")
            if len(trip) > 120
        ]
        if trips:
            yield str(vid), trips


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/ved"
    for vid, trips in load_trips(target, max_vehicles=3):
        total = sum(len(t) for t in trips)
        print(f"vehicle {vid}: {len(trips)} trips, {total:,} samples")
