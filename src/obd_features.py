"""
Turn raw OBD-II trip logs into wear stressors.

Every feature here has to answer one question before it earns a place: what
physically wears out because of it? A feature that correlates but has no
mechanism behind it is how you end up telling a driver their brakes are fine
because they happen to drive on Tuesdays.

Expected trip schema (one row per sample, ~1 Hz):
    t_s, speed_kph, rpm, engine_load_pct, coolant_c, throttle_pct,
    stft_pct, fuel_rate_lph, ambient_c

Note on brakes: standard OBD-II has no brake pedal PID. Brake use is inferred
from deceleration with a closed throttle. That inference is the weakest link in
this whole file and is flagged as such wherever it is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

G = 9.80665


@dataclass(frozen=True)
class Stressor:
    key: str
    label: str
    mechanism: str
    units: str


STRESSORS: dict[str, Stressor] = {
    s.key: s
    for s in [
        Stressor(
            "brake_dwell_frac",
            "time resting on the brake",
            "light continuous pad contact keeps the pad above its designed operating "
            "temperature, glazing the friction material and scoring the rotor",
            "fraction of moving time",
        ),
        Stressor(
            "harsh_decel_per_100km",
            "hard stops",
            "each hard stop dumps kinetic energy into the pads as heat; repeated "
            "thermal cycling cracks the friction layer and warps the rotor",
            "events / 100 km",
        ),
        Stressor(
            "cold_high_load_events",
            "hard driving before warm-up",
            "cold oil is thick and slow to reach the top end, so bearings and bores "
            "run partly dry under exactly the loads that need the film most",
            "events / 100 km",
        ),
        Stressor(
            "short_trip_frac",
            "trips too short to warm up",
            "an engine below operating temperature condenses water and fuel into the "
            "oil, and the alternator never recovers the charge the starter drew",
            "fraction of trips",
        ),
        Stressor(
            "lugging_frac",
            "high load at low revs",
            "high cylinder pressure at low piston speed promotes pre-ignition and "
            "loads the crank and gearbox in the band they are least happy in",
            "fraction of moving time",
        ),
        Stressor(
            "high_rpm_frac",
            "sustained high revs",
            "valve train and bearing wear scale steeply with engine speed",
            "fraction of moving time",
        ),
        Stressor(
            "idle_frac",
            "idling",
            "no airflow through the radiator and no oil pressure headroom, while the "
            "hour meter runs anyway",
            "fraction of engine-on time",
        ),
        Stressor(
            "thermal_excursions_per_100km",
            "coolant temperature spikes",
            "every excursion past the thermostat's design point ages the hoses, the "
            "head gasket and the oil itself",
            "events / 100 km",
        ),
        Stressor(
            "stft_abs_mean",
            "fuel trim drift",
            "the ECU correcting hard on average means it is compensating for "
            "something: a vacuum leak, fouled injectors, or a lazy O2 sensor",
            "mean |short-term trim| %",
        ),
        Stressor(
            "stop_start_per_km",
            "stop-start density",
            "starter, battery, brakes and transmission all wear per event rather "
            "than per kilometre",
            "stops / km",
        ),
    ]
}


def _accel_g(df: pd.DataFrame) -> pd.Series:
    dt = df["t_s"].diff().replace(0, np.nan)
    dv = (df["speed_kph"] / 3.6).diff()
    return (dv / dt / G).fillna(0.0)


def trip_features(trip: pd.DataFrame) -> dict[str, float]:
    """Reduce one trip to a dict of stressors plus its distance."""
    df = trip.sort_values("t_s").reset_index(drop=True)
    dt = df["t_s"].diff().fillna(0.0)
    dist_km = float(((df["speed_kph"] / 3.6) * dt).sum() / 1000.0)
    moving = df["speed_kph"] > 3
    engine_on = df["rpm"] > 300
    accel = _accel_g(df)

    per_100km = lambda n: float(n) / max(dist_km, 0.1) * 100.0
    frac = lambda mask, base: float(mask.sum()) / max(int(base.sum()), 1)

    closed_throttle = df["throttle_pct"] < 5
    riding = moving & closed_throttle & accel.between(-0.15, -0.02)

    hard_stops = (accel < -0.35) & moving
    warm = df["coolant_c"] > 70
    cold_load = (df["coolant_c"] < 50) & (df["engine_load_pct"] > 60) & engine_on

    stops = int(((df["speed_kph"] <= 1) & (df["speed_kph"].shift() > 1)).sum())

    return {
        "distance_km": dist_km,
        "reached_operating_temp": bool(warm.any()),
        "brake_dwell_frac": frac(riding, moving),
        "harsh_decel_per_100km": per_100km(hard_stops.sum()),
        "cold_high_load_events": per_100km(cold_load.sum()),
        "lugging_frac": frac((df["engine_load_pct"] > 70) & (df["rpm"] < 1500) & moving, moving),
        "high_rpm_frac": frac(df["rpm"] > 4000, moving),
        "idle_frac": frac((df["speed_kph"] <= 1) & engine_on, engine_on),
        "thermal_excursions_per_100km": per_100km(((df["coolant_c"] > 105) & engine_on).sum()),
        "stft_abs_mean": float(df.loc[engine_on, "stft_pct"].abs().mean()),
        "stop_start_per_km": float(stops) / max(dist_km, 0.1),
    }


def driver_profile(trips: list[pd.DataFrame]) -> dict[str, float]:
    """Aggregate a driver's trips into one distance-weighted stressor profile."""
    rows = [trip_features(t) for t in trips]
    frame = pd.DataFrame(rows)
    weights = frame["distance_km"].clip(lower=0.1)

    profile = {
        key: float(np.average(frame[key], weights=weights))
        for key in STRESSORS
        if key != "short_trip_frac"
    }
    profile["short_trip_frac"] = float((~frame["reached_operating_temp"]).mean())
    profile["total_km"] = float(frame["distance_km"].sum())
    profile["n_trips"] = int(len(frame))
    return profile
