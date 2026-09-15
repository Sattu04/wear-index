"""
Synthetic trip generator.

THIS IS NOT REAL DATA. It exists so the pipeline runs end to end in thirty
seconds without a 180 MB download, and so the fleet baseline has something to be
a baseline of. Nothing in here should ever be quoted as a finding.

The real loader is src/load_ved.py, which reads the University of Michigan
Vehicle Energy Dataset. Both produce the same schema, so swapping one for the
other changes no downstream code.

The trips are generated from a crude drive-cycle model with per-driver
behavioural knobs, so the stressors that come out are at least self-consistent:
a driver who rides the brakes really does show sustained low-deceleration events
with a closed throttle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ARCHETYPES = {
    "commuter": dict(brake_rider=0.05, aggression=0.35, trip_min=(18, 40), cold_start_load=0.25),
    "city_short_hop": dict(brake_rider=0.22, aggression=0.55, trip_min=(4, 11), cold_start_load=0.55),
    "highway": dict(brake_rider=0.02, aggression=0.30, trip_min=(45, 95), cold_start_load=0.15),
    "impatient": dict(brake_rider=0.30, aggression=0.85, trip_min=(10, 30), cold_start_load=0.75),
    "gentle": dict(brake_rider=0.03, aggression=0.15, trip_min=(15, 45), cold_start_load=0.10),
}


def _drive_cycle(rng, minutes: float, aggression: float, brake_rider: float):
    """Speed trace in km/h at 1 Hz, plus a mask of samples where the driver is
    resting a foot on the brake.

    Brake riding is modelled where it actually happens: during cruise, as a slow
    unexplained sag in speed with the throttle closed. Modelling it inside the
    deceleration ramp would be wrong -- everyone decelerates."""
    n = int(minutes * 60)
    speed = np.zeros(n)
    riding = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        cruise = rng.uniform(25, 110) * (0.8 + 0.4 * aggression)
        hold = int(rng.uniform(40, 260))
        ramp = max(int(cruise / (2.5 + 6 * aggression)), 3)
        end = min(i + ramp, n)
        speed[i:end] = np.linspace(speed[i - 1] if i else 0, cruise, end - i)
        i = end

        end = min(i + hold, n)
        seg = max(end - i, 0)
        if seg:
            if rng.random() < brake_rider * 3.0:
                # foot on the pedal: speed bleeds off at 0.03-0.09 g, throttle shut
                decel_g = rng.uniform(0.03, 0.09)
                drop = decel_g * 9.80665 * seg * 3.6
                speed[i:end] = np.maximum(np.linspace(cruise, cruise - drop, seg), 5)
                riding[i:end] = True
            else:
                speed[i:end] = cruise + rng.normal(0, 2.5, seg)
        i = end

        decel = max(int(cruise / (2.0 + 7 * aggression)), 3)
        end = min(i + decel, n)
        speed[i:end] = np.linspace(speed[i - 1] if i else cruise, 0, end - i)
        i = end
        i = min(i + int(rng.uniform(3, 45)), n)
    return np.clip(speed, 0, None), riding


def make_trip(rng, archetype: str) -> pd.DataFrame:
    cfg = ARCHETYPES[archetype]
    minutes = rng.uniform(*cfg["trip_min"])
    speed, riding = _drive_cycle(rng, minutes, cfg["aggression"], cfg["brake_rider"])
    n = len(speed)
    t = np.arange(n, dtype=float)

    accel = np.gradient(speed / 3.6)
    moving = speed > 3

    # Engine speed: gearing that shifts up with road speed, plus idle.
    gear = np.clip(np.ceil(speed / 22), 1, 6)
    rpm = np.where(moving, 800 + (speed / 3.6) / (gear * 0.0042), 750 + rng.normal(0, 25, n))
    rpm = np.clip(rpm + rng.normal(0, 60, n), 600, 6500)

    load = np.clip(18 + accel * 55 + speed * 0.18 + rng.normal(0, 6, n), 2, 100)

    # Coolant: first-order warm-up toward 90 C, faster under load.
    coolant = np.empty(n)
    coolant[0] = rng.uniform(-2, 28)
    tau = 430.0
    for k in range(1, n):
        target = 90 + (load[k] - 40) * 0.08
        coolant[k] = coolant[k - 1] + (target - coolant[k - 1]) / tau * (1 + load[k] / 90)
    coolant += rng.normal(0, 0.4, n)

    throttle = np.clip(load * 0.9 + rng.normal(0, 4, n), 0, 100)
    # Closed throttle wherever the foot is on the brake instead.
    throttle[riding] = rng.uniform(0, 3, int(riding.sum()))
    load[riding] = np.clip(load[riding] * 0.25, 2, 100)

    # Cold-start abuse.
    cold = coolant < 50
    hot_foot = cold & (rng.random(n) < cfg["cold_start_load"] * 0.25)
    load[hot_foot] = np.clip(load[hot_foot] + rng.uniform(25, 50, hot_foot.sum()), 2, 100)

    stft = rng.normal(0, 1.6, n) + rng.normal(0, 0.8)
    fuel = np.clip(load * 0.09 + rpm * 0.0009 + rng.normal(0, 0.25, n), 0.3, None)

    return pd.DataFrame(
        {
            "t_s": t,
            "speed_kph": speed,
            "rpm": rpm,
            "engine_load_pct": load,
            "coolant_c": coolant,
            "throttle_pct": throttle,
            "stft_pct": stft,
            "fuel_rate_lph": fuel,
            "ambient_c": 18.0,
        }
    )


def make_fleet(n_drivers: int = 40, trips_per_driver: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    names = list(ARCHETYPES)
    fleet = []
    for d in range(n_drivers):
        archetype = names[d % len(names)]
        trips = [make_trip(rng, archetype) for _ in range(trips_per_driver)]
        fleet.append({"driver_id": f"D{d:03d}", "archetype": archetype, "trips": trips})
    return fleet
