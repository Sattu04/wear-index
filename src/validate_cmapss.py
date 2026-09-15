"""
Validate the *method* on labelled degradation data.

No public automotive dataset carries real failure labels. NASA's C-MAPSS turbofan
data does. So the question this file answers is not "can I predict car failures"
but the one that actually matters:

    Does a transparent, mechanism-traceable health index lose much accuracy
    against a black box, on data where we can score both honestly?

If the answer is "not much", then building the automotive side as a transparent
index is a free choice rather than a compromise -- and a transparent index is the
only kind a driver or a mechanic will ever act on.

Run:  python src/validate_cmapss.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "cmapss"

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle", "op1", "op2", "op3"] + SENSOR_COLS

# Sensors that are flat in FD001 carry no degradation signal. Dropping them is a
# judgement call made once, in the open, rather than buried in a feature importance.
DEAD_SENSORS = ["s1", "s5", "s10", "s16", "s18", "s19"]

HEALTHY_FRAC = 0.05  # first 5% of a unit's life is treated as "as new"
FAILED_FRAC = 0.05  # last 5% is treated as "about to fail"
RUL_CAP = 125  # standard piecewise-linear cap used in the C-MAPSS literature
EXTRAP_WINDOW = 30  # cycles of recent history used to extrapolate the trend


def load(split: str) -> pd.DataFrame:
    path = DATA / f"{split}_FD001.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, : len(COLS)]
    df.columns = COLS
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    last = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = last - df["cycle"]
    return df


def live_sensors() -> list[str]:
    return [c for c in SENSOR_COLS if c not in DEAD_SENSORS]


def fit_health_index(train: pd.DataFrame, cols: list[str]):
    """Build a health index from the two ends of life only.

    Every unit starts healthy and ends failed. We label only those two ends --
    which we actually know -- and let a linear model interpolate the middle.
    The result is one number per reading, with a readable coefficient per sensor.
    """
    span = train.groupby("unit")["cycle"].transform("max")
    frac = (train["cycle"] - 1) / (span - 1).clip(lower=1)

    healthy = train[frac <= HEALTHY_FRAC]
    failed = train[frac >= 1 - FAILED_FRAC]

    x = pd.concat([healthy[cols], failed[cols]])
    y = np.concatenate([np.ones(len(healthy)), np.zeros(len(failed))])

    mu, sigma = train[cols].mean(), train[cols].std().replace(0, 1)
    model = LinearRegression().fit((x - mu) / sigma, y)
    return model, mu, sigma


def health(df: pd.DataFrame, model, mu, sigma, cols: list[str]) -> np.ndarray:
    return model.predict((df[cols] - mu) / sigma)


def rul_by_extrapolation(test: pd.DataFrame, hi: np.ndarray) -> np.ndarray:
    """For each unit: fit a line through its recent health trend, read off where
    it crosses zero. This is what a mechanic does by eye with a wear gauge."""
    out = []
    frame = test.assign(hi=hi)
    for _, unit in frame.groupby("unit", sort=True):
        tail = unit.tail(EXTRAP_WINDOW)
        # Smooth first: single readings are noisy, trends are not.
        smoothed = tail["hi"].rolling(7, min_periods=1).mean().to_numpy()
        cycles = tail["cycle"].to_numpy()
        slope, intercept = np.polyfit(cycles, smoothed, 1)
        if slope >= -1e-6:  # not degrading detectably -> assume plenty of life
            out.append(RUL_CAP)
            continue
        zero_at = (0.0 - intercept) / slope
        out.append(float(np.clip(zero_at - cycles[-1], 0, RUL_CAP)))
    return np.asarray(out)


def rul_by_calibration(train: pd.DataFrame, test: pd.DataFrame, hi_train, hi_test):
    """Map the health index to remaining life with one monotone curve.

    Fitted once on training engines, then published. A mechanic can read this
    curve off a single chart: this health number means roughly this much life
    left. Nothing is hidden inside it.
    """
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(hi_train, train["rul"].clip(upper=RUL_CAP))
    latest = test.assign(hi=hi_test).groupby("unit").tail(1).sort_values("unit")
    return iso.predict(latest["hi"]), iso


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    cols = live_sensors()
    train = add_rul(load("train"))
    test = load("test")
    truth = pd.read_csv(DATA / "RUL_FD001.txt", sep=r"\s+", header=None).iloc[:, 0].to_numpy()

    print(f"train: {train['unit'].nunique()} engines, {len(train):,} readings")
    print(f"test:  {test['unit'].nunique()} engines, {len(test):,} readings")
    print(f"sensors used: {len(cols)} of 21 ({len(DEAD_SENSORS)} dropped as flat)\n")

    # --- transparent, two flavours -----------------------------------------
    model, mu, sigma = fit_health_index(train, cols)
    hi_train = health(train, model, mu, sigma, cols)
    hi_test = health(test, model, mu, sigma, cols)

    pred_extrap = rul_by_extrapolation(test, hi_test)
    pred_transparent, _ = rul_by_calibration(train, test, hi_train, hi_test)

    # --- black box: random forest on capped RUL ----------------------------
    y_train = train["rul"].clip(upper=RUL_CAP)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1)
    rf.fit(train[cols], y_train)
    last_reading = test.groupby("unit").tail(1).sort_values("unit")
    pred_blackbox = rf.predict(last_reading[cols])

    # --- naive floor: everyone gets the fleet average ----------------------
    pred_naive = np.full_like(truth, np.mean(y_train), dtype=float)

    capped = np.clip(truth, 0, RUL_CAP)
    rows = [
        ("fleet average (does nothing)", rmse(pred_naive, capped)),
        ("health index -> zero-crossing extrapolation", rmse(pred_extrap, capped)),
        ("health index -> monotone calibration curve", rmse(pred_transparent, capped)),
        ("random forest (black box)", rmse(pred_blackbox, capped)),
    ]
    width = max(len(r[0]) for r in rows)
    print("RMSE in cycles, 100 held-out engines, lower is better")
    for name, score in rows:
        print(f"  {name:<{width}}  {score:6.2f}")

    gap = rmse(pred_transparent, capped) - rmse(pred_blackbox, capped)
    print(f"\ncost of insisting on transparency: {gap:+.2f} cycles RMSE")
    print(
        "the zero-crossing variant was the first thing I tried and it is clearly worse;\n"
        "it is left in because the failure is the more useful half of the result."
    )

    print("\ntop health-index coefficients (sensor -> contribution to health):")
    order = np.argsort(-np.abs(model.coef_))[:6]
    for i in order:
        print(f"  {cols[i]:>4}  {model.coef_[i]:+.3f}")

    out = pd.DataFrame(
        {
            "unit": sorted(test["unit"].unique()),
            "true_rul": truth,
            "pred_transparent": pred_transparent.round(1),
            "pred_blackbox": pred_blackbox.round(1),
        }
    )
    dest = DATA.parent / "cmapss_predictions.csv"
    out.to_csv(dest, index=False)
    print(f"\nper-engine predictions written to {dest.relative_to(DATA.parents[1])}")


if __name__ == "__main__":
    main()
