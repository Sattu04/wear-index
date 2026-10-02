"""
Wear Index -- driver-facing readout.

    streamlit run app.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

import pandas as pd
import streamlit as st

from obd_features import STRESSORS, driver_profile
from synth import ARCHETYPES, make_fleet
from wear_index import assess, fleet_stats

st.set_page_config(page_title="Wear Index", layout="wide")

# What a driver could actually do about each stressor. A verdict with no lever
# attached is just an accusation, and people close apps that accuse them.
ACTIONS = {
    "brake_dwell_frac": "easing off the pedal on long descents rather than resting on it",
    "harsh_decel_per_100km": "leaving a little more space and braking earlier",
    "cold_high_load_events": "keeping the revs light for the first two minutes",
    "short_trip_frac": "combining short trips, or one longer run a week",
    "lugging_frac": "changing down before the engine starts labouring",
    "high_rpm_frac": "shifting up slightly earlier",
    "idle_frac": "switching off rather than idling for over a minute",
    "thermal_excursions_per_100km": "having the cooling system looked at",
    "stft_abs_mean": "having the intake and sensors checked",
    "stop_start_per_km": "a quieter route where there is a choice",
}

st.markdown(
    """
    <style>
      .stApp { background:#12151a; color:#e8eaed; }
      h1,h2,h3 { color:#e8eaed; font-weight:600; letter-spacing:-0.01em; }
      .rail { border-left:3px solid #3a4150; padding:0.15rem 0 0.15rem 0.9rem; margin:0 0 1.15rem 0; }
      .rail.hot { border-left-color:#e8603c; }
      .part { font-size:1.05rem; font-weight:600; }
      .rate { font-variant-numeric:tabular-nums; font-size:2.1rem; font-weight:600; line-height:1.1; }
      .rate.hot { color:#e8603c; }
      .plain { color:#e8eaed; font-size:0.97rem; line-height:1.55; max-width:62ch;
               margin:0.5rem 0 0.15rem; }
      .why { color:#9aa3b0; font-size:0.86rem; line-height:1.5; max-width:62ch; }
      .km { color:#9aa3b0; font-size:0.9rem; font-variant-numeric:tabular-nums; }
      .warn { background:#1c1a14; border:1px solid #5a4a22; color:#d8c89a;
              padding:0.7rem 0.9rem; font-size:0.84rem; margin-bottom:1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Reading trip logs…")
def build(n_drivers: int):
    fleet = make_fleet(n_drivers=n_drivers, trips_per_driver=12)
    profiles = [driver_profile(d["trips"]) for d in fleet]
    return fleet, profiles, fleet_stats(profiles)


st.title("Wear Index")
st.markdown(
    '<div class="warn">Running on synthetic trips. The numbers below demonstrate the '
    "method, not the state of any real vehicle. Point <code>src/load_ved.py</code> at the "
    "Michigan Vehicle Energy Dataset to run it on real logs.</div>",
    unsafe_allow_html=True,
)

fleet, profiles, stats = build(40)
labels = [f"{d['driver_id']} · {d['archetype'].replace('_', ' ')}" for d in fleet]

left, right = st.columns([1, 2.3], gap="large")

with left:
    choice = st.selectbox("Driver", range(len(fleet)), format_func=lambda i: labels[i])
    odo = st.number_input("Kilometres since last service", 0, 200_000, 24_000, step=1_000)
    profile = profiles[choice]
    st.markdown(
        f'<div class="km">{profile["n_trips"]} trips · '
        f'{profile["total_km"]:,.0f} km logged</div>',
        unsafe_allow_html=True,
    )

verdicts = assess(profile, stats, {s.key: odo for s in __import__("wear_index").SUBSYSTEMS})

with right:
    for rank, v in enumerate(verdicts):
        hot = v["multiplier"] >= 1.25 or v["remaining_km"] < 5_000
        cls = " hot" if hot and rank == 0 else ""
        st.markdown(f'<div class="rail{cls}">', unsafe_allow_html=True)
        shorter = v["effective_life_km"] < v["nominal_km"] * 0.9
        st.markdown(
            f'<div class="part">{v["label"]}</div>'
            f'<div class="rate{cls}">{v["remaining_km"]:,.0f}'
            f'<span style="font-size:1rem;font-weight:400;"> km</span></div>'
            f'<div class="km">before it needs attention &middot; '
            + (
                f'this car gets there around {v["effective_life_km"]:,.0f} km '
                f'rather than the usual {v["nominal_km"]:,}'
                if shorter
                else f'tracking the usual {v["nominal_km"]:,} km interval'
            )
            + '</div>',
            unsafe_allow_html=True,
        )
        for reason in v["reasons"][:2]:
            if reason["z"] < 0.25:  # only surface what is actually adding wear
                continue
            lever = ACTIONS.get(reason["stressor"])
            tail = f' {lever[0].upper()}{lever[1:]} would stretch that.' if lever else ''
            st.markdown(
                f'<div class="why"><strong>Biggest factor:</strong> '
                f'{reason["label"]}.{tail} '
                f'<span style="opacity:0.65">{reason["mechanism"]}.</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Every stressor, and what it is measured against"):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "stressor": s.label,
                    "this driver": round(profile[k], 4),
                    "fleet median": round(stats[k][0], 4),
                    "units": s.units,
                    "wears out": s.mechanism,
                }
                for k, s in STRESSORS.items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
