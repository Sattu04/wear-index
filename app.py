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
        st.markdown(
            f'<div class="part">{v["label"]}</div>'
            f'<div class="rate{cls}">{v["multiplier"]:.2f}&times;</div>'
            f'<div class="km">wear rate vs fleet · '
            f'{v["nominal_km"]:,} km book interval becomes {v["effective_life_km"]:,.0f} km · '
            f'<strong>{v["remaining_km"]:,.0f} km left</strong></div>',
            unsafe_allow_html=True,
        )
        for reason in v["reasons"][:2]:
            if abs(reason["z"]) < 0.25:
                continue
            arrow = "above" if reason["z"] > 0 else "below"
            st.markdown(
                f'<div class="why">{reason["label"]} — {reason["value"]:.3g} '
                f'{reason["units"]}, {arrow} the fleet median of '
                f'{reason["fleet_median"]:.3g}. {reason["mechanism"]}.</div>',
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
