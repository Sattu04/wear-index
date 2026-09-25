# Wear Index
Live demo: https://wear-index-gxgccuwtpazeeut7ryo3yc.streamlit.app/

A sketch, not a product. Built in a day, off public data, by someone who does not
work at SPARQ.

In May, Codrin said the next step was predicting mechanical issues from driving
patterns before they happen — brake-riding, oil mistakes, the things that should
be predictable. This is one engineer's attempt at the shape of that, put in front
of you so you can tell me where it's wrong.

## What it does

Takes OBD-II trip logs. Reduces them to ten wear stressors. Converts those into a
per-subsystem wear rate against a fleet baseline, and turns book service intervals
into intervals for *this* driver.

```
Front brake pads: wearing 3.1x faster than the fleet median.
Book-interval 55,000 km becomes 17,739 km for this driver.
  - time resting on the brake: 0.216 of moving time (fleet median 0.142).
    Light continuous pad contact keeps the pad above its designed operating
    temperature, glazing the friction material and scoring the rotor.
  - hard stops: 2.43 per 100 km (fleet median 1.14).
```
That is driver D001 from `python src/run_demo.py`, unedited.
Every number traces back to a stressor, and every stressor traces back to a
mechanism. No step in the chain is a black box, because a driver won't act on a
number they can't argue with, and a mechanic won't stake a repair order on one.

## What it explicitly does not do

It does not predict failures. It is an **exposure model**, not a failure model.

That distinction is the whole reason this is honest. No public automotive dataset
carries real failure labels, so "your pads fail in 3,000 km" is a claim I have no
basis for. "You wear pads 2.3x faster than the fleet and here are the two
behaviours doing it" follows from the data that does exist.

Only warranty and parts-replacement records close that gap. SPARQ is one of the
few companies positioned to have them.

## Does transparency actually cost accuracy?

Worth checking rather than asserting. NASA's C-MAPSS turbofan dataset has real
remaining-life labels, so both approaches can be scored on the same held-out
engines:

```
RMSE in cycles, 100 held-out engines, lower is better
  fleet average (does nothing)                  41.94
  health index -> zero-crossing extrapolation   33.81
  health index -> monotone calibration curve    19.53
  random forest (black box)                     17.21
```

The transparent method costs **2.31 cycles of RMSE**, about 13%. That is the price
of being able to explain the number, and on this evidence it's a price worth
paying.

The zero-crossing variant was my first attempt and it was much worse. It's left in
the output because the failure is the more useful half of the result.

Reproduce: `python src/validate_cmapss.py`

## Running it

```bash
pip install -r requirements.txt
python src/validate_cmapss.py   # the validation above, on real NASA data
python src/run_demo.py          # the automotive pipeline, on synthetic trips
streamlit run app.py            # driver-facing readout
```

The automotive demo runs on **synthetic** trips (`src/synth.py`) so it works in
thirty seconds with no download. Nothing generated there is a finding. The real
loader is `src/load_ved.py`, for the Michigan Vehicle Energy Dataset.

## What two OBD-II PIDs are worth

Public driving datasets don't log engine coolant temperature (Mode 01 PID `0x05`)
or throttle position (`0x11`). Both are standard, and any device sitting in the
port already reads them. Losing them removes four of the ten stressors —
including the brake one, which is the only output a driver would pay for.

Worth measuring rather than asserting. Every driver scored twice, once on all ten
stressors and once on the six that survive:

```
                   removed  stressors_lost  top-pick changes  wear-rate error  rank shift
  throttle position (0x11)               1             15.0%             4.1%        0.05
coolant temperature (0x05)               3             12.0%            11.1%        0.28
                 both PIDs               4             25.5%            15.2%        0.32
```

**Without those two channels, the subsystem a driver is told to worry about first
changes for about one in four drivers.** Stable across seeds (23–28% across five seeds of 200 drivers).

The damage isn't spread evenly. Brakes and engine internals take ~25% wear-rate
error; transmission takes none, because it never depended on those channels.

The degraded run renormalises the surviving weights rather than scoring missing
stressors as zero. That's deliberately generous to the degraded model — treating
absent data as "no stress" would inflate the gap and would be a dishonest way to
make the point.

Reproduce: `python src/ablate_pids.py`

Caveat worth stating plainly: this runs on synthetic trips, so it measures how
much the *model* depends on those channels, not how much real-world accuracy is
lost. The honest version of this experiment needs real logs from a device that
reads all ten.

## The three things wrong with this

1. **The sensitivities are guessed.** Ten constants in `wear_index.py` set how
   sharply each stressor multiplies wear. They come from engineering judgement,
   not from data. They are the first thing real parts-replacement data would fit,
   and the first thing I'd expect to be wrong.

2. **Brake use is inferred, not measured.** Standard OBD-II has no brake pedal
   PID, so brake riding is inferred from sustained light deceleration with a
   closed throttle. There's a false-positive floor: coasting downhill looks the
   same. A single brake-light or pressure signal would replace the whole
   inference.

3. **Public data is missing the best channels.** VED has no coolant temperature
   and no throttle position, which kills four of the ten stressors — including
   the brake one. Both are standard Mode 01 PIDs that any device in the port
   already reads. That gap is a gap in public data, not in the method.

## Layout

```
src/validate_cmapss.py   transparent vs black box on labelled NASA data
src/obd_features.py      ten stressors, each with its wear mechanism attached
src/wear_index.py        stressors -> per-subsystem wear rate and interval
src/synth.py             synthetic trip generator (demo only)
src/load_ved.py          real loader, and what VED is missing
src/ablate_pids.py       what coolant temp and throttle position are worth
src/run_demo.py          the whole automotive pipeline, end to end
app.py                   Streamlit readout
```

## Licence and data

Code MIT. C-MAPSS is NASA open data. VED is not redistributed here — fetch it from
the source and check its terms.
