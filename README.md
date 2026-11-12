# slitcal — EQ-SANS chopper phase-offset calibration

The method, why it works and how it was validated: `EQSANS_chopper_phase_calibration.pdf`.

## Set up on analysis.sns.gov

    git clone <repository> slitcal && cd slitcal
    PY=/usr/local/pixi/sans/.pixi/envs/default/bin/python      # has numpy, scipy, h5py, matplotlib

`bl6_flux_2025A_Jan_rebinned.txt` (the wavelength spectrum used by the model) must stay next to
`slitcal.py`, or pass `--spectrum`.

## 1. Plan — the phases to type in, step by step

Pick a **reference run**: any recent run taken with an ordinary 60 Hz band request (all six discs
spinning). The tool reads from it the offsets the control system uses now
(offset = recorded TotalDelay − what calc6Phases computes for that request).

    $PY slitcal.py plan --ipts 37424 --ref-run 180099                 # six values per step
    $PY slitcal.py plan --ipts 37424 --ref-run 180099 --mode parked   # only the pair under test spins

Writes `slitcal_IPTS-37424/steps.txt` (instructions), `worksheet.csv` (the same as a table with an
empty run-number column), `plan.csv`, `offsets_in_use.json`. 33 steps: for each pair a 500 µs slit
at 3, 4.5, 6, 8, 10.5, 13.5 Å (slit scan) and slits of 800–1400 µs at 6 Å (width scan).

Columns are in beam order, each with its EPICS chopper number:

| position (upstream → downstream) | 1a | 1b | 2a | 2b | 3a | 3b |
|---|---|---|---|---|---|---|
| EPICS chopper | Skf5 | Skf1 | Skf6 | Skf2 | Skf3 | Skf4 |

### Or: the EQ-SANS scan script

    $PY slitcal.py script --ipts <IPTS for the runs> --ref-run 190564 --ref-ipts 37485

writes `slitcal_IPTS-<ipts>/eqsans_chopcal.py`, a script for the EQ-SANS scan server
(`eqsans_scanfunctions_live`): `setipts`, `loadconf('conf_4000mm_2p5A_60Hz_trans')`, then per run
shutter closed → `set_chopper_delay` ×6 → `openShutter` → `runsampleid('T-chopcal ...')` →
`closeShutter`; at the end the choppers go back to automatic mode. Beam slit and proton charge per
run come from the 2026 flux runs of IPTS-37618 (`flux_refs_ipts37618.json`, see `scanscript.py`):
d25_Cd for the 3 and 4.5 Å slits, direct beam d5mm from 6 Å on (peak rate ≤ 0.62 of the highest
already measured with d5mm), about 9.4 C in total. `runs_rates.csv` lists the expected counts.
Empty beam: `--pos -1` (default) leaves the sample table where it is.
It also writes `time_distance.pdf`: a time-distance diagram of every run, eight per page, wavelength
in colour (the eqsans-model scale), showing which wavelengths reach the detector and their share.
For any plan: `$PY slitcal.py diagrams slitcal_IPTS-<ipts>/plan.csv`.

## 2. Measure

Choppers at 60 Hz, SDD 4 m (the same for every run), attenuated empty beam, the same proton charge
per run. For each step: set the six chopper phases (TotalDelay) to the listed values, take the run,
write down the run number. (Only the run-number range is really needed: each run file carries its
own phases.) In parked mode: park the four listed discs open, set the two.

## 3. Analyse — from the IPTS and the run numbers

    $PY slitcal.py analyze --ipts 37424 --runs 180100-180132
    $PY slitcal_gui.py --ipts 37424 --ref-run 180099 --runs 180100-180132      # same, in a window

* each `/SNS/EQSANS/IPTS-37424/nexus/EQSANS_<run>.nxs.h5` is read once; its monitor and detector
  events are summed and rebinned (10 µs, one frame) into
  `slitcal_IPTS-37424/spectra/EQSANS_<run>_spectra.dat` — columns `tof_us monitor detector`, with
  Phase1..6, Speed1..6, detector distance and proton charge in the header
  (`$PY slitcal.py extract --ipts ... --runs ...` does only this step);
* every run is recognised from its phases and speeds: which pair is the slit, its wavelength and
  width, slit scan (width < 700 µs) or width scan; parked discs (speed 0) count as open;
* all runs are fitted, starting from `offsets_in_use.json`; results go to `slitcal_IPTS-37424/results/`.

The analysis does not need the current offsets: the recorded phases include them, and the fit
returns the offsets themselves. The values in use are only its starting point.

## 4. Result

`results/offsets.csv`: offset in use, correction and new offset per disc (EPICS index, 60 Hz and
30 Hz). `results.json` adds the emission scale and the width-scan numbers; `fit_spectra.png`,
`width_scan.png`, `corrections.png`. Look at the residuals before installing; put the same values
in `hexaSub.c` and the drtsans JSON.

## Test without the instrument

    $PY slitcal.py plan --out p                                   # plan from the built-in offsets
    $PY slitcal.py simulate p/plan.csv --out fake                 # reference run 900000 + runs 900001-900033
    $PY slitcal.py plan --ipts 0 --ref-run 900000 --data-root fake --out work
    $PY slitcal.py analyze --data-root fake --ipts 0 --runs 900001-900033 --out work
    # hidden truth: fake/TRUTH.json

## Read from each NeXus file

`entry/monitor1/event_time_offset`, `entry/bank*_events/event_time_offset`; from `entry/DASlogs`:
`Phase1..6` (falls back to `PhaseRequest1..6`), `Speed1..6`, `detectorZ`, `Wavelength`;
`entry/proton_charge`. EPICS chopper index 1..6 = discs 1b, 2b, 3a, 3b, 1a, 2a.
