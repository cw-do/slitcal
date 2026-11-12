# slitcal — EQ-SANS chopper phase-offset calibration

The method, why it works and how it was validated: `EQSANS_chopper_phase_calibration.pdf`.

## Set up on analysis.sns.gov

    git clone <repository> slitcal && cd slitcal
    PY=/usr/local/pixi/sans/.pixi/envs/default/bin/python      # has numpy, scipy, h5py, matplotlib

`bl6_flux_2025A_Jan_rebinned.txt` (the wavelength spectrum used by the model) must stay next to
`slitcal.py`, or pass `--spectrum`.

## 1. Plan — what to type into the chopper phases

    $PY slitcal.py plan --out plan.csv

33 runs, each with the six TotalDelay values (Skf1..Skf6), computed from today's offsets:
for each pair a 500 µs slit at 3, 4.5, 6, 8, 10.5, 13.5 Å (slit scan) and slits of 800–1400 µs
at 6 Å (width scan). If today's offsets are not the built-in ones, give them with
`--offsets offsets.json` (`{"1a": 15072.89, "1b": ..., ...}`) and use the same file in step 3.

## 2. Measure

Choppers at 60 Hz, one SDD (4 m), attenuated empty beam. For each row enter the six TotalDelay
values as the chopper phases and take a run. Nothing else needs to be written down — every run
file carries its own phases — except the range of run numbers.
If a slit run shows no peak in the monitor, today's offsets are off by more than ~200 µs: widen it.

## 3. Analyse — from the IPTS and the run numbers

    $PY slitcal.py analyze --ipts 37424 --runs 180100-180132
    $PY slitcal_gui.py --ipts 37424 --runs 180100-180132        # same, in a window

What happens:

* each `/SNS/EQSANS/IPTS-37424/nexus/EQSANS_<run>.nxs.h5` is read once; its monitor and detector
  events are summed and rebinned (10 µs, one frame) and written to
  `slitcal_IPTS-37424/spectra/EQSANS_<run>_spectra.dat` — columns `tof_us monitor detector`, with
  Phase1..6, Speed1..6, detector distance and proton charge in the header;
  (`$PY slitcal.py extract --ipts ... --runs ...` does only this step)
* every run is recognised from its phases: which pair is the slit, its wavelength and width,
  slit scan (width < 700 µs) or width scan;
* all runs are fitted; results go to `slitcal_IPTS-37424/results/`.

The fit can also be run on a folder of `*_spectra.dat` files: `$PY slitcal.py analyze path/to/spectra`.

## 4. Result

`results/offsets.csv`: reference, correction and new offset per disc (EPICS index, 60 Hz and 30 Hz).
`results.json` adds the emission scale and the width-scan numbers; `fit_spectra.png`,
`width_scan.png`, `corrections.png`. Look at the residuals before installing; put the same values
in `hexaSub.c` and the drtsans JSON.

## Test without the instrument

    $PY slitcal.py plan --out plan.csv
    $PY slitcal.py simulate plan.csv --out fake_SNS_EQSANS     # hidden truth: fake_SNS_EQSANS/TRUTH.json
    $PY slitcal.py analyze --data-root fake_SNS_EQSANS --ipts 0 --runs 900001-900033

## Read from each NeXus file

`entry/monitor1/event_time_offset`, `entry/bank*_events/event_time_offset`; from `entry/DASlogs`:
`Phase1..6` (falls back to `PhaseRequest1..6`), `Speed1..6`, `detectorZ`; `entry/proton_charge`.
EPICS chopper index 1..6 = discs 1b, 2b, 3a, 3b, 1a, 2a.
