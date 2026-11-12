"""EQ-SANS scan script for the calibration runs: phases from the plan, beam slit and proton charge
per run from the 2026 flux measurements.

    python slitcal.py script --ipts <IPTS for the runs> --ref-run <recent 60 Hz run> [--ref-ipts N]

writes slitcal_IPTS-<ipts>/eqsans_chopcal.py (to load on the EQ-SANS scan server) next to the plan
files. Uses eqsans_scanfunctions_live: setipts, loadconf, beamslit1, set_chopper_delay, openShutter,
runsampleid, closeShutter, setpv, estimatetime.

Count rates and the beam slit
-----------------------------
Basis: the 2026 flux measurements of IPTS-37618 (SDD 4 m, 60 Hz, six discs spinning), saved in
flux_refs_ipts37618.json: recorded phases, beamslit1 position, proton charge, total and peak
(per 10 us TOF bin) monitor and detector counts of 25 runs. Slit names from the run titles:
position 1 = d5mm (direct beam, used by the standard 4 m 10 A transmission), 3 = d15mm,
5 = d25mm, 7 = d25_Cd (standard 4 m 2.5 A transmission, strongly attenuated).

Each reference run is modelled with Model from its own recorded phases. The ratio measured / model
gives, per beam slit, a factor K(lambda), fitted as a power law:
  detector, d25_Cd : 0.032-0.037 from 2.5 to 14 A (flat: the model's spectrum shape is right)
  detector, d5mm   : rises with wavelength, 0.47 at 2.5 A to 2.4 at 10 A
  monitor          : the same for both slits, steeply rising (monitor efficiency)
A run's rates are K(slit wavelength) x the model's prediction for its phases.

Overflow guard: the largest detector peak already recorded without trouble is 2.79e5 counts per C
per 10 us bin (EQSANS_186454, 2.5 A mono, 10 % spread, d5mm). A run uses the direct beam d5mm when
its predicted peak stays below PEAK_FRAC of that, otherwise d25_Cd. Each width-scan pair keeps one
slit for all five widths, since the width scan compares counts per proton charge.
"""
import json
import os
import re
import subprocess
from datetime import date

import numpy as np

import slitcal as sc

HERE = os.path.dirname(os.path.abspath(__file__))
REFS_FILE = os.path.join(HERE, 'flux_refs_ipts37618.json')

SLITS = {1: 'd5mm', 7: 'd25_Cd'}
SAFE_PEAK = 2.79e5     # counts / C / 10 us bin, EQSANS_186454
PEAK_FRAC = 0.8
NEED_SLIT = 4e5        # counts in each counter for a slit-scan run
NEED_WIDTH_MON = 1e6   # monitor counts for the narrowest width-scan run of a pair
PC_STEP = 0.05
PC_MIN = 0.1
EXCLUDE = ('186456',)  # 4 A sp5 d5mm: K_det 0.20, a quarter of the fit through its neighbours; left out
BEAM_C_PER_H = 3600 * 0.3005 / 244.0   # proton charge per hour during the reference runs (186454)
TRANS_CONF = 'conf_4000mm_2p5A_60Hz_trans'


def _lam(title):
    m = re.search(r'4m ([\d.]+)a', title, re.I)
    lam = float(m.group(1))
    return lam if 'sp' in title.lower() else lam + 2.0   # band runs: middle of a ~4 A band


class Rates:
    """K(lambda) per slit from the flux reference runs; predicted rates of plan rows."""

    def __init__(self, model=None):
        self.model = model or sc.Model()
        refs = json.load(open(REFS_FILE))
        pts = {'det1': [], 'det7': [], 'mon': []}
        pk = []
        for r, x in refs.items():
            lam = _lam(x['title'])
            if lam < 2.4 or x['s1'] not in SLITS:
                continue
            c = {d: (x['phase'][str(sc.DISCS[d][0])] - sc.REF_OFFSETS[d]) % sc.T60 for d in sc.ORDER}
            ym, yd = self.model.spectra(c, 1.0, (sc.L_MON, sc.MSD + 4.0))
            kd, km = x['det'] / x['pc'] / yd.sum(), x['mon'] / x['pc'] / ym.sum()
            if r not in EXCLUDE:
                pts[f'det{x["s1"]}'].append((lam, kd))
            pts['mon'].append((lam, km))
            if x['s1'] == 1:
                pk.append((x['det_peak'] / x['pc'] / yd.max()) / kd)
        self.fit = {}
        for k, v in pts.items():
            l, y = np.array(v).T
            p, lna = np.polyfit(np.log(l), np.log(y), 1)
            self.fit[k] = (float(np.exp(lna)), float(p))
        self.peak_factor = float(max(pk))           # measured peak / model peak, worst case

    def k_det(self, s1, lam):
        a, p = self.fit[f'det{s1}']
        return a * lam ** p

    def k_mon(self, lam):
        a, p = self.fit['mon']
        return a * lam ** p

    def annotate(self, rows, offsets, sdd=4.0):
        """Add beamslit1, predicted rates and the proton charge to band-mode plan rows."""
        for x in rows:
            c = {d: (x[f'Skf{sc.DISCS[d][0]}_TotalDelay'] - offsets[d]) % sc.T60 for d in sc.ORDER}
            ym, yd = self.model.spectra(c, 1.0, (sc.L_MON, sc.MSD + sdd))
            lam = x['slit_lambda']
            x['_ym'], x['_yd'], x['_ydmax'] = ym.sum(), yd.sum(), yd.max()
            x['s1'] = 1 if self.k_det(1, lam) * self.peak_factor * yd.max() <= PEAK_FRAC * SAFE_PEAK else 7
        for p in ('1', '2', '3'):
            ws = [x for x in rows if x['block'] == 'width' and x['pair'] == p]
            s = 1 if all(x['s1'] == 1 for x in ws) else 7
            for x in ws:
                x['s1'] = s
        for x in rows:
            lam = x['slit_lambda']
            x['beamslit1'] = SLITS[x['s1']]
            x['rate_mon'] = self.k_mon(lam) * x['_ym']
            x['rate_det'] = self.k_det(x['s1'], lam) * x['_yd']
            x['peak_per_C'] = self.k_det(x['s1'], lam) * self.peak_factor * x['_ydmax']
            if x['block'] == 'slit':
                pc = NEED_SLIT / min(x['rate_mon'], x['rate_det'])
                x['pc'] = max(PC_MIN, float(np.ceil(pc / PC_STEP) * PC_STEP))
        for p in ('1', '2', '3'):
            ws = [x for x in rows if x['block'] == 'width' and x['pair'] == p]
            pc = NEED_WIDTH_MON / min(x['rate_mon'] for x in ws)
            for x in ws:
                x['pc'] = max(PC_MIN, float(np.ceil(pc / PC_STEP) * PC_STEP))
        for x in rows:
            x['mon_expected'] = x['rate_mon'] * x['pc']
            x['det_expected'] = x['rate_det'] * x['pc']
            for k in ('_ym', '_yd', '_ydmax', 's1'):
                x.pop(k)
        return rows


def _commit():
    try:
        return subprocess.run(['git', '-C', HERE, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                              timeout=10).stdout.strip() or '?'
    except Exception:  # noqa: BLE001
        return '?'


def title(x, n):
    return (f"T-chopcal {x['step']:02d}/{n} pair{x['pair']} {x['block']} {x['slit_lambda']:g}A "
            f"{x['width_us']:g}us 4m {x['beamslit1']}")


def write_script(rows, offsets, source, ipts, path, rack='peltier', pos=-1):
    """The EQ-SANS script. Runs grouped by beam slit (d25_Cd first, as loadconf leaves it), each
    run: shutter closed -> six phases -> open -> runsampleid -> close."""
    order = sorted(rows, key=lambda x: (x['beamslit1'] != 'd25_Cd', x['step']))
    n = len(rows)
    tot = sum(x['pc'] for x in rows)
    L = ['# EQ-SANS chopper phase-offset calibration (slitcal), %d transmission runs at SDD 4 m, 60 Hz' % n,
         f'# Generated {date.today().isoformat()} by slitcal {_commit()}: python slitcal.py script',
         f'# Offsets in use: {source}',
         '#   ' + '  '.join(f'{d} (Skf{sc.DISCS[d][0]}) {offsets[d]:.2f}' for d in sc.ORDER),
         f'# Estimated beam: {tot:.2f} C of proton charge, about {tot / BEAM_C_PER_H:.1f} h at the 2026 beam power,',
         '# plus the time to re-phase the choppers between runs.',
         '#',
         '# Before loading:',
         '#   * EMPTY BEAM: nothing in the beam at the sample position (pos = -1 leaves the table where it is).',
         '#   * Every run is a transmission run in the 4 m 2.5 A transmission configuration; only beamslit1',
         '#     changes: d25_Cd for the short-wavelength runs, direct beam d5mm from 6 A on. The proton charge',
         '#     per run and the slit come from the 2026 flux runs of IPTS-37618 (peak rate <= 0.62 of the',
         '#     highest peak already measured there with d5mm).',
         '#   * set_chopper_delay(i, v) puts chopper Skf<i> in manual mode and sets its TotalDelay to v (us).',
         '#     The shutter is closed while the phases change, so no white beam reaches the detector.',
         '#   * After the first run, check that its file records the six phases as set:',
         '#       python slitcal.py extract --ipts <ipts> --runs <first run>   (header of the .dat file)',
         '#   * Order does not matter to the analysis; a repeated run does no harm.',
         '#   * At the end the choppers go back to automatic mode (set_choppers_auto() covers Skf1-4 only,',
         '#     so Skf5 and Skf6 are set with setpv) and the standard configuration is reloaded.',
         '',
         'from epics import caget',
         'from scan import *',
         'import os',
         'import sys',
         "sys.path.append('/home/controls/var/tmp/scripting/dev/')",
         'from eqsans_scanfunctions_live import *',
         '',
         f'setipts({ipts})',
         '',
         f"loadconf('{TRANS_CONF}')    # 4 m, 60 Hz, transmission beamstop, beamslit1 d25_Cd",
         'closeShutter()',
         '']
    cur = 'd25_Cd'
    for x in order:
        if x['beamslit1'] != cur:
            cur = x['beamslit1']
            L += ['', f'# ---- from here: beamslit1 {cur}' + (' (direct beam)' if cur == 'd5mm' else ''),
                  f"beamslit1('{cur}')", '']
        L.append(f"# step {x['step']:2d}: pair {x['pair']} {x['block']} at {x['slit_lambda']:g} A, {x['width_us']:g} us; "
                 f"expected monitor {x['mon_expected']:.2g}, detector {x['det_expected']:.2g}")
        for d in sc.ORDER:   # beam order 1a 1b 2a 2b 3a 3b
            i = sc.DISCS[d][0]
            L.append(f"set_chopper_delay({i}, {x[f'Skf{i}_TotalDelay']:.2f})    # {d}"
                     + ('  <- slit' if x[f'Skf{i}_action'] == 'SET' else ''))
        L += ['openShutter()',
              f"runsampleid('{title(x, n)}', 0, '{rack}', 'pc', {pos}, {x['pc']:.2f})",
              'closeShutter()', '']
    L += ['# ---- back to normal operation',
          'closeShutter()',
          'set_choppers_auto()',
          "setpv('BL6:Chop:Skf5:OpModeSet', 1)",
          "setpv('BL6:Chop:Skf6:OpModeSet', 1)",
          f"loadconf('{TRANS_CONF}')",
          '',
          'estimatetime()',
          '']
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(L))
    return path


def write_rates_csv(rows, path):
    import csv
    keys = ['step', 'block', 'pair', 'slit_lambda', 'width_us', 'request_lmin', 'purity', 'beamslit1', 'rate_mon',
            'rate_det', 'peak_per_C', 'pc', 'mon_expected', 'det_expected']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for x in rows:
            w.writerow([f'{x[k]:.4g}' if isinstance(x[k], float) else x[k] for k in keys])
