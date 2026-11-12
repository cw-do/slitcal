"""slitcal -- EQ-SANS chopper phase-offset calibration with the two-flight-path slit method.

Commands:

  python slitcal.py plan --out plan.csv
        Run list with the six TotalDelay values for every run (slit scan + width scan), computed
        from the reference offsets (today's values unless --offsets is given).

  python slitcal.py extract --ipts 37424 --runs 180100-180132
        Reads /SNS/EQSANS/IPTS-37424/nexus/EQSANS_<run>.nxs.h5 for every run and writes one ASCII
        file per run, slitcal_IPTS-37424/spectra/EQSANS_<run>_spectra.dat: the summed monitor and
        detector TOF spectra (10 us bins, one frame) with the six phases, speeds, detector distance
        and proton charge in the header.

  python slitcal.py analyze --ipts 37424 --runs 180100-180132
        Extracts (skipping runs already extracted), recognises in every run which pair was the slit,
        its wavelength and width, fits all runs, and writes slitcal_IPTS-37424/results/
        (offsets.csv, results.json, figures). Folders or files of *_spectra.dat can be given instead.

  python slitcal.py simulate plan.csv --out fake_SNS_EQSANS
        Synthetic NeXus files laid out like the archive (fake_SNS_EQSANS/IPTS-0/nexus/...), from a
        hidden instrument, to test the analysis: analyze --data-root fake_SNS_EQSANS --ipts 0 ...

The GUI (slitcal_gui.py) drives the same functions.

Conventions (match hexaSub.c / drtsans): TotalDelay = time of the cut-out centre after T0 plus
the disc's phase offset, modulo one period. EPICS chopper index 1..6 = discs 1b, 2b, 3a, 3b, 1a, 2a.
Distances and opening angles: drtsans EQSANS_chopper_configurations.json, entry 2026-03-04.
Physics and validation: report/calibration_report.pdf (Topic_2026_choppers).
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

H = 3956.0346                       # m/s * A
T60 = 1e6 / 60.0                    # us
HERE = os.path.dirname(os.path.abspath(__file__))

DISCS = {  # name: (EPICS index, distance from moderator m, opening deg)
    '1a': (5, 5.6601247, 129.600), '1b': (1, 5.6757668, 129.600),
    '2a': (6, 7.7601247, 180.000), '2b': (2, 7.7757668, 180.000),
    '3a': (3, 9.4978, 230.010), '3b': (4, 9.5078, 230.007),
}
ORDER = ['1a', '1b', '2a', '2b', '3a', '3b']
PAIRS = (('1a', '1b'), ('2a', '2b'), ('3a', '3b'))
BY_IDX = {v[0]: k for k, v in DISCS.items()}
REF_OFFSETS = {'1a': 15072.89, '1b': 14954.46, '2a': 14834.04, '2b': 14805.40, '3a': 14726.06, '3b': 14565.69}
L_MON = 10.006
MSD = 14.122                        # moderator to sample (m)
TAU_B = 0.040 / (2 * np.pi * 0.28925 * 60) * 1e6   # beam sweep, us
BIN = 10.0
EDGES = np.arange(0, T60 + 1e-6, BIN)
TC = 0.5 * (EDGES[1:] + EDGES[:-1])
SLIT_W, WIDTHS = 500.0, [800.0, 950.0, 1100.0, 1250.0, 1400.0]
SLIT_LAMS, WIDTH_LAM = [3.0, 4.5, 6.0, 8.0, 10.5, 13.5], 6.0


def tof(L, lam):
    return L * np.asarray(lam, float) / H * 1e6


def open_time(d):
    return DISCS[d][2] / 360.0 * T60


def emission_delay(lam):
    """drtsans correct_frame.emission_delay (us), vectorised."""
    w = np.asarray(lam, float)
    p6 = 0.5 * (1280.5 - 7448.4 * w + 16509 * w**2 - 17872 * w**3 + 10445 * w**4 - 3169.3 * w**5 + 392.31 * w**6)
    p3 = 0.5 * (231.99 + 6.4797 * w - 0.5233 * w**2 + 0.0148 * w**3)
    return np.maximum(np.where(w < 0.5, 9.451719 * w, np.where(w < 1.879329786923, p6, p3)), 1.0)


def load_spectrum(path=None):
    d = np.loadtxt(path or os.path.join(HERE, 'bl6_flux_2025A_Jan_rebinned.txt'), skiprows=1)
    lam, phi = d[:, 0], d[:, 1]

    def f(x):
        x = np.asarray(x, float)
        y = np.interp(x, lam, phi, left=0.0, right=0.0)
        return np.where(x > lam[-1], phi[-4:].mean() * (x / lam[-3]) ** -5, y)
    return f


# ------------------------------------------------------------------ forward model
class Model:
    """Expected TOF histograms for given cut-out centre times. Departure times Gamma(k) with
    mean s * drtsans delay; each disc edge ramps linearly over tau_b."""

    def __init__(self, phi=None, lam=(1.2, 22.0, 0.003), nt=24, k=2.0, tau_b=TAU_B):
        self.phi = phi or load_spectrum()
        self.lam = np.arange(*lam)
        self.nt, self.k, self.tau_b = nt, k, tau_b
        self._tof = {d: tof(DISCS[d][1], self.lam)[:, None] for d in ORDER}
        self.phil = self.phi(self.lam)[:, None]

    def nodes(self, s):
        m = s * emission_delay(self.lam)[:, None]
        k = self.k
        u = np.linspace(0, k + 12 * np.sqrt(k), self.nt + int(16 / k))[None, :]
        w = u ** (k - 1) * np.exp(-u)
        return u * (m / k), w / w.sum(axis=1, keepdims=True)

    def gate(self, t, tc, To):
        u = np.mod(t - tc + T60 / 2, T60) - T60 / 2
        return np.clip((To / 2 - np.abs(u)) / self.tau_b + 0.5, 0.0, 1.0)

    def spectra(self, centres, s, paths, d0=0.0):
        t0, w = self.nodes(s)
        g = np.ones_like(t0)
        for d in ORDER:
            g = g * self.gate(t0 + self._tof[d], centres[d], open_time(d))
        wt = (self.phil * w * g).ravel()
        nb = len(EDGES) - 1
        out = []
        for Lx in paths:
            T = np.mod(t0 + tof(Lx, self.lam)[:, None] + d0, T60).ravel()
            u = T / BIN - 0.5
            i0 = np.floor(u).astype(int)
            f = u - i0
            out.append(np.bincount(np.mod(i0, nb), wt * (1 - f), nb) + np.bincount(np.mod(i0 + 1, nb), wt * f, nb))
        return out


# ------------------------------------------------------------------ planning
def slit_centres(pair, lam_s, w):
    """Wanted cut-out centre times (us after T0) for a slit of width w on `pair` at lam_s;
    the other stations wide open around lam_s."""
    a, b = pair
    t_mid = tof(DISCS[a][1], lam_s) + float(emission_delay(lam_s))
    c = {a: t_mid - w / 2 + open_time(a) / 2, b: t_mid + w / 2 - open_time(b) / 2}
    for p in PAIRS:
        if p != pair:
            for d in p:
                c[d] = tof(DISCS[d][1], lam_s) + float(emission_delay(lam_s))
    return c


def to_totaldelay(centres, offsets):
    return {DISCS[d][0]: (centres[d] + offsets[d]) % T60 for d in ORDER}


def make_plan(offsets=None, lams=SLIT_LAMS, widths=WIDTHS, wlam=WIDTH_LAM, w=SLIT_W):
    offsets = offsets or REF_OFFSETS
    rows = []
    for pair in PAIRS:
        for l in lams:
            rows.append(('slit', pair, l, w))
        for ww in widths:
            rows.append(('width', pair, wlam, ww))
    out = []
    for n, (block, pair, l, ww) in enumerate(rows, 1):
        td = to_totaldelay(slit_centres(pair, l, ww), offsets)
        r = dict(step=n, block=block, pair=pair[0][0], slit_lambda=l, width_us=ww, speed_hz=60)
        r.update({f'Skf{i}_TotalDelay': round(td[i], 2) for i in range(1, 7)})
        out.append(r)
    return out


# ------------------------------------------------------------------ reading NeXus
def _log_mean(logs, name):
    if name not in logs:
        return np.nan
    v = np.asarray(logs[name]['value'][:], float)
    return float(np.nanmean(v)) if v.size else np.nan


def _hist_events(ds, chunk=20_000_000):
    """Histogram an event_time_offset dataset in chunks (real runs hold 10^7-10^8 events)."""
    h = np.zeros(len(EDGES) - 1)
    for i in range(0, ds.shape[0], chunk):
        h += np.histogram(np.mod(np.asarray(ds[i:i + chunk], float), T60), EDGES)[0]
    return h


def read_run(path, det_banks=None):
    """Monitor and detector TOF histograms (10 us bins, one frame) and the chopper settings."""
    import h5py
    with h5py.File(path, 'r') as f:
        e = f['entry']
        logs = e['DASlogs']
        mon_h = _hist_events(e['monitor1']['event_time_offset'])
        det_h = np.zeros(len(EDGES) - 1)
        for k in e.keys():
            if k.startswith('bank') and k.endswith('_events') and (det_banks is None or k in det_banks):
                det_h += _hist_events(e[k]['event_time_offset'])
        speed = {i: _log_mean(logs, f'Speed{i}') for i in range(1, 7)}
        phase = {i: _log_mean(logs, f'Phase{i}') for i in range(1, 7)}
        req = {i: _log_mean(logs, f'PhaseRequest{i}') for i in range(1, 7)}
        dz = _log_mean(logs, 'detectorZ')
        pc = float(np.ravel(e['proton_charge'][()])[0]) if 'proton_charge' in e else np.nan
        run = os.path.basename(path).split('.')[0]
    for i in range(1, 7):                     # fall back to the request if the readback is missing or wild
        if not np.isfinite(phase[i]) or abs(phase[i] - req[i]) > 50:
            phase[i] = req[i]
    return dict(path=path, run=run, mon=mon_h, det=det_h, speed=speed, phase=phase,
                L_det=MSD + (dz / 1000.0 if np.isfinite(dz) else 4.0), pcharge=pc)


# ------------------------------------------------------------------ run numbers -> files -> ASCII spectra
DATA_ROOT = '/SNS/EQSANS'


def parse_runs(text):
    """'180100-180105, 180110' -> [180100, ..., 180105, 180110]"""
    out = []
    for part in str(text).replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = (int(x) for x in part.split('-'))
            out += list(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def nexus_path(ipts, run, root=DATA_ROOT):
    return os.path.join(root, f'IPTS-{int(ipts)}', 'nexus', f'EQSANS_{int(run)}.nxs.h5')


def write_spectra(r, folder):
    """One ASCII file per run: chopper settings in the header, then TOF, monitor, detector."""
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, f"{r['run']}_spectra.dat")
    with open(p, 'w') as f:
        f.write(f"# slitcal spectra; source: {r['path']}\n")
        f.write(f"# run: {r['run']}\n")
        for i in range(1, 7):
            f.write(f"# Phase{i}: {r['phase'][i]:.4f}\n# Speed{i}: {r['speed'][i]:.4f}\n")
        f.write(f"# L_det_m: {r['L_det']:.5f}\n# proton_charge: {r['pcharge']:.6g}\n")
        f.write('# columns: tof_us_bin_centre  monitor_counts  detector_counts\n')
        for t, m, d in zip(TC, r['mon'], r['det']):
            f.write(f'{t:.1f} {m:.0f} {d:.0f}\n')
    return p


def read_spectra(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            if ':' in line:
                k, v = line[1:].split(':', 1)
                meta[k.strip()] = v.strip()
    d = np.loadtxt(path, comments='#')
    return dict(path=path, run=meta.get('run', os.path.basename(path).split('_spectra')[0]),
                mon=d[:, 1], det=d[:, 2],
                phase={i: float(meta[f'Phase{i}']) for i in range(1, 7)},
                speed={i: float(meta[f'Speed{i}']) for i in range(1, 7)},
                L_det=float(meta['L_det_m']), pcharge=float(meta['proton_charge']))


def extract(ipts, runs, folder, root=DATA_ROOT, progress=print, overwrite=False):
    """Read EQSANS_<run>.nxs.h5 of an IPTS, write <folder>/EQSANS_<run>_spectra.dat; return the paths."""
    out = []
    for run in runs:
        dat = os.path.join(folder, f'EQSANS_{run}_spectra.dat')
        if os.path.exists(dat) and not overwrite:
            out.append(dat)
            progress(f'EQSANS_{run}: already extracted')
            continue
        src = nexus_path(ipts, run, root)
        if not os.path.exists(src):
            progress(f'EQSANS_{run}: {src} not found; skipped')
            continue
        r = read_run(src)
        out.append(write_spectra(r, folder))
        progress(f'EQSANS_{run}: monitor {r["mon"].sum():.0f}, detector {r["det"].sum():.0f} counts -> {os.path.basename(out[-1])}')
    return out


def load_inputs(paths):
    """Files or folders of *_spectra.dat or *.nxs.h5 -> list of run dicts (ASCII preferred)."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            dats = sorted(glob.glob(os.path.join(p, '*_spectra.dat')))
            files += dats if dats else sorted(glob.glob(os.path.join(p, '*.nxs.h5')))
        else:
            files.append(p)
    return [read_spectra(f) if f.endswith('.dat') else read_run(f) for f in files]


def classify(run, offsets):
    """Which pair is the slit, its wavelength and width, from the six phases."""
    cen = {d: (run['phase'][DISCS[d][0]] - offsets[d]) % T60 for d in ORDER}
    best = None
    for a, b in PAIRS:
        o = cen[a] - open_time(a) / 2                  # a opening edge
        c = cen[b] + open_time(b) / 2                  # b closing edge
        w = (c - o) % T60
        if w < 3000 and (best is None or w < best[2]):
            t_mid = (o + w / 2) % T60
            best = ((a, b), t_mid, w)
    if best is None:
        return None
    pair, t_mid, w = best
    # The slit time is known only modulo one period, so its wavelength has aliases every
    # h*T/L (11.6 A at station 1). The other stations are centred on the same wavelength with
    # different alias spacings; the wavelength on which all of them agree is the real one.
    def cands(t, L):
        return [(t + k * T60 - 125.0) / tof(L, 1.0) for k in range(5)]
    mine = [x for x in cands(t_mid, DISCS[pair[0]][1]) if 1.0 < x < 25]
    others = []
    for p in PAIRS:
        if p != pair:
            tm = (cen[p[0]] + cen[p[1]]) / 2
            Lm = (DISCS[p[0]][1] + DISCS[p[1]][1]) / 2
            others.append([x for x in cands(tm % T60, Lm) if 0.5 < x < 30])
    lam_s = min(mine, key=lambda x: sum(min(abs(x - y) for y in o) for o in others)) if mine else None
    block = 'slit' if w < 700 else 'width'
    return dict(pair=pair, lam=lam_s, width=w, block=block, centres=cen)


# ------------------------------------------------------------------ analysis
def fit_slit(runs, model, progress=print):
    """Global fit of offset corrections e (us, added to the reference offsets) and emission scale."""
    from scipy.optimize import least_squares
    data = []
    for r in runs:
        for key, L in (('mon', L_MON), ('det', r['L_det'])):
            data.append((r, key, L))

    def resid(z):
        e = dict(zip(ORDER, z[:6] * 50.0))
        s = z[6]
        out = []
        for r in runs:
            cen = {d: r['cls']['centres'][d] - e[d] for d in ORDER}
            ym, yd = model.spectra(cen, s, (L_MON, r['L_det']))
            for obs, y in ((r['mon'], ym), (r['det'], yd)):
                sel = obs > 0
                a = obs.sum() / max(y.sum(), 1e-30)
                out.append((obs[sel] - a * y[sel]) / np.sqrt(obs[sel]))
        return np.concatenate(out)
    progress(f'slit-scan fit: {len(runs)} runs, {2 * len(runs)} spectra, 7 parameters ...')
    fit = least_squares(resid, np.r_[np.zeros(6), 1.0], diff_step=0.02)
    e = dict(zip(ORDER, fit.x[:6] * 50.0))
    J = fit.jac
    try:
        cov = np.linalg.inv(J.T @ J) * (2 * fit.cost / max(len(fit.fun) - 7, 1))
        err = dict(zip(ORDER, np.sqrt(np.diag(cov))[:6] * 50.0))
    except np.linalg.LinAlgError:
        err = dict.fromkeys(ORDER, np.nan)
    progress(f'  done: chi2/dof = {2 * fit.cost / max(len(fit.fun) - 7, 1):.2f}, emission scale {fit.x[6]:.3f}')
    return e, err, float(fit.x[6]), 2 * fit.cost / max(len(fit.fun) - 7, 1)


def fit_width(runs, eB, s, model, progress=print):
    """Per pair: e_b - e_a from total monitor counts per proton charge versus slit width."""
    from scipy.optimize import minimize_scalar
    out = {}
    for a, b in PAIRS:
        rr = [r for r in runs if r['cls']['pair'] == (a, b)]
        if len(rr) < 3:
            progress(f'width scan: pair {a[0]} has {len(rr)} runs; skipped')
            continue
        n = np.array([r['mon'].sum() for r in rr])
        pc = np.array([r['pcharge'] if np.isfinite(r['pcharge']) and r['pcharge'] > 0 else 1.0 for r in rr])
        y = n / pc
        wts = pc ** 2 / np.maximum(n, 1)
        c = (eB[a] + eB[b]) / 2

        def chi2(delta):
            m = []
            for r in rr:
                e = dict(eB)
                e[a], e[b] = c - delta / 2, c + delta / 2
                cen = {d: r['cls']['centres'][d] - e[d] for d in ORDER}
                m.append(model.spectra(cen, s, (L_MON,))[0].sum())
            m = np.array(m)
            k = (y * m * wts).sum() / (m * m * wts).sum()
            return ((y - k * m) ** 2 * wts).sum()
        res = minimize_scalar(chi2, bounds=(-450, 450), method='bounded', options={'xatol': 0.05})
        out[(a, b)] = float(res.x)
        progress(f'width scan pair {a[0]}: e_b - e_a = {res.x:+.1f} us from {len(rr)} runs')
    return out


def prepare(raw, offsets, progress=print):
    """Classify every run; drop runs that are not part of the calibration."""
    runs = []
    for r in raw:
        r['cls'] = classify(r, offsets)
        if r['cls'] is None:
            progress(f'{r["run"]}: no pair forms a slit; skipped')
            continue
        if any(abs(r['speed'][i] - 60) > 0.5 for i in range(1, 7) if np.isfinite(r['speed'][i])):
            progress(f'{r["run"]}: not all discs at 60 Hz; skipped')
            continue
        runs.append(r)
        c = r['cls']
        progress(f'{r["run"]}: {c["block"]:5s} pair {c["pair"][0][0]}  lambda {c["lam"]:.2f} A  width {c["width"]:.0f} us  '
                 f'monitor {r["mon"].sum():.0f}  detector {r["det"].sum():.0f}')
    return runs


def analyze(raw, offsets=None, out=None, progress=print, spectrum=None):
    """raw: run dicts (from load_inputs / extract). Fits and, if `out`, writes the results."""
    offsets = offsets or REF_OFFSETS
    runs = prepare(raw, offsets, progress)
    model = Model(load_spectrum(spectrum))
    slit = [r for r in runs if r['cls']['block'] == 'slit']
    width = [r for r in runs if r['cls']['block'] == 'width']
    eB, errB, s, chi2 = fit_slit(slit, model, progress)
    dW = fit_width(width, eB, s, model, progress) if width else {}
    e = dict(eB)
    for (a, b), dl in dW.items():
        c = (eB[a] + eB[b]) / 2
        e[a], e[b] = c - dl / 2, c + dl / 2
    new = {d: offsets[d] + e[d] for d in ORDER}
    res = dict(reference_offsets=offsets, correction_us=e, slit_only_correction_us=eB, slit_fit_sigma_us=errB,
               width_scan_delta_us={f'{a}-{b}': v for (a, b), v in dW.items()}, emission_scale=s,
               chi2_per_dof=chi2, new_offsets=new, runs=[dict(run=r['run'], block=r['cls']['block'],
               pair=r['cls']['pair'][0][0], lam=r['cls']['lam'], width=r['cls']['width']) for r in runs])
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=1)
        with open(os.path.join(out, 'offsets.csv'), 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['disc', 'EPICS_index', 'reference_us', 'correction_us', 'new_offset_us', 'new_30Hz_offset_us'])
            for d in ORDER:
                wr.writerow([d, DISCS[d][0], f'{offsets[d]:.2f}', f'{e[d]:+.2f}', f'{new[d]:.2f}', f'{2 * new[d]:.2f}'])
        plots(runs, res, model, out)
        progress(f'wrote {out}')
    return res, runs, model


def plots(runs, res, model, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs = figures(runs, res, model, plt)
    for name, fig in figs.items():
        fig.savefig(os.path.join(out, name + '.png'), dpi=130, bbox_inches='tight')
        plt.close(fig)


def figures(runs, res, model, plt):
    """Figures as matplotlib Figure objects (used by the CLI and the GUI)."""
    e, s = res['correction_us'], res['emission_scale']
    figs = {}
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, pair in zip(axs, PAIRS):
        for r in [r for r in runs if r['cls']['block'] == 'slit' and r['cls']['pair'] == pair]:
            cen = {d: r['cls']['centres'][d] - e[d] for d in ORDER}
            ym, yd = model.spectra(cen, s, (L_MON, r['L_det']))
            for obs, y, col in ((r['mon'], ym, '#7f8c8d'), (r['det'], yd, '#2c3e50')):
                ax.plot(TC / 1000, obs, color=col, lw=0.7)
                ax.plot(TC / 1000, y * obs.sum() / y.sum(), color='#e67e22', lw=0.9, ls='--')
        ax.set_title(f'pair {pair[0][0]}: slit-scan spectra (grey monitor, black detector) and fit (orange)', fontsize=9)
        ax.set_ylabel('counts / 10 us')
    axs[-1].set_xlabel('TOF (ms, one frame)')
    fig.tight_layout()
    figs['fit_spectra'] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    for pair, col in zip(PAIRS, ('#1f6fb2', '#16a085', '#e67e22')):
        rr = [r for r in runs if r['cls']['block'] == 'width' and r['cls']['pair'] == pair]
        if rr:
            w = [r['cls']['width'] for r in rr]
            y = [r['mon'].sum() / (r['pcharge'] if np.isfinite(r['pcharge']) and r['pcharge'] > 0 else 1) for r in rr]
            ax.plot(w, np.array(y) / max(y), 'o', color=col, label=f'pair {pair[0][0]}')
    ax.set_xlabel('slit width from the reference offsets (us)')
    ax.set_ylabel('monitor counts / proton charge (normalised)')
    ax.legend(frameon=False)
    ax.set_title('width scan', fontsize=10)
    figs['width_scan'] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(6)
    ax.bar(x, [e[d] for d in ORDER], color='#1f6fb2')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}\n(Skf{DISCS[d][0]})' for d in ORDER])
    ax.axhline(0, color='#555', lw=0.8)
    ax.set_ylabel('correction to the reference offset (us)')
    ax.set_title(f'new offset = reference + correction; emission scale {s:.3f}', fontsize=10)
    figs['corrections'] = fig
    return figs


# ------------------------------------------------------------------ synthetic runs
def simulate(plan_rows, out, truth=None, seed=1, slit_counts=4e5, width_counts=1e6, progress=print, ipts=0, first_run=900001):
    """Write fake event NeXus files for a plan from a hidden instrument (for testing), laid out like
    the real archive: <out>/IPTS-<ipts>/nexus/EQSANS_<run>.nxs.h5, runs numbered from first_run.
    Analyse them with --data-root <out> --ipts <ipts> --runs <first_run>-<last>."""
    import h5py
    rng = np.random.default_rng(seed)
    truth = truth or {'offset_error_us': {d: float(rng.uniform(-150, 150)) for d in ORDER},
                      'emission_scale': 1.15, 'das_zero_us': 6.0, 'k': 2.0, 'tau_b': TAU_B}
    model = Model(k=truth['k'], tau_b=truth['tau_b'])
    os.makedirs(out, exist_ok=True)
    json.dump(truth, open(os.path.join(out, 'TRUTH.json'), 'w'), indent=1)
    nxdir = os.path.dirname(nexus_path(ipts, 0, out))
    os.makedirs(nxdir, exist_ok=True)
    for r in plan_rows:
        run = first_run + int(r['step']) - 1
        td = {i: float(r[f'Skf{i}_TotalDelay']) for i in range(1, 7)}
        phys = {d: (td[DISCS[d][0]] - REF_OFFSETS[d] - truth['offset_error_us'][d]) % T60 for d in ORDER}
        ym, yd = model.spectra(phys, truth['emission_scale'], (L_MON, MSD + 4.0), truth['das_zero_us'])
        if r['block'] == 'slit':
            nm, nd = slit_counts, slit_counts
        else:                                          # counts proportional to the transmitted flux
            ref = model.spectra(slit_centres((r['pair'] + 'a', r['pair'] + 'b'), WIDTH_LAM, max(WIDTHS)), truth['emission_scale'], (L_MON,))[0].sum()
            nm = width_counts * ym.sum() / ref
            nd = nm * 0.4
        cm_ = rng.poisson(ym / ym.sum() * nm)
        cd_ = rng.poisson(yd / yd.sum() * nd)
        ev = lambda c: (np.repeat(EDGES[:-1], c) + rng.uniform(0, BIN, c.sum())).astype(np.float32)
        path = nexus_path(ipts, run, out)
        with h5py.File(path, 'w') as f:
            e = f.create_group('entry')
            e.create_dataset('proton_charge', data=[1.0e12])
            e.create_dataset('duration', data=[600.0])
            e.create_group('monitor1').create_dataset('event_time_offset', data=ev(cm_))
            e.create_group('bank1_events').create_dataset('event_time_offset', data=ev(cd_))
            lg = e.create_group('DASlogs')
            for i in range(1, 7):
                lg.create_group(f'Speed{i}').create_dataset('value', data=np.full(5, 60.0))
                lg.create_group(f'PhaseRequest{i}').create_dataset('value', data=[td[i]])
                lg.create_group(f'Phase{i}').create_dataset('value', data=td[i] + rng.normal(0, 0.5, 20))
            lg.create_group('detectorZ').create_dataset('value', data=[4000.0])
        progress(f'wrote {os.path.basename(path)}  ({r["block"]}, pair {r["pair"]}, {r["slit_lambda"]} A, {r["width_us"]} us)')
    return truth


def read_plan(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r['step'] = int(r['step'])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--out', default='plan.csv')
    p.add_argument('--offsets', help='JSON {disc: offset_us} to use instead of the built-in reference')
    p = sub.add_parser('extract', help='NeXus -> one ASCII spectra file per run')
    p.add_argument('--ipts', type=int, required=True)
    p.add_argument('--runs', required=True, help='e.g. "180100-180132, 180140"')
    p.add_argument('--data-root', default=DATA_ROOT)
    p.add_argument('--out', help='work folder (default slitcal_IPTS-<ipts>); spectra go to <out>/spectra')
    p.add_argument('--overwrite', action='store_true')
    p = sub.add_parser('analyze', help='fit; from --ipts/--runs (extracts first) or from spectra/NeXus paths')
    p.add_argument('inputs', nargs='*', help='folders or files (*_spectra.dat or *.nxs.h5)')
    p.add_argument('--ipts', type=int)
    p.add_argument('--runs')
    p.add_argument('--data-root', default=DATA_ROOT)
    p.add_argument('--out', help='work folder (default slitcal_IPTS-<ipts> or slitcal_results)')
    p.add_argument('--offsets', help='JSON {disc: offset_us} that the plan was made with')
    p.add_argument('--spectrum', help='wavelength spectrum file (lambda, intensity)')
    p = sub.add_parser('simulate', help='synthetic NeXus files for a plan (testing)')
    p.add_argument('plan')
    p.add_argument('--out', default='fake_SNS_EQSANS')
    p.add_argument('--ipts', type=int, default=0)
    p.add_argument('--first-run', type=int, default=900001)
    p.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    offs = json.load(open(a.offsets)) if getattr(a, 'offsets', None) else None
    if a.cmd == 'plan':
        rows = make_plan(offs)
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(f"{r['step']:2d} {r['block']:5s} pair {r['pair']} {r['slit_lambda']:5.1f} A {r['width_us']:6.0f} us  " +
                  '  '.join(f"Skf{i} {r[f'Skf{i}_TotalDelay']:9.2f}" for i in range(1, 7)))
        print(f'wrote {a.out}')
    elif a.cmd == 'simulate':
        simulate(read_plan(a.plan), a.out, seed=a.seed, ipts=a.ipts, first_run=a.first_run)
    elif a.cmd == 'extract':
        out = a.out or f'slitcal_IPTS-{a.ipts}'
        extract(a.ipts, parse_runs(a.runs), os.path.join(out, 'spectra'), a.data_root, overwrite=a.overwrite)
    else:
        if a.ipts is not None and a.runs:
            out = a.out or f'slitcal_IPTS-{a.ipts}'
            files = extract(a.ipts, parse_runs(a.runs), os.path.join(out, 'spectra'), a.data_root)
            raw = [read_spectra(f) for f in files]
        else:
            out = a.out or 'slitcal_results'
            raw = load_inputs(a.inputs)
        res, _, _ = analyze(raw, offs, os.path.join(out, 'results'), spectrum=a.spectrum)
        print(json.dumps({d: round(res['new_offsets'][d], 2) for d in ORDER}, indent=1))


if __name__ == '__main__':
    main()
