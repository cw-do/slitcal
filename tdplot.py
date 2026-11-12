"""Time-distance diagrams of the plan steps, several per page, wavelength in colour.

    python slitcal.py diagrams --plan slitcal_IPTS-<ipts>/plan.csv      (or written by `script`)

Each panel: time after the proton pulse (ms, four pulses) against distance from the moderator
(m). Every chopper disc is drawn at its distance as a bar that is dark where the disc is closed;
the two discs of the pair under test are red. Neutrons leave each pulse at the mean emission delay
for their wavelength; those that pass all six discs are drawn to the detector in their wavelength's
colour (the eqsans-model scale: violet 1 A to red 15 A), the blocked ones faintly up to the disc
that stops them. Opacity is in proportion to each ray's flux (spectrum x transmission). The label gives the wavelength ranges that reach the detector, each with its share
of the transmitted flux (flux spectrum and emission spread included).
"""
import colorsys
import os

import numpy as np

import slitcal as sc

LAM_TOP = 15.0                         # top of the colour scale, as in eqsans-model
PULSES = range(-2, 4)                  # earlier pulses too: slow neutrons arrive frames later
T_MAX = 4 * sc.T60 / 1000.0            # ms shown
PAIR_SPLIT = 0.22                      # m, drawing offset between the two discs of a pair


def lam_rgb(lam):
    t = np.clip((np.asarray(lam, float) - 1.0) / (LAM_TOP - 1.0), 0, 1)
    return np.array([colorsys.hls_to_rgb(0.76 * (1 - x), 0.55, 1.0) for x in np.ravel(t)])


def _gate(t, centre, open_us):
    u = np.mod(t - centre + sc.T60 / 2, sc.T60) - sc.T60 / 2
    return np.clip((open_us / 2 - np.abs(u)) / sc.TAU_B + 0.5, 0.0, 1.0)


def bands(y, lam, frac=0.01):
    """Contiguous wavelength ranges carrying more than frac of the peak, with their share."""
    on = y > frac * y.max()
    out, i = [], 0
    while i < len(lam):
        if on[i]:
            j = i
            while j + 1 < len(lam) and on[j + 1]:
                j += 1
            out.append((lam[i], lam[j], y[i:j + 1].sum() / y.sum()))
            i = j + 1
        else:
            i += 1
    return sorted(out, key=lambda b: -b[2])


def panel(ax, row, offsets, model, sdd=4.0):
    L_det = sc.MSD + sdd
    cen = {d: (float(row[f'Skf{sc.DISCS[d][0]}_TotalDelay']) - offsets[d]) % sc.T60 for d in sc.ORDER
           if str(row[f'Skf{sc.DISCS[d][0]}_TotalDelay']).strip() not in ('', 'nan')}
    active = list(cen)
    pair = [p for p in sc.PAIRS if p[0][0] == str(row['pair'])][0]

    # frames, monitor, detector
    for k in range(1, 4):
        ax.axvline(k * sc.T60 / 1000, color='0.85', lw=0.6, zorder=0)
    for y, lab in ((sc.L_MON, 'monitor'), (L_det, 'detector')):
        ax.axhline(y, color='0.55', lw=0.6, ls='--', zorder=1)
        ax.text(T_MAX * 0.995, y + 0.15, lab, ha='right', va='bottom', fontsize=5.5, color='0.4')

    # rays
    from matplotlib.collections import LineCollection
    phi = model.phi
    wmax = None
    for step, faint in ((0.25, True), (0.02, False)):
        lam = np.arange(0.6, 35.0, step)
        segs, cols, alph = [], [], []
        rgb = lam_rgb(lam)
        delay = sc.emission_delay(lam)
        flux = phi(lam)
        for k in PULSES:
            t0 = k * sc.T60 + delay
            g = np.ones_like(lam)
            stop = np.full_like(lam, L_det)
            for d in sorted(active, key=lambda d: sc.DISCS[d][1]):
                L = sc.DISCS[d][1]
                gd = _gate(t0 + sc.tof(L, lam), cen[d], sc.open_time(d))
                newly = (g > 0.05) & (gd <= 0.05)
                stop[newly] = L
                g = g * gd
            for i in range(len(lam)):
                passed = g[i] > 0.05
                if faint == passed:
                    continue
                L_end = L_det if passed else stop[i]
                ta = t0[i] / 1000
                tb = (t0[i] + sc.tof(L_end, lam[i])) / 1000
                if tb < 0 or ta > T_MAX:
                    continue
                segs.append([(ta, 0), (tb, L_end)])
                cols.append(rgb[i])
                alph.append(0.10 if faint else float(g[i] * flux[i]))
        if segs:
            a = np.array(alph)
            if not faint:   # opacity in proportion to the ray's flux: overlapping rays then add up to the flux share
                wmax = a.max()
                a = 0.03 + 0.97 * (a / wmax)
            c = np.column_stack([np.array(cols), a])
            ax.add_collection(LineCollection(segs, colors=c, linewidths=0.35 if faint else 0.7, zorder=2 if faint else 4))

    # discs: dark where closed
    for d in active:
        L = sc.DISCS[d][1] + (-PAIR_SPLIT if d.endswith('a') else PAIR_SPLIT)
        half = sc.open_time(d) / 2
        col = '#c0392b' if d in pair else '0.15'
        for k in range(-1, 5):
            a = (cen[d] + half + k * sc.T60) / 1000
            b = (cen[d] - half + (k + 1) * sc.T60) / 1000
            ax.plot([max(a, 0), min(b, T_MAX)], [L, L], color=col, lw=1.6, solid_capstyle='butt', zorder=5) if b > 0 and a < T_MAX else None
    # labels
    y = sc.passed_by_wavelength(cen, model, active=active)
    bs = bands(y, model.lam)
    txt = '\n'.join(f'{a:.1f}-{b:.1f} A  {s * 100:.0f}%' for a, b, s in bs[:3]) or 'nothing passes'
    ax.text(0.015, 0.97, txt, transform=ax.transAxes, va='top', ha='left', fontsize=5.5,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', lw=0.4, alpha=0.9), zorder=6)
    extra = ''
    if row.get('beamslit1'):
        extra = f"  |  {row['beamslit1']}, {float(row['pc']):.2f} C"
    ax.set_title(f"Step {int(row['step'])}: pair {row['pair']} {row['block']} {float(row['slit_lambda']):g} A, "
                 f"{float(row['width_us']):g} us{extra}", fontsize=6.5, pad=2)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0, L_det + 0.8)
    ax.set_yticks([0, sc.DISCS['1a'][1], sc.DISCS['2a'][1], sc.DISCS['3a'][1], L_det])
    ax.set_yticklabels(['0', 'st 1', 'st 2', 'st 3', f'det {L_det:.1f} m'], fontsize=5.5)
    ax.tick_params(axis='x', labelsize=5.5, length=2)
    ax.tick_params(axis='y', length=2)


def diagrams(rows, offsets, out_pdf, sdd=4.0, per_page=8, header=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    model = sc.Model(lam=(0.8, 35.0, 0.01), nt=12)
    cmap = ListedColormap(lam_rgb(np.linspace(1, LAM_TOP, 256)))
    ncol, nrow = 2, per_page // 2
    pages = [rows[i:i + per_page] for i in range(0, len(rows), per_page)]
    with PdfPages(out_pdf) as pdf:
        for pi, page in enumerate(pages, 1):
            fig, axs = plt.subplots(nrow, ncol, figsize=(8.5, 11), squeeze=False)
            for ax, row in zip(axs.flat, page):
                panel(ax, row, offsets, model, sdd)
            for ax in list(axs.flat)[len(page):]:
                ax.axis('off')
            for ax in axs[-1]:
                ax.set_xlabel('time after the proton pulse (ms)', fontsize=6)
            fig.suptitle(f'EQ-SANS chopper calibration runs: time-distance diagrams ({pi}/{len(pages)})', fontsize=10, y=0.985)
            fig.text(0.5, 0.963, 'Bars: disc closed (red: pair under test; disc a drawn below b). Coloured lines: neutrons '
                     'reaching the detector, opacity in proportion to their flux; faint: blocked.\nBox: wavelength ranges at the '
                     'detector and their share of the transmitted flux (flux spectrum and emission spread included).',
                     ha='center', va='center', fontsize=6, color='0.3')
            if header:
                fig.text(0.5, 0.947, header, ha='center', fontsize=5.5, color='0.45')
            cax = fig.add_axes([0.25, 0.062, 0.5, 0.008])
            cb = fig.colorbar(ScalarMappable(Normalize(1, LAM_TOP), cmap), cax=cax, orientation='horizontal')
            cb.set_label('wavelength (A); 15 A and longer: red', fontsize=7)
            cb.ax.tick_params(labelsize=6)
            fig.text(0.5, 0.012, 'Note: the two discs of a pair are really 1.0-1.6 cm apart; they are drawn 0.44 m apart '
                     '(a below, b above) so that both bars can be seen.\nRays are drawn at the true distances, so near a '
                     'bar a ray can appear to cross it (a slow neutron takes up to ~1 ms for 0.22 m). '
                     'Whether a neutron passes is calculated at the true disc positions.', ha='center', va='bottom',
                     fontsize=6, color='0.2', linespacing=1.4)
            fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.11, hspace=0.38, wspace=0.12)
            pdf.savefig(fig)
            plt.close(fig)
    return out_pdf
