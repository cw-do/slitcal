"""slitcal GUI: plan the runs, load the NeXus files, fit, and look at the result.

    python slitcal_gui.py
    python slitcal_gui.py --ipts 37424 --runs 180100-180132
    python slitcal_gui.py --data-root fake_SNS_EQSANS --ipts 0 --runs 900001-900033 --fit --screenshot shot.png  (test)

Tab 1 (Plan): the six TotalDelay values to type in for every run. The offsets in use are read from a
reference run (an ordinary 60 Hz band request); * marks the two discs that make the slit.
Tab 2 (Runs): give the IPTS and run numbers; the tool reads /SNS/EQSANS/IPTS-x/nexus/EQSANS_<run>.nxs.h5,
writes one ASCII spectra file per run in the work folder, and classifies each run (which pair is the
slit, its wavelength and width, slit scan or width scan) from the phases stored in it.
Tab 3 (Result): new offsets, the fit figures, and Save.
"""
import argparse
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slitcal as sc  # noqa: E402


class App:
    def __init__(self, root):
        self.root = root
        root.title('slitcal — EQ-SANS chopper phase-offset calibration')
        root.geometry('1250x820')
        self.q = queue.Queue()
        self.runs, self.res, self.model = [], None, None
        nb = ttk.Notebook(root)
        nb.pack(fill='both', expand=True)
        self.nb = nb
        self._plan_tab(nb)
        self._runs_tab(nb)
        self._result_tab(nb)
        root.after(100, self._poll)

    # ------------------------------------------------------------ plan
    def _plan_tab(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text='1  Plan the runs')
        ref = ttk.LabelFrame(f, text='Offsets in use now — read them from a reference run (any run taken with an ordinary 60 Hz band request)', padding=6)
        ref.pack(fill='x')
        self.ref_ipts, self.ref_run, self.ref_lmin = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.plan_root = tk.StringVar(value=sc.DATA_ROOT)
        for j, (lab, var, wd) in enumerate((('IPTS', self.ref_ipts, 8), ('reference run', self.ref_run, 10),
                                            ('its minimum λ (blank: from file)', self.ref_lmin, 6), ('data root', self.plan_root, 20))):
            ttk.Label(ref, text=lab).grid(row=0, column=2 * j, sticky='e', padx=(8, 2))
            ttk.Entry(ref, textvariable=var, width=wd).grid(row=0, column=2 * j + 1, sticky='w')
        ttk.Button(ref, text='Read reference run', command=self.read_reference).grid(row=0, column=8, padx=10)
        self.off_vars = {}
        for j, d in enumerate(sc.ORDER):
            ttk.Label(ref, text=f'{d} (Skf{sc.DISCS[d][0]})').grid(row=1, column=(j % 6) * 2 if j < 4 else (j % 6) * 2, sticky='e', padx=(8, 2), pady=4)
            v = tk.StringVar(value=f'{sc.REF_OFFSETS[d]:.2f}')
            ttk.Entry(ref, textvariable=v, width=10).grid(row=1, column=(j % 6) * 2 + 1, sticky='w')
            self.off_vars[d] = v
        self.off_source = tk.StringVar(value='Offsets above: built-in drtsans values (2026-03-04). Read a reference run to use what the IOC has now.')
        ttk.Label(ref, textvariable=self.off_source).grid(row=2, column=0, columnspan=10, sticky='w')

        opt = ttk.Frame(f)
        opt.pack(fill='x', pady=4)
        self.mode_var = tk.StringVar(value='band')
        ttk.Radiobutton(opt, text='all six discs spin (six values per step)', variable=self.mode_var, value='band').pack(side='left')
        ttk.Radiobutton(opt, text='only the pair under test spins, the others parked open', variable=self.mode_var, value='parked').pack(side='left', padx=10)
        self.sdd_var = tk.StringVar(value='4.0')
        ttk.Label(opt, text='SDD (m)').pack(side='left', padx=(20, 2))
        ttk.Entry(opt, textvariable=self.sdd_var, width=5).pack(side='left')
        ttk.Button(opt, text='Make plan', command=self.make_plan).pack(side='left', padx=20)
        ttk.Button(opt, text='Save plan…', command=self.save_plan).pack(side='left')
        self.lams_var = tk.StringVar(value=', '.join(f'{x:g}' for x in sc.SLIT_LAMS))
        self.w_var = tk.StringVar(value=f'{sc.SLIT_W:g}')
        self.ws_var = tk.StringVar(value=', '.join(f'{x:g}' for x in sc.WIDTHS))
        self.wl_var = tk.StringVar(value=f'{sc.WIDTH_LAM:g}')
        adv = ttk.Frame(f)
        adv.pack(fill='x')
        for i, (lab, var, wd) in enumerate((('slit-scan λ (Å)', self.lams_var, 24), ('slit width (μs)', self.w_var, 6),
                                            ('width-scan widths (μs)', self.ws_var, 26), ('at λ (Å)', self.wl_var, 5))):
            ttk.Label(adv, text=lab).pack(side='left', padx=(8, 2))
            ttk.Entry(adv, textvariable=var, width=wd).pack(side='left')
        ttk.Label(f, text='Each row: set the six chopper phases (TotalDelay, μs) as listed, take the run, write down the run number.').pack(anchor='w', pady=(6, 2))
        cols = ['step', 'scan', 'pair', 'λ (Å)', 'width (μs)'] + [f'Skf{i} ({sc.BY_IDX[i]})' for i in range(1, 7)] + ['run number']
        self.plan_tv = ttk.Treeview(f, columns=cols, show='headings', height=24)
        for c in cols:
            self.plan_tv.heading(c, text=c)
            self.plan_tv.column(c, width=100 if 'Skf' in c else 70, anchor='e')
        self.plan_tv.pack(fill='both', expand=True)
        self.plan_rows, self.plan_source = [], 'built-in drtsans values (2026-03-04)'
        self.make_plan()

    def offsets(self):
        return {d: float(v.get()) for d, v in self.off_vars.items()}

    def read_reference(self):
        try:
            lm = float(self.ref_lmin.get()) if self.ref_lmin.get().strip() else None
            offs, sdd, lmin, src = sc.reference_offsets(int(self.ref_ipts.get()), int(self.ref_run.get()), lm, self.plan_root.get())
        except Exception as e:  # noqa: BLE001
            messagebox.showerror('slitcal', f'Could not read the reference run: {e}')
            return
        for d in sc.ORDER:
            self.off_vars[d].set(f'{offs[d]:.2f}')
        self.sdd_var.set(f'{sdd:g}')
        diff = ', '.join(f'{d} {offs[d] - sc.REF_OFFSETS[d]:+.1f}' for d in sc.ORDER)
        self.plan_source = src
        self.off_source.set(f'{src}. Difference from the built-in values (μs): {diff}')
        if not self.ipts_var.get():
            self.ipts_var.set(self.ref_ipts.get())
        self.make_plan()

    def make_plan(self):
        try:
            lams = [float(x) for x in self.lams_var.get().split(',') if x.strip()]
            ws = [float(x) for x in self.ws_var.get().split(',') if x.strip()]
            self.plan_rows = sc.make_plan(self.offsets(), lams, ws, float(self.wl_var.get()), float(self.w_var.get()),
                                          mode=self.mode_var.get(), sdd=float(self.sdd_var.get()))
        except ValueError as e:
            messagebox.showerror('slitcal', f'Could not make the plan: {e}')
            return
        self.plan_tv.delete(*self.plan_tv.get_children())
        for r in self.plan_rows:
            vals = []
            for i in range(1, 7):
                if r[f'Skf{i}_action'] == 'PARK OPEN':
                    vals.append('park open')
                else:
                    vals.append(f"{r[f'Skf{i}_TotalDelay']:.2f}" + (' *' if r[f'Skf{i}_action'] == 'SET' else ''))
            self.plan_tv.insert('', 'end', values=[r['step'], r['block'], r['pair'], r['slit_lambda'], r['width_us']] + vals + [''])

    def save_plan(self):
        p = filedialog.askdirectory(title='Folder for plan.csv, worksheet.csv, steps.txt')
        if p:
            sc.write_plan_files(self.plan_rows, self.offsets(), self.plan_source, p)
            self.say(f'plan written to {p}')

    # ------------------------------------------------------------ runs
    def _runs_tab(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text='2  Runs and fit')
        g = ttk.Frame(f)
        g.pack(fill='x')
        self.ipts_var, self.runs_var = tk.StringVar(), tk.StringVar()
        self.root_var = tk.StringVar(value=sc.DATA_ROOT)
        self.work_var = tk.StringVar(value='')          # blank: ./slitcal_IPTS-<ipts>, as on the command line
        for i, (lab, var, wd) in enumerate((('IPTS', self.ipts_var, 8), ('Runs (e.g. 180100-180132, 180140)', self.runs_var, 40),
                                            ('Data root', self.root_var, 22), ('Work folder (blank: slitcal_IPTS-<ipts>)', self.work_var, 40))):
            ttk.Label(g, text=lab).grid(row=i // 2, column=(i % 2) * 2, sticky='e', padx=(10, 3), pady=2)
            ttk.Entry(g, textvariable=var, width=wd).grid(row=i // 2, column=(i % 2) * 2 + 1, sticky='w')
        b = ttk.Frame(f)
        b.pack(fill='x', pady=4)
        ttk.Button(b, text='Extract runs → ASCII and load', command=self.extract_runs).pack(side='left')
        ttk.Button(b, text='Load spectra / NeXus folder…', command=self.add_folder).pack(side='left', padx=6)
        ttk.Button(b, text='Clear', command=self.clear_runs).pack(side='left')
        self.fit_btn = ttk.Button(b, text='Fit', command=self.fit)
        self.fit_btn.pack(side='left', padx=20)
        self.status = tk.StringVar(value='Give the IPTS and the run numbers of the slit scan and the width scan.')
        ttk.Label(b, textvariable=self.status).pack(side='left', padx=8)
        cols = ('run', 'block', 'pair', 'lambda (Å)', 'width (μs)', 'monitor counts', 'detector counts', 'L_det (m)', 'proton charge')
        self.run_tv = ttk.Treeview(f, columns=cols, show='headings', height=15)
        for c in cols:
            self.run_tv.heading(c, text=c)
            self.run_tv.column(c, width=120, anchor='e')
        self.run_tv.pack(fill='both', expand=True, pady=6)
        ttk.Label(f, text='Log').pack(anchor='w')
        self.log = tk.Text(f, height=10, font=('Consolas', 9))
        self.log.pack(fill='both', expand=False)

    def say(self, msg):
        self.q.put(('log', msg))

    def add_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.load_paths([p])

    def clear_runs(self):
        self.runs = []
        self.run_tv.delete(*self.run_tv.get_children())

    def extract_runs(self):
        try:
            ipts, runs = int(self.ipts_var.get()), sc.parse_runs(self.runs_var.get())
        except ValueError:
            messagebox.showerror('slitcal', 'IPTS must be a number and runs like 180100-180132, 180140.')
            return
        folder = os.path.join(self.work_var.get().strip() or f'slitcal_IPTS-{ipts}', 'spectra')
        root = self.root_var.get()
        self.status.set(f'extracting {len(runs)} runs …')

        def work():
            files = sc.extract(ipts, runs, folder, root, progress=self.say)
            self.say(f'ASCII spectra in {folder}')
            self._read_and_queue(files)
        threading.Thread(target=work, daemon=True).start()

    def load_paths(self, paths):
        import glob
        files = []
        for p in paths:
            if os.path.isdir(p):
                dats = sorted(glob.glob(os.path.join(p, '*_spectra.dat')))
                files += dats if dats else sorted(glob.glob(os.path.join(p, '*.nxs.h5')))
            else:
                files.append(p)
        threading.Thread(target=self._read_and_queue, args=(files,), daemon=True).start()

    def _read_and_queue(self, files):
        offs = self.offsets()
        for fp in files:
            try:
                r = sc.read_spectra(fp) if fp.endswith('.dat') else sc.read_run(fp)
                r['cls'] = sc.classify(r, offs)
                self.q.put(('run', r))
            except Exception as e:  # noqa: BLE001
                self.say(f'{os.path.basename(fp)}: could not read ({e})')
        self.q.put(('status', 'loaded'))

    def fit(self):
        good = [r for r in self.runs if r['cls'] is not None]
        if not any(r['cls']['block'] == 'slit' for r in good):
            messagebox.showwarning('slitcal', 'No slit-scan runs loaded.')
            return
        self.fit_btn.state(['disabled'])
        self.status.set('fitting … (a few minutes)')
        offs = self.offsets()

        def work():
            try:
                model = sc.Model()
                slit = [r for r in good if r['cls']['block'] == 'slit']
                width = [r for r in good if r['cls']['block'] == 'width']
                eB, errB, s, chi2 = sc.fit_slit(slit, model, self.say)
                dW = sc.fit_width(width, eB, s, model, self.say) if width else {}
                e = dict(eB)
                for (a, b), dl in dW.items():
                    c = (eB[a] + eB[b]) / 2
                    e[a], e[b] = c - dl / 2, c + dl / 2
                res = dict(reference_offsets=offs, correction_us=e, slit_only_correction_us=eB, slit_fit_sigma_us=errB,
                           width_scan_delta_us={f'{a}-{b}': v for (a, b), v in dW.items()}, emission_scale=s,
                           chi2_per_dof=chi2, new_offsets={d: offs[d] + e[d] for d in sc.ORDER})
                self.q.put(('result', (res, good, model)))
            except Exception as ex:  # noqa: BLE001
                self.q.put(('error', str(ex)))
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ result
    def _result_tab(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text='3  Result')
        left = ttk.Frame(f)
        left.pack(side='left', fill='y')
        cols = ('disc', 'EPICS', 'reference', 'correction', 'new 60 Hz', 'new 30 Hz')
        self.res_tv = ttk.Treeview(left, columns=cols, show='headings', height=7)
        for c in cols:
            self.res_tv.heading(c, text=c)
            self.res_tv.column(c, width=85, anchor='e')
        self.res_tv.pack()
        self.res_info = tk.StringVar(value='No fit yet.')
        ttk.Label(left, textvariable=self.res_info, justify='left', wraplength=500).pack(anchor='w', pady=8)
        ttk.Label(left, text='Figure').pack(anchor='w')
        self.fig_var = tk.StringVar(value='fit_spectra')
        for k, lab in (('fit_spectra', 'slit-scan spectra and fit'), ('width_scan', 'width scan'), ('corrections', 'corrections')):
            ttk.Radiobutton(left, text=lab, variable=self.fig_var, value=k, command=self.show_fig).pack(anchor='w')
        ttk.Button(left, text='Save results…', command=self.save_results).pack(anchor='w', pady=10)
        self.fig_frame = ttk.Frame(f)
        self.fig_frame.pack(side='left', fill='both', expand=True)
        self.canvas = None
        self.figs = {}

    def show_result(self, res, runs, model):
        self.res, self.fit_runs, self.model = res, runs, model
        self.res_tv.delete(*self.res_tv.get_children())
        for d in sc.ORDER:
            self.res_tv.insert('', 'end', values=(d, sc.DISCS[d][0], f"{res['reference_offsets'][d]:.2f}",
                                                  f"{res['correction_us'][d]:+.2f}", f"{res['new_offsets'][d]:.2f}",
                                                  f"{2 * res['new_offsets'][d]:.2f}"))
        dw = ', '.join(f'{k}: {v:+.1f} μs' for k, v in res['width_scan_delta_us'].items()) or 'no width scan'
        self.res_info.set(f"Emission scale (× drtsans delay): {res['emission_scale']:.3f}\n"
                          f"χ²/dof of the slit-scan fit: {res['chi2_per_dof']:.2f}\n"
                          f"Width scan (b − a): {dw}\n"
                          'New offset = reference + correction. Put the same values in hexaSub.c and the drtsans JSON.')
        for fg in self.figs.values():
            plt.close(fg)
        self.figs = sc.figures(runs, res, model, plt)
        self.show_fig()
        self.nb.select(2)

    def show_fig(self):
        if not self.figs:
            return
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.toolbar.destroy()
        fig = self.figs[self.fig_var.get()]
        self.canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.fig_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def save_results(self, folder=None):
        if not self.res:
            return
        folder = folder or filedialog.askdirectory()
        if not folder:
            return
        import json
        import csv
        os.makedirs(folder, exist_ok=True)
        json.dump(self.res, open(os.path.join(folder, 'results.json'), 'w'), indent=1)
        with open(os.path.join(folder, 'offsets.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['disc', 'EPICS_index', 'reference_us', 'correction_us', 'new_offset_us', 'new_30Hz_offset_us'])
            for d in sc.ORDER:
                w.writerow([d, sc.DISCS[d][0], self.res['reference_offsets'][d], round(self.res['correction_us'][d], 2),
                            round(self.res['new_offsets'][d], 2), round(2 * self.res['new_offsets'][d], 2)])
        for k, fg in self.figs.items():
            fg.savefig(os.path.join(folder, k + '.png'), dpi=130, bbox_inches='tight')
        self.say(f'saved {folder}')

    # ------------------------------------------------------------ queue
    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == 'log':
                    self.log.insert('end', val + '\n')
                    self.log.see('end')
                elif kind == 'status':
                    self.status.set(f'{len(self.runs)} runs loaded' if val == 'loaded' else val)
                elif kind == 'run':
                    self.runs.append(val)
                    c = val['cls']
                    pc = val['pcharge']
                    self.run_tv.insert('', 'end', values=(val['run'], c['block'] if c else '—', c['pair'][0][0] if c else '—',
                                                          f"{c['lam']:.2f}" if c and c['lam'] else '—', f"{c['width']:.0f}" if c else '—',
                                                          f"{val['mon'].sum():.0f}", f"{val['det'].sum():.0f}", f"{val['L_det']:.3f}",
                                                          f'{pc:.3g}'))
                elif kind == 'result':
                    self.fit_btn.state(['!disabled'])
                    self.status.set('fit done')
                    self.show_result(*val)
                elif kind == 'error':
                    self.fit_btn.state(['!disabled'])
                    self.status.set('fit failed')
                    messagebox.showerror('slitcal', val)
        except queue.Empty:
            pass
        self.root.after(150, self._poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', nargs='*', help='files or folders to load at start')
    ap.add_argument('--ipts', help='IPTS to fill in (and extract at start, with --runs)')
    ap.add_argument('--runs', help='run numbers to fill in')
    ap.add_argument('--data-root', help='archive root (default /SNS/EQSANS)')
    ap.add_argument('--work', help='work folder for the ASCII spectra')
    ap.add_argument('--ref-run', help='reference run (ordinary 60 Hz band request) to read the offsets in use from')
    ap.add_argument('--fit', action='store_true', help='fit once loaded (scripted test)')
    ap.add_argument('--screenshot', help='save screenshots of the three tabs and quit (scripted test)')
    a = ap.parse_args()
    if sys.platform == 'win32':                       # window coordinates in real pixels (screen grab, sharp text)
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass
    root = tk.Tk()
    app = App(root)
    for var, val in ((app.ipts_var, a.ipts), (app.runs_var, a.runs), (app.root_var, a.data_root), (app.work_var, a.work)):
        if val:
            var.set(val)
    if a.ref_run and a.ipts:
        app.ref_ipts.set(a.ipts)
        app.ref_run.set(a.ref_run)
        if a.data_root:
            app.plan_root.set(a.data_root)
        app.read_reference()
    if a.load:
        app.load_paths(a.load)
    elif a.ipts and a.runs:
        app.extract_runs()
    if a.screenshot:
        from PIL import ImageGrab
        root.attributes('-topmost', True)          # the grab copies the screen: keep this window on top
        stem, ext = os.path.splitext(a.screenshot)

        def grab(tab, tag):
            app.nb.select(tab)
            root.update()
            root.after(800, lambda: _shot(tag))

        def _shot(tag):
            root.lift()
            root.attributes('-topmost', True)
            root.update()
            x, y = root.winfo_rootx(), root.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height())).save(f'{stem}_{tag}{ext}')

        def step():
            if a.fit and not app.res:
                if app.fit_btn.instate(['!disabled']) and app.runs and not getattr(app, '_started', False):
                    app._started = True
                    grab(1, 'runs')
                    root.after(1500, app.fit)
                root.after(2000, step)
                return
            grab(0, 'plan')
            root.after(2500, lambda: grab(2, 'result'))
            root.after(5000, lambda: os._exit(0))
        root.after(4000, step)
    root.mainloop()


if __name__ == '__main__':
    main()
