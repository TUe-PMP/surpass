"""Build release-0.9 measured-excitation and TRPL-observation notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md(r"""
# Experimental excitation and detected TRPL — v0.9

This notebook connects the carrier/escape model to experimental observables:

1. measured or sampled laser-pulse shapes and finite pulse trains;
2. wavelength-dependent optical/detector throughput;
3. a normalized measured instrument-response function (IRF);
4. emitting/collection area, time bins, gain and background;
5. fitting of only amplitude, timing offset and background to a fixed physical trace.

An empirical one- or two-exponential fit is retained only as a descriptive
summary. Its time constants are not automatically bulk or surface lifetimes.
The optical spectrum, absorption, detector response and pulse below are
synthetic examples—replace them with measured arrays before comparison with
your InP data. No experimental result is fitted or reinterpreted here.
""")
code(r"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
root = Path.cwd()
if not (root/'src').exists(): root = root.parent
sys.path.insert(0, str(root/'src'))
import surpass
print('surpass version:', surpass.__version__)
print('surpass loaded from:', Path(surpass.__file__).resolve())
if surpass.__version__ != '1.0.0':
    raise ImportError('Notebook 09 requires the flat v1.0.0 source tree; replace project files and restart the kernel.')
from surpass import (
    SampledPulse, PulseTrain, InstrumentResponse, SpectralResponse, DetectionModel,
    fit_detection_nuisance, fit_empirical_exponentials,
    BulkModel, BulkTransportTable, TimeDependent1DSolver, Wafer,
    InterfaceDefectModel, InterfaceChargeModel, SurfaceSRHModel, SurfaceBoundaryTable,
    EmissionSpectrum, EmissionBoundary, SlabEmissionModel,
    OpticalStack, Layer, get_optical_material,
)
try:
    import scienceplots
    plt.style.use(['science', 'notebook', 'no-latex'])
except ImportError:
    plt.style.use('default')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
""")
md(r"""
## 1. Sampled pulse and repetition-rate bookkeeping

`SampledPulse` normalizes the continuous PCHIP interpolant—not merely the
trapezoid through its samples. `density_cm3` is therefore the full generated
carrier dose per pulse in every spatial cell. Samples outside the measured
support are exactly zero. Subtract laser-monitor background before constructing
the pulse and include enough baseline that its support is physically complete.

For a pulse train, specify the start time and either a finite number of pulses
or `pulse_count=None`. The solver automatically limits its step using the
sample spacing. Repetition can cause carrier accumulation; a single-pulse
measurement represents isolated excitation only when the sample returns close
enough to equilibrium before the next pulse.
""")
code(r"""
relative_time = np.linspace(-1e-9, 2e-9, 25)
# Synthetic asymmetric laser monitor: fast Gaussian plus weak positive tail.
pulse_monitor = np.exp(-.5*(relative_time/.28e-9)**2)
pulse_monitor += .12*np.exp(-np.maximum(relative_time, 0)/.65e-9)*(relative_time >= 0)
unit_pulse = SampledPulse(np.array([1.]), relative_time, pulse_monitor,
                         source='SYNTHETIC laser monitor')
dense = np.linspace(relative_time[0], relative_time[-1], 5001)
normalization = np.trapezoid([unit_pulse(t)[0] for t in dense], dense)
train = PulseTrain(unit_pulse, period_s=20e-9, first_pulse_s=0., pulse_count=3)
train_time = np.linspace(-2e-9, 65e-9, 4001)
train_shape = np.array([train(t)[0] for t in train_time])
train_dose = np.trapezoid(train_shape, train_time)
print('Single-pulse temporal integral:', normalization)
print('Three-pulse temporal integral:', train_dose)
assert abs(normalization-1) < 2e-6
# This diagnostic uses the coarse display grid; the solver evaluates the
# normalized continuous interpolant and checks generated dose independently.
assert abs(train_dose-3) < 2e-4
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
axes[0].plot(relative_time*1e9, unit_pulse.temporal_shape*1e-9, 'o-', label='normalized interpolant')
axes[0].set(xlabel='Relative time (ns)', ylabel='Temporal density (ns$^{-1}$)')
axes[1].plot(train_time*1e9, train_shape*1e-9)
axes[1].set(xlabel='Time (ns)', ylabel='Pulse train (ns$^{-1}$)')
fig.tight_layout(); plt.show()
""")
md(r"""
## 2. IRF validation

The IRF is a delay probability density normalized to unit area. It redistributes
detected events in time but does not change their number when the source and
detection windows contain all tails. The convolution below recovers the known
width $\sigma=\sqrt{\sigma_s^2+\sigma_h^2}$ for two Gaussians.
""")
code(r"""
irf_delay = np.linspace(-3e-9, 3e-9, 121)
irf_sigma = .55e-9
irf = InstrumentResponse(irf_delay, np.exp(-.5*(irf_delay/irf_sigma)**2),
                         source='SYNTHETIC Gaussian IRF')
check_time = np.linspace(-8e-9, 8e-9, 801)
source_sigma = .8e-9
source = np.exp(-.5*(check_time/source_sigma)**2)/(np.sqrt(2*np.pi)*source_sigma)
broadened = irf.convolve(check_time, source, check_time)
combined_sigma = np.sqrt(source_sigma**2+irf_sigma**2)
exact = np.exp(-.5*(check_time/combined_sigma)**2)/(np.sqrt(2*np.pi)*combined_sigma)
mask = abs(check_time) < 3e-9  # avoid magnifying tiny truncation errors in far tails
irf_error = np.max(abs(broadened[mask]/exact[mask]-1))
print('Gaussian-convolution maximum central relative error:', irf_error)
assert irf_error < 2e-4
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(check_time*1e9, source*1e-9, label='source')
ax.plot(check_time*1e9, broadened*1e-9, label='convolved')
ax.plot(check_time*1e9, exact*1e-9, '--', label='analytic')
ax.set(xlabel='Time (ns)', ylabel='Normalized temporal density (ns$^{-1}$)'); ax.legend()
fig.tight_layout(); plt.show()
""")
md(r"""
## 3. Editable synthetic InP experiment

The semiconductor example uses n-InP, positive external charge, instantaneous
interface response and local photon recycling. The pulse fluence, 20-ns period,
NA, emitting area and response curves are illustrative. `emitting_area_cm2`
must describe the portion of the simulated per-area flux imaged onto the
detector; it is not necessarily either the laser spot or aperture area alone.

The spectral response can combine calibrated filter transmission, spectrometer
throughput and detector quantum efficiency. Keep the NA in the optical escape
calculation so it is not counted twice. Electronic gain converts a detected
photon event to reported counts and should be independently calibrated if an
absolute PL prediction is intended.
""")
code(r"""
thickness_cm = 620e-4
wafer = Wafer('InP', donor_cm3=8.5e18, thickness_um=620)
bulk = BulkModel(wafer, tau_srh_s=100e-9)
table = BulkTransportTable.from_bulk_model(bulk, 1., 1e20, points=121)
surface = SurfaceSRHModel(bulk, 4.2e12, InterfaceDefectModel(energy_points=101),
    interface_charge=InterfaceChargeModel(.5, .02))
boundary = SurfaceBoundaryTable.from_surface_model(surface, 1., 1e20, points=81)

air, inp = get_optical_material('air'), get_optical_material('InP_demo')
alox, pox = get_optical_material('AlOx_demo'), get_optical_material('POx_demo')
pump_stack = OpticalStack(air, [Layer(alox, 10), Layer(pox, 10)], inp)
emission_front = EmissionBoundary(air, [Layer(pox, 10), Layer(alox, 10)])
emission_rear = EmissionBoundary(air)
edges = np.r_[0, np.geomspace(2e-8, 5e-4, 34), np.linspace(5e-4, thickness_cm, 17)[1:]]
widths, depth = np.diff(edges), (edges[1:]+edges[:-1])/2

emission_wavelength = np.linspace(875, 975, 11)
emission_spectrum = EmissionSpectrum(
    emission_wavelength, np.exp(-.5*((emission_wavelength-920)/12)**2),
    source='SYNTHETIC internal spectrum per nm')
alpha_active = 500*np.exp((920-emission_wavelength)/12)  # SYNTHETIC cm^-1
alpha_parasitic = 20.                                    # SYNTHETIC cm^-1
fates = SlabEmissionModel(inp, emission_front, emission_rear,
    alpha_active, alpha_parasitic).solve(
        thickness_cm, depth, emission_spectrum, angular_points=48, front_na=.5)
p_recycle = fates.averaged()['active_reabsorption']

detector_response = SpectralResponse(
    [850., 890., 920., 950., 1000.], [.05, .25, .62, .48, .08],
    source='SYNTHETIC filters × detector QE')
fluence_j_cm2 = 1e-8
spatial_dose = pump_stack.generation_cell_average_cm3_s(514., 1., edges)*fluence_j_cm2
measured_pulse = SampledPulse(spatial_dose, relative_time, pulse_monitor,
                              source='SYNTHETIC sampled laser pulse')
pulse_train = PulseTrain(measured_pulse, period_s=20e-9,
                         first_pulse_s=0., pulse_count=3)
simulation_time = np.linspace(-5e-9, 75e-9, 401)
result = TimeDependent1DSolver(table).solve(
    thickness_cm, simulation_time, 0., boundary, 1e5,
    generation=pulse_train, cell_edges_cm=edges,
    recycling_probability=p_recycle, rtol=2e-6)
intrinsic_rate = table.evaluate(result.delta_n_cm3)['r_rad']
intrinsic_rate = np.where(result.delta_n_cm3 > 0, intrinsic_rate, 0.)
collected_flux = fates.response_weighted_flux(
    intrinsic_rate, widths, detector_response, fate='front_collected')
print('Generated dose / three nominal absorbed doses:',
      result.cumulative_generated_cm2[-1]/(3*np.dot(spatial_dose, widths)))
print('Maximum carrier-inventory error:', np.max(abs(result.inventory_relative_error)))
assert abs(result.cumulative_generated_cm2[-1]/(3*np.dot(spatial_dose, widths))-1) < 2e-4
assert np.max(abs(result.inventory_relative_error)) < 2e-5
""")
md(r"""
## 4. From collected photon flux to synthetic measured counts

The detector model convolves the continuous source with the IRF and integrates
over bin edges. Background is a rate in counts/s. Here a scale is chosen only
to generate a well-conditioned synthetic dataset; it stands in for uncalibrated
throughput. The subsequent fit is allowed to adjust only this scale, the timing
offset and a constant background—the carrier trace itself remains fixed.
""")
code(r"""
bin_edges = np.arange(-4e-9, 72.0001e-9, .25e-9)
bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
detector = DetectionModel(emitting_area_cm2=1e-6, irf=irf,
                          electronic_gain_counts_per_event=1.)
unscaled = detector.expected_counts(simulation_time, collected_flux, bin_edges)
true_scale = 900/max(unscaled)
true_shift = .35e-9
true_background_per_bin = 2.
true_background_rate = true_background_per_bin/np.mean(np.diff(bin_edges))
expected = detector.expected_counts(simulation_time, collected_flux, bin_edges,
    scale=true_scale, time_shift_s=true_shift,
    background_rate_counts_s=true_background_rate)
rng = np.random.default_rng(20260919)
measured = rng.poisson(expected)
fit = fit_detection_nuisance(detector, simulation_time, collected_flux,
    bin_edges, measured, initial_scale=.7*true_scale,
    initial_time_shift_s=0., initial_background_rate_counts_s=.5*true_background_rate,
    shift_bounds_s=(-1e-9, 1e-9))
display(pd.DataFrame({
    'quantity': ['scale', 'time shift (ns)', 'background (counts/bin)', f'reduced {fit.statistic_name}'],
    'true': [true_scale, true_shift*1e9, true_background_per_bin, np.nan],
    'fitted': [fit.scale, fit.time_shift_s*1e9,
               fit.background_rate_counts_s*np.mean(np.diff(bin_edges)), fit.reduced_chi_square],
}))
assert fit.success
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})
axes[0].step(bin_centers*1e9, measured, where='mid', color='black', lw=.8, label='synthetic counts')
axes[0].plot(bin_centers*1e9, fit.expected_counts, color='tab:red', label='fixed physical trace + fitted nuisance')
axes[0].set(ylabel='Counts per 0.25-ns bin', yscale='log', ylim=(.7, None)); axes[0].legend()
axes[1].axhline(0, color='0.5', lw=.8)
axes[1].plot(bin_centers*1e9, fit.standardized_residual, '.', ms=3)
axes[1].set(xlabel='Time (ns)', ylabel='Residual / σ')
fig.tight_layout()
output_dir = root/'examples'/'output'; output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir/'v09_detected_trpl.png', dpi=180, bbox_inches='tight')
plt.show()
""")
md(r"""
## 5. Repetition-rate accumulation and empirical fit-window dependence

The carrier inventory immediately before successive pulses reveals whether the
experiment has returned to equilibrium. The empirical fits below summarize the
noiseless detected rate after the final pulse. Changing the fit window can
change one- and two-exponential time constants even though the underlying
physical simulation is unchanged. A good biexponential fit therefore does not
establish two microscopic recombination channels.
""")
code(r"""
rows = []
for pulse_time in (0., 20e-9, 40e-9):
    index = np.argmin(abs(simulation_time-(pulse_time-0.4e-9)))
    rows.append({'pulse at (ns)': pulse_time*1e9,
                 'pre-pulse sheet excess (cm-2)': result.sheet_excess_cm2[index]})
display(pd.DataFrame(rows))
rate_at_bins = detector.event_rate(simulation_time, collected_flux, bin_centers,
                                   time_shift_s=true_shift, scale=true_scale)
for start_ns in (41., 45.):
    window = (start_ns*1e-9, 64e-9)
    one = fit_empirical_exponentials(bin_centers, rate_at_bins, components=1,
                                     fit_window_s=window)
    two = fit_empirical_exponentials(bin_centers, rate_at_bins, components=2,
                                     fit_window_s=window)
    print(f'window {start_ns:.0f}–64 ns: one-exp tau={one.lifetimes_s[0]*1e9:.3g} ns; '
          f'two-exp taus={two.lifetimes_s*1e9} ns')
fig, ax = plt.subplots(figsize=(7, 3.5))
mask = (bin_centers >= 40e-9)&(bin_centers <= 65e-9)
two = fit_empirical_exponentials(bin_centers, rate_at_bins, components=2,
                                 fit_window_s=(41e-9, 64e-9))
ax.semilogy(bin_centers[mask]*1e9, rate_at_bins[mask], label='physical detected rate')
ax.semilogy(bin_centers[two.fit_mask]*1e9, two.fitted_signal[two.fit_mask], '--', label='descriptive biexponential')
ax.set(xlabel='Time (ns)', ylabel='Detected event rate (s$^{-1}$)'); ax.legend()
fig.tight_layout(); plt.show()
""")
md(r"""
## 6. How to replace the synthetic inputs

```python
laser = pd.read_csv('laser_monitor.csv')
pulse = SampledPulse(spatial_dose,
                     laser['time_ns'].to_numpy()*1e-9,
                     laser['background_subtracted_signal'].to_numpy(),
                     source='measured laser monitor, file/date/settings')

irf_data = pd.read_csv('detector_irf.csv')
irf = InstrumentResponse(irf_data['delay_ns']*1e-9,
                         irf_data['background_subtracted_counts'],
                         source='measured IRF, detector/filter/settings')

qe = pd.read_csv('spectral_response.csv')
response = SpectralResponse(qe['wavelength_nm'], qe['total_efficiency'],
                            source='calibrated filter × optics × detector response')
```

Keep raw files and preprocessing documented. In particular:

* distinguish pulse energy at the sample from a monitor's arbitrary amplitude;
* use the actual repetition period and simulate enough pulses to test pile-up;
* measure the IRF with the same detector, timing electronics and spectral path;
* distinguish illuminated, emitting, imaged and detector-active areas;
* do not use the same optical loss once in the escape model and again in the
  spectral-response curve;
* include the complete IRF and simulation tails—truncation loses events;
* use independent uncertainties for analog PL; `sqrt(counts)` is only a simple
  Poisson-counting approximation;
* fit physical parameters only after the optical and instrument nuisance
  parameters are independently constrained where possible.

Still omitted: nonlocal photon reabsorption, dynamic traps, detector dead time,
afterpulsing, wavelength-dependent IRF, spatial PSF/finite spot transport, and
full Bayesian uncertainty propagation.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
notebook.metadata['language_info'] = {'name': 'python', 'version': '3.10+'}
nbf.write(notebook, ROOT/'notebooks'/'09_experimental_trpl_detection.ipynb')
print('Built release-0.9 notebook')
