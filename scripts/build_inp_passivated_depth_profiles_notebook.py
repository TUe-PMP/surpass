"""Build notebook 12: passivated InP band, carrier, generation and loss profiles."""
from pathlib import Path
import json
import uuid


ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({'cell_type':'markdown','id':uuid.uuid4().hex[:8],
                  'metadata':{},'source':text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({'cell_type':'code','execution_count':None,
                  'id':uuid.uuid4().hex[:8],'metadata':{},'outputs':[],
                  'source':text.strip().splitlines(keepends=True)})


md(r"""
# Passivated n- and p-InP: spatial bands, carriers, generation and recombination

This notebook constructs a depth-resolved physical picture of the illustrative
passivated PO$_x$/AlO$_x$/InP state used in the manuscript modelling. Each
wafer column follows the causal chain

$$
\text{band bending and quasi-Fermi levels}
\rightarrow n(z),p(z),G(z)
\rightarrow R_{rad}(z),R_{Auger}(z),U_s.
$$

The figure combines two rigorously calculated but scale-separated models:

- the nanometre-scale space-charge region is reconstructed from Poisson's
  first integral at the converged surface injection;
- the full 620-$\mu$m wafer uses the nonlinear ambipolar finite-volume solver.

The present transport solver assumes local charge neutrality in the
quasi-neutral bulk and does not solve a monolithic drift--diffusion--Poisson
system. Quasi-Fermi levels are taken as flat across the thin space-charge
region and vary through the quasi-neutral wafer according to the calculated
carrier profile. The resulting plot is a physically consistent composite
reconstruction, not a full device-level electrostatic solution.
""")

code(r"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
from IPython.display import display

root = Path.cwd()
if not (root/'src').exists():
    root = root.parent
sys.path.insert(0, str(root/'src'))

import surpass
from surpass import (
    BulkModel, BulkTransportTable, InterfaceDefectModel, Layer, OpticalStack,
    SteadyState1DSolver, SurfaceBoundaryTable, SurfaceSRHModel, Wafer,
    get_optical_material, reconstruct_band_profile,
)

print('surpass version:', surpass.__version__)
print('loaded from:', Path(surpass.__file__).resolve())
if tuple(map(int, surpass.__version__.split('.')[:3])) < (0,9,0):
    raise ImportError('This notebook requires surpass 1.0.0 or newer.')

try:
    import scienceplots
    plt.style.use(['science','notebook','no-latex'])
except ImportError:
    plt.style.use('default')
plt.rcParams.update({
    'figure.dpi':120, 'savefig.dpi':300, 'font.size':10,
    'axes.spines.top':True, 'axes.spines.right':True,
})
output_dir = root/'examples'/'output'
output_dir.mkdir(parents=True,exist_ok=True)
""")

md(r"""
## 1. Editable passivated-state assumptions

The nominal state uses $D_{it}=3\times10^{10}$ eV$^{-1}$cm$^{-2}$ and
$Q_f/q=+2\times10^{12}$ cm$^{-2}$. These coordinates are illustrative and
must not be described as fitted or extracted values. The electron and hole
capture cross sections retain the literature-motivated InP starting values.

The optical stack explicitly contains AlO$_x$ and PO$_x$. Their bundled optical
constants and the 10-nm film thicknesses are placeholders until sample-specific
ellipsometry is inserted.

The paper baseline uses an injection-independent bulk SRH lifetime of 5 ns and
an active photon-recycling probability of 0.93. These are representative
sensitivity parameters motivated by the measured TRPL timescale and the
spectral reabsorption estimate, respectively; they are not unique fitted values.
""")

code(r"""
WAVELENGTH_NM = 514.0
THICKNESS_UM = 620.0
THICKNESS_CM = THICKNESS_UM*1e-4
DIT_PASSIVATED = 3e10
QF_PASSIVATED_CM2 = 2e12
SIGMA_N_CM2 = 1e-14
SIGMA_P_CM2 = 1e-13
VTH_N_CM_S = 4.13e7
VTH_P_CM_S = 1.51e7
REAR_S_CM_S = 1e5
ENABLE_BGN = False
BULK_SRH_LIFETIME_S = 5e-9
PHOTON_RECYCLING_PROBABILITY = 0.93

wafer_settings = {
    'n-InP':dict(wafer=Wafer('InP',donor_cm3=8.5e18,thickness_um=THICKNESS_UM),
                 intensity_w_cm2=1.0,color='tab:blue'),
    'p-InP':dict(wafer=Wafer('InP',acceptor_cm3=5.1e18,thickness_um=THICKNESS_UM),
                 intensity_w_cm2=13.0,color='tab:red'),
}

defects = InterfaceDefectModel(
    DIT_PASSIVATED,SIGMA_N_CM2,SIGMA_P_CM2,VTH_N_CM_S,VTH_P_CM_S,
    shape='constant',energy_points=201,
)
print(defects.source)

air=get_optical_material('air')
inp_optical=get_optical_material('InP_demo')
pox_optical=get_optical_material('POx_demo')
alox_optical=get_optical_material('AlOx_demo')
stack=OpticalStack(
    air,
    [Layer(alox_optical,10.0,'AlOx'),Layer(pox_optical,10.0,'POx')],
    inp_optical,
)
optical=stack.solve(WAVELENGTH_NM)
display(pd.DataFrame([{
    'wavelength (nm)':WAVELENGTH_NM,
    'reflectance':optical.reflectance,
    'substrate entry fraction':optical.substrate_entry_fraction,
    'absorption depth (nm)':optical.absorption_depth_nm,
    'Dit (eV^-1 cm^-2)':DIT_PASSIVATED,
    'Qf/q (cm^-2)':QF_PASSIVATED_CM2,
}]))
""")

md(r"""
## 2. Solve the full-wafer steady state

The mesh resolves the strongly front-loaded optical generation before
transitioning to a coarser wafer-scale grid. The nominal calculation uses a
5-ns, injection-independent bulk SRH lifetime and an active photon-recycling
probability of 0.93. Radiative and Auger recombination are retained. The
rear-surface rate is deliberately finite and is reported separately.
""")

code(r"""
edges_cm=np.r_[0.0,np.geomspace(1e-8,5e-4,120),
               np.linspace(5e-4,THICKNESS_CM,121)[1:]]
solutions={}
summary=[]

for label,settings in wafer_settings.items():
    bulk=BulkModel(settings['wafer'],enable_bgn=ENABLE_BGN,
                   tau_srh_s=BULK_SRH_LIFETIME_S)
    transport=BulkTransportTable.from_bulk_model(bulk,1.0,1e21,points=181)
    solver=SteadyState1DSolver(transport)
    generation=stack.generation_cell_average_cm3_s(
        WAVELENGTH_NM,settings['intensity_w_cm2'],edges_cm)
    surface=SurfaceSRHModel(
        bulk,QF_PASSIVATED_CM2,defects,electrostatics='self_consistent',
        driving_force='excess_product')
    boundary=SurfaceBoundaryTable.from_surface_model(surface,1.0,1e21,points=101)
    result=solver.solve(
        THICKNESS_CM,generation,boundary,REAR_S_CM_S,
        cell_edges_cm=edges_cm,max_nfev=1800,
        recycling_probability=PHOTON_RECYCLING_PROBABILITY)
    if not result.converged:
        raise RuntimeError(f'{label} transport solution did not converge')
    surface_state=surface.state(result.front_surface_delta_cm3)
    profile=reconstruct_band_profile(
        surface,result.front_surface_delta_cm3,points=1201,tail_fraction=1e-7)
    solutions[label]=dict(bulk=bulk,transport=transport,surface=surface,result=result,
                          surface_state=surface_state,profile=profile)
    summary.append({
        'wafer':label,
        'average delta n (cm^-3)':result.average_delta_n_cm3,
        'surface delta n (cm^-3)':result.front_surface_delta_cm3,
        'psi_s (V)':surface_state.psi_surface_v,
        'n_s (cm^-3)':surface_state.n_surface_cm3,
        'p_s (cm^-3)':surface_state.p_surface_cm3,
        'front Seff (cm/s)':result.front_effective_s_cm_s,
        'Us (cm^-2 s^-1)':result.front_surface_recombination_cm2_s,
        'radiative fraction':result.bulk_radiative_recombination_cm2_s/result.generated_flux_cm2_s,
        'Auger fraction':result.bulk_auger_recombination_cm2_s/result.generated_flux_cm2_s,
        'front-surface fraction':result.front_surface_recombination_cm2_s/result.generated_flux_cm2_s,
        'rear-surface fraction':result.rear_surface_recombination_cm2_s/result.generated_flux_cm2_s,
        'balance error':result.relative_balance_error,
    })

summary=pd.DataFrame(summary)
display(summary)
assert max(abs(summary['balance error'])) < 1e-5
""")

md(r"""
## 3. Composite depth profiles

The shared depth axis is symmetric-logarithmic: it is linear within 1 nm of the
surface and logarithmic through the remainder of the wafer. This makes the
space-charge region and the optical/transport profile visible in one figure.

The surface recombination rate $U_s$ is a sheet flux in cm$^{-2}$s$^{-1}$,
whereas the bulk rates are volumetric in cm$^{-3}$s$^{-1}$. It is therefore
reported in a boxed surface annotation and is not converted into an arbitrary
volumetric spike.
""")

code(r"""
def qfl_profiles(bulk,result,delta_profile=None):
    delta=result.delta_n_cm3 if delta_profile is None else delta_profile
    states=[bulk.injection(max(float(d),1e-300)) for d in delta]
    eg=bulk.equilibrium().eg_effective_ev
    vt=bulk.wafer.temperature_k*8.617333262145e-5
    efn=np.array([eg+vt*s.eta_n for s in states])
    efp=np.array([-vt*s.eta_p for s in states])
    return efn,efp

def resolved_deep_tail(item):
    # Continue the unresolved far tail with its low-injection asymptote.
    result,table=item['result'],item['transport']
    delta=result.delta_n_cm3.copy()
    x=result.depth_cm
    # Anchor at the final finely resolved cell, before the wafer-scale mesh.
    fine=np.flatnonzero(result.cell_width_cm < 1e-4)  # narrower than 1 um
    anchor=int(fine[-1])
    low=table.evaluate(np.array([1.0]))
    inverse_tau=float((1-PHOTON_RECYCLING_PROBABILITY)*low['r_rad'][0]
                      +low['r_auger'][0]+low['r_srh'][0])
    low_length=np.sqrt(float(low['d_amb'][0])/inverse_tau)
    exponent=np.clip(-(x[anchor:]-x[anchor])/low_length,-745.0,0.0)
    delta[anchor:]=delta[anchor]*np.exp(exponent)
    return delta,low_length

for item in solutions.values():
    item['resolved_delta'],item['low_injection_length_cm']=resolved_deep_tail(item)

fig,axes=plt.subplots(3,2,figsize=(13.2,10.5),sharex='col',
                      constrained_layout=True)
generation_twins=[]

for col,(label,settings) in enumerate(wafer_settings.items()):
    item=solutions[label]
    bulk,result,profile=(item['bulk'],item['result'],item['profile'])
    equilibrium=bulk.equilibrium()
    x_um=result.depth_cm*1e4
    x_scr_um=profile.depth_cm*1e4
    resolved_delta=item['resolved_delta']
    efn_bulk,efp_bulk=qfl_profiles(bulk,result,resolved_delta)

    # Top: Poisson reconstruction in the SCR, followed by flat bulk bands.
    ax=axes[0,col]
    bulk_after_scr=x_um>x_scr_um[-1]
    x_band=np.r_[x_scr_um,x_um[bulk_after_scr]]
    ec_full=np.r_[profile.ec_ev,
                  np.full(bulk_after_scr.sum(),equilibrium.eg_effective_ev)]
    ev_full=np.r_[profile.ev_ev,np.zeros(bulk_after_scr.sum())]
    ax.plot(x_band,ec_full,color='tab:blue',lw=1.8,label=r'$E_C$')
    ax.plot(x_band,ev_full,color='tab:green',lw=1.8,label=r'$E_V$')
    # The transport-derived quasi-Fermi levels already span the full wafer;
    # plotting the SCR constants as well would duplicate the same curves.
    ax.plot(x_um,efn_bulk,color='tab:orange',ls='--',lw=1.6,label=r'$E_{Fn}$')
    ax.plot(x_um,efp_bulk,color='tab:red',ls='--',lw=1.6,label=r'$E_{Fp}$')
    ax.set(ylabel='Energy relative to bulk $E_V$ (eV)',title=label)
    ax.legend(fontsize=10.5,frameon=True,fancybox=True,framealpha=.85,
              facecolor='white',edgecolor='.65',ncol=2)
    ax.text(.98,.05,
            rf'$Q_f/q=+{QF_PASSIVATED_CM2/1e12:g}\times10^{{12}}$ cm$^{{-2}}$'+'\n'
            rf'$\psi_s={item["surface_state"].psi_surface_v:+.3f}$ V',
            transform=ax.transAxes,ha='right',va='bottom',fontsize=10,
            bbox=dict(facecolor='white',alpha=.82,edgecolor='.7',
                      boxstyle='round,pad=.25'))

    # Middle: stitch the SCR populations to the quasi-neutral transport result.
    ax=axes[1,col]
    n_transport=equilibrium.n0_cm3+resolved_delta
    p_transport=equilibrium.p0_cm3+resolved_delta
    x_carrier=np.r_[x_scr_um,x_um[bulk_after_scr]]
    n_full=np.r_[profile.n_cm3,n_transport[bulk_after_scr]]
    p_full=np.r_[profile.p_cm3,p_transport[bulk_after_scr]]
    ax.semilogy(x_carrier,n_full,color='tab:blue',lw=1.8,label=r'$n(z)$')
    ax.semilogy(x_carrier,p_full,color='tab:red',lw=1.8,label=r'$p(z)$')
    ax.axhline(equilibrium.n0_cm3,color='tab:blue',ls=':',lw=.9)
    ax.axhline(equilibrium.p0_cm3,color='tab:red',ls=':',lw=.9)
    ax.set_ylabel(r'Carrier concentration (cm$^{-3}$)')
    axg=ax.twinx()
    generation_floor=result.generation_cm3_s.max()*1e-12
    generation_visible=result.generation_cm3_s>=generation_floor
    axg.semilogy(x_um[generation_visible],result.generation_cm3_s[generation_visible],
                 color='tab:green',lw=1.25,
                 alpha=.85,label=r'$G(z)$')
    axg.set_ylim(generation_floor,result.generation_cm3_s.max()*2)
    axg.set_ylabel(r'Generation rate (cm$^{-3}$ s$^{-1}$)',color='tab:green')
    axg.tick_params(axis='y',colors='tab:green')
    generation_twins.append(axg)
    h1,l1=ax.get_legend_handles_labels(); h2,l2=axg.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,fontsize=10,frameon=True,fancybox=True,
              framealpha=.85,facecolor='white',edgecolor='.65',loc='best')

    # Bottom: physically commensurate volumetric bulk channels.
    ax=axes[2,col]
    rate_max=max(result.r_rad_cm3_s.max(),result.r_auger_cm3_s.max(),
                 result.r_srh_cm3_s.max())
    rate_floor=rate_max*1e-10
    rad_visible=result.r_rad_cm3_s>=rate_floor
    auger_visible=result.r_auger_cm3_s>=rate_floor
    ax.loglog(x_um[rad_visible],result.r_rad_cm3_s[rad_visible],color='tab:green',
              lw=1.7,label=r'$R_{rad}(z)$')
    ax.loglog(x_um[auger_visible],result.r_auger_cm3_s[auger_visible],color='tab:orange',
              lw=1.7,label=r'$R_{Auger}(z)$')
    if (result.bulk_srh_recombination_cm2_s/result.generated_flux_cm2_s)>1e-12:
        srh_visible=result.r_srh_cm3_s>=rate_floor
        ax.loglog(x_um[srh_visible],result.r_srh_cm3_s[srh_visible],color='tab:purple',
                  lw=1.4,label=r'$R_{SRH,bulk}(z)$')
    ax.set_ylim(rate_floor,rate_max*2)
    ax.axvline(0,color='black',lw=1.2)
    ax.text(.03,.95,
            rf'$U_s={result.front_surface_recombination_cm2_s:.2e}$ cm$^{{-2}}$ s$^{{-1}}$'+'\n'
            rf'$S_{{eff}}={result.front_effective_s_cm_s:.2e}$ cm s$^{{-1}}$',
            transform=ax.transAxes,ha='left',va='top',fontsize=10,
            bbox=dict(facecolor='white',alpha=.85,edgecolor='.65',
                      boxstyle='round,pad=.3'))
    ax.set(xlabel=r'Depth into InP ($\mu$m)',
           ylabel=r'Bulk excess recombination (cm$^{-3}$ s$^{-1}$)')
    ax.legend(fontsize=10,frameon=True,fancybox=True,framealpha=.85,
              facecolor='white',edgecolor='.65',loc='lower left')

for ax in axes.ravel():
    ax.set_xscale('symlog',linthresh=1e-3,linscale=1.0)
    ax.set_xlim(0,THICKNESS_UM)

fig.savefig(output_dir/'paper_inp_si_composite_depth_profiles.png',bbox_inches='tight')
""")

md(r"""
## 4. Surface-region zoom

The shared full-wafer axis necessarily compresses the first few nanometres.
This companion figure shows the same calculated space-charge region directly.
It is suitable as a supplementary panel or as the basis of a simplified
main-text schematic.
""")

code(r"""
fig,axes=plt.subplots(2,2,figsize=(11.5,7.2),sharex='col',
                      constrained_layout=True)
for col,(label,settings) in enumerate(wafer_settings.items()):
    item=solutions[label]; profile=item['profile']; state=item['surface_state']
    x_nm=profile.depth_cm*1e7
    ax=axes[0,col]
    ax.plot(x_nm,profile.ec_ev,label=r'$E_C$',lw=1.8)
    ax.plot(x_nm,profile.ev_ev,label=r'$E_V$',lw=1.8)
    ax.plot(x_nm,profile.efn_ev,'--',label=r'$E_{Fn}$',lw=1.5)
    ax.plot(x_nm,profile.efp_ev,'--',label=r'$E_{Fp}$',lw=1.5)
    ax.set(title=label,ylabel='Energy relative to bulk $E_V$ (eV)')
    ax.legend(fontsize=10,frameon=True,fancybox=True,framealpha=.85,
              facecolor='white',edgecolor='.65',ncol=2)
    ax=axes[1,col]
    ax.semilogy(x_nm,profile.n_cm3,label=r'$n(z)$',color='tab:blue',lw=1.8)
    ax.semilogy(x_nm,profile.p_cm3,label=r'$p(z)$',color='tab:red',lw=1.8)
    ax.set(xlabel='Depth into InP (nm)',ylabel=r'Carrier concentration (cm$^{-3}$)')
    ax.legend(fontsize=10,frameon=True,fancybox=True,framealpha=.85,
              facecolor='white',edgecolor='.65')
    xmax=max(profile.potential_99_depth_cm*1e7*1.5,2.0)
    axes[0,col].set_xlim(0,xmax)
fig.savefig(output_dir/'paper_inp_si_surface_region_zoom.png',bbox_inches='tight')
""")

md(r"""
## 5. Broken-axis view: surface region and deep-wafer relaxation

The surface zoom is retained because it resolves the field-effect passivation
region. The figure below adds a second, logarithmic depth segment extending to
the rear of the wafer. Both segments use micrometres, and diagonal marks denote
the change from a linear near-surface axis to a logarithmic deep-wafer axis.

The dotted grey line in the upper panels is the equilibrium bulk Fermi level.
The quasi-Fermi levels merge with it only if the calculated excess carrier
population has relaxed sufficiently within the finite wafer. The model is not
forced to show equilibrium at 620 $\mu$m: a remaining rear-side splitting is a
physical result of the illumination, diffusion length and rear boundary.

Once the excess density is too small to affect any integrated balance, the
globally scaled finite-volume solution no longer resolves its many-decade tail
reliably. Beyond the last sub-micrometre transport cell, the displayed profile
is therefore continued with the analytic low-injection diffusion asymptote,
$\Delta n\propto\exp(-z/L_D)$. This removes a numerical plateau while retaining
the fully nonlinear result throughout the generated and recombination-active
region.
""")

code(r"""
SURFACE_BREAK_UM=0.025  # 25 nm retained as the linear near-surface segment
fig=plt.figure(figsize=(17.0,7.4))
grid=fig.add_gridspec(
    2,5,width_ratios=[1.0,1.65,.24,1.0,1.65],
    wspace=0.08,hspace=0.08,
)
axes=np.empty((2,4),dtype=object)
for row in range(2):
    axes[row,0]=fig.add_subplot(grid[row,0])
    axes[row,1]=fig.add_subplot(grid[row,1])
    axes[row,2]=fig.add_subplot(grid[row,3])
    axes[row,3]=fig.add_subplot(grid[row,4])

def add_break_marks(left,right):
    size=.018
    style=dict(color='black',clip_on=False,lw=1.0)
    left.plot((1-size,1+size),(-size,+size),transform=left.transAxes,**style)
    left.plot((1-size,1+size),(1-size,1+size),transform=left.transAxes,**style)
    right.plot((-size,+size),(-size,+size),transform=right.transAxes,**style)
    right.plot((-size,+size),(1-size,1+size),transform=right.transAxes,**style)
    left.spines['right'].set_visible(False)
    right.spines['left'].set_visible(False)
    left.tick_params(right=False)
    right.tick_params(left=False,labelleft=False)

deep_rows=[]
for wafer_index,(label,settings) in enumerate(wafer_settings.items()):
    near_col=2*wafer_index
    deep_col=near_col+1
    item=solutions[label]
    bulk,result,profile=item['bulk'],item['result'],item['profile']
    equilibrium=bulk.equilibrium()
    vt=bulk.wafer.temperature_k*8.617333262145e-5
    eq_fermi=equilibrium.eg_effective_ev+vt*equilibrium.eta_n
    x_um=result.depth_cm*1e4
    x_scr_um=profile.depth_cm*1e4
    resolved_delta=item['resolved_delta']
    efn_bulk,efp_bulk=qfl_profiles(bulk,result,resolved_delta)
    n_transport=equilibrium.n0_cm3+resolved_delta
    p_transport=equilibrium.p0_cm3+resolved_delta
    average_state=bulk.injection(max(result.average_delta_n_cm3,1.0))
    diffusion_length_um=average_state.diffusion_length_um

    band_near,band_deep=axes[0,near_col],axes[0,deep_col]
    carrier_near,carrier_deep=axes[1,near_col],axes[1,deep_col]
    # Both sides of a broken axis must use the same ordinate scale; otherwise
    # identical bulk band edges can appear artificially offset.
    band_deep.sharey(band_near)
    carrier_deep.sharey(carrier_near)

    # Near-surface bands and quasi-Fermi levels from the Poisson reconstruction.
    band_near.plot(x_scr_um,profile.ec_ev,color='tab:blue',lw=1.8,label=r'$E_C$')
    band_near.plot(x_scr_um,profile.ev_ev,color='tab:green',lw=1.8,label=r'$E_V$')
    band_near.plot(x_scr_um,profile.efn_ev,color='tab:orange',ls='--',lw=1.6,
                   label=r'$E_{Fn}$')
    band_near.plot(x_scr_um,profile.efp_ev,color='tab:red',ls='--',lw=1.6,
                   label=r'$E_{Fp}$')
    band_near.axhline(eq_fermi,color='.35',ls=':',lw=1.0,label=r'$E_{F,eq}$')

    # Deep-wafer bands are flat; the quasi-Fermi levels follow delta n(z).
    deep_mask=x_um>=SURFACE_BREAK_UM
    band_deep.plot(x_um[deep_mask],np.full(deep_mask.sum(),equilibrium.eg_effective_ev),
                   color='tab:blue',lw=1.5)
    band_deep.plot(x_um[deep_mask],np.zeros(deep_mask.sum()),
                   color='tab:green',lw=1.5)
    band_deep.plot(x_um[deep_mask],efn_bulk[deep_mask],color='tab:orange',
                   ls='--',lw=1.5)
    band_deep.plot(x_um[deep_mask],efp_bulk[deep_mask],color='tab:red',
                   ls='--',lw=1.5)
    band_deep.axhline(eq_fermi,color='.35',ls=':',lw=1.0)

    # Carrier populations: resolved SCR on the left, transport solution on right.
    carrier_near.semilogy(x_scr_um,profile.n_cm3,color='tab:blue',lw=1.8,
                          label=r'$n(z)$')
    carrier_near.semilogy(x_scr_um,profile.p_cm3,color='tab:red',lw=1.8,
                          label=r'$p(z)$')
    carrier_near.axhline(equilibrium.n0_cm3,color='tab:blue',ls=':',lw=1.0,
                         label=r'$n_0$')
    carrier_near.axhline(equilibrium.p0_cm3,color='tab:red',ls=':',lw=1.0,
                         label=r'$p_0$')
    carrier_deep.semilogy(x_um[deep_mask],n_transport[deep_mask],
                          color='tab:blue',lw=1.6)
    carrier_deep.semilogy(x_um[deep_mask],p_transport[deep_mask],
                          color='tab:red',lw=1.6)
    carrier_deep.axhline(equilibrium.n0_cm3,color='tab:blue',ls=':',lw=1.0)
    carrier_deep.axhline(equilibrium.p0_cm3,color='tab:red',ls=':',lw=1.0)

    # Physically useful depth markers, shown only in the segment they occupy.
    absorption_um=optical.absorption_depth_nm/1000
    for depth,color,text in [
        (absorption_um,'tab:green',r'$1/\alpha$'),
        (diffusion_length_um,'tab:purple',r'$L_D$')]:
        target=(carrier_near if depth<SURFACE_BREAK_UM else carrier_deep)
        if depth<=THICKNESS_UM:
            target.axvline(depth,color=color,ls='--',lw=1.0)
            target.text(depth,.97,text,transform=target.get_xaxis_transform(),
                        color=color,ha='right',va='top',fontsize=9.5,
                        bbox=dict(facecolor='white',alpha=.7,edgecolor='none',
                                  boxstyle='round,pad=.15'))

    band_near.set_xlim(0,SURFACE_BREAK_UM)
    carrier_near.set_xlim(0,SURFACE_BREAK_UM)
    band_deep.set_xscale('log'); carrier_deep.set_xscale('log')
    band_deep.set_xlim(SURFACE_BREAK_UM,THICKNESS_UM)
    carrier_deep.set_xlim(SURFACE_BREAK_UM,THICKNESS_UM)
    band_near.set_title(f'{label}: 0--{SURFACE_BREAK_UM*1000:g} nm')
    band_deep.set_title(r'deep wafer to 620 $\mu$m')
    band_near.set_ylabel('Energy relative to bulk $E_V$ (eV)')
    carrier_near.set_ylabel(r'Carrier concentration (cm$^{-3}$)')
    carrier_near.set_xlabel(r'Depth ($\mu$m)')
    carrier_deep.set_xlabel(r'Depth ($\mu$m)')
    band_near.legend(fontsize=9.5,frameon=True,fancybox=True,framealpha=.85,
                     facecolor='white',edgecolor='.65',ncol=2,loc='best')
    carrier_near.legend(fontsize=9.5,frameon=True,fancybox=True,framealpha=.85,
                        facecolor='white',edgecolor='.65',ncol=2,loc='best')
    add_break_marks(band_near,band_deep)
    add_break_marks(carrier_near,carrier_deep)

    rear_delta=float(resolved_delta[-1])
    rear_state=bulk.injection(max(rear_delta,1e-300))
    rear_split=(equilibrium.eg_effective_ev+vt*rear_state.eta_n)-(-vt*rear_state.eta_p)
    deep_rows.append({
        'wafer':label,
        'absorption depth (um)':absorption_um,
        'diffusion length at average injection (um)':diffusion_length_um,
        'rear delta n (cm^-3)':rear_delta,
        'rear EFn-EFp (eV)':rear_split,
        'rear delta / average delta':rear_delta/result.average_delta_n_cm3,
    })

fig.savefig(output_dir/'paper_inp_si_broken_axis_deep_profiles.png',bbox_inches='tight')
display(pd.DataFrame(deep_rows))
""")

md(r"""
### Reading the deep-wafer panels

The dotted carrier lines are the dark equilibrium concentrations, and the
dotted energy line is the single equilibrium Fermi level. Convergence toward
those references quantifies relaxation toward equilibrium. If $E_{Fn}$ and
$E_{Fp}$ remain separated at the rear, the illuminated wafer has not returned
fully to equilibrium within its physical thickness; extending the curve beyond
620 $\mu$m would then describe a hypothetical thicker wafer rather than the
measured sample.
""")

md(r"""
## 6. Interpretation and limitations

For n-InP, positive $Q_f$ bends both band edges downward and produces electron
accumulation. Holes are excluded from the surface, suppressing interface SRH
recombination. For p-InP, the same downward bending depletes holes and moves the
surface toward electron inversion. Depending on the exact charge, injection
and capture asymmetry, this can increase recombination by bringing the surface
populations closer to balance.

The carrier and loss profiles also show why the optical absorption depth alone
does not determine the sampled volume: diffusion redistributes carriers well
beyond the generation region, while the nonlinear surface boundary changes the
near-surface excess density.

The following qualifications should accompany the figure:

- $D_{it}$ and $Q_f$ are illustrative, not extracted;
- PO$_x$/AlO$_x$ optical constants and thicknesses should be replaced by
  sample-specific values;
- the nominal calculation uses $p_\mathrm{rec}=0.93$ and an
  injection-independent 5-ns bulk SRH lifetime; both remain sensitivity
  parameters rather than uniquely extracted quantities;
- the space-charge reconstruction is semi-infinite and locally coupled to the
  converged surface injection;
- a monolithic Poisson--drift--diffusion solver would be required to calculate
  electric fields and quasi-Fermi gradients everywhere without the
  scale-separation approximation.
""")


notebook={
    'cells':cells,
    'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3'},
    },
    'nbformat':4,
    'nbformat_minor':5,
}
destination=ROOT/'notebooks'/'12_inp_passivated_depth_profiles.ipynb'
destination.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
print(destination)
