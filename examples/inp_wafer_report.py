"""Generate a report and injection plots for the experimental InP wafers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from surpass import BulkModel, Wafer


OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


def print_report(model: BulkModel):
    w, s = model.wafer, model.equilibrium()
    print(f"\n{w.label}: {w.material.symbol}, {w.polarity}-type")
    print(f"  net doping              = {w.net_doping_cm3:+.3e} cm^-3")
    print(f"  nominal/effective Eg    = {s.eg_nominal_ev:.4f}/{s.eg_effective_ev:.4f} eV")
    print(f"  eta_n, eta_p            = {s.eta_n:+.3f}, {s.eta_p:+.3f}")
    print(f"  n0, p0                  = {s.n0_cm3:.3e}, {s.p0_cm3:.3e} cm^-3")
    print(f"  mu_n, mu_p              = {s.mu_n_cm2_vs:.0f}, {s.mu_p_cm2_vs:.0f} cm^2/Vs")
    print(f"  resistivity             = {s.resistivity_ohm_cm:.3e} ohm cm")
    print(f"  Debye/Fermi screening   = {s.debye_length_nm:.3f}/{s.fermi_screening_length_nm:.3f} nm")


def main():
    wafers = [
        Wafer("InP", donor_cm3=8.5e18, thickness_um=620, label="experimental n-InP"),
        Wafer("InP", acceptor_cm3=5.1e18, thickness_um=620, label="experimental p-InP"),
    ]
    # 10 ns is an illustrative bulk-SRH lifetime, deliberately exposed as an
    # assumption. Change or disable it rather than treating it as a constant.
    models = [BulkModel(w, enable_bgn=False, tau_srh_s=10e-9) for w in wafers]
    for model in models:
        print_report(model)

    injections = np.logspace(10, 19, 121)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    for model in models:
        states = [model.injection(x) for x in injections]
        label = model.wafer.label
        axes[0].loglog(injections, [x.tau_rad_s for x in states], label=f"radiative, {label}")
        axes[0].loglog(injections, [x.tau_auger_s for x in states], "--", label=f"Auger, {label}")
        axes[0].loglog(injections, [x.tau_total_s for x in states], ":", lw=2, label=f"total, {label}")
        axes[1].loglog(injections, [x.diffusion_length_um for x in states], label=label)

    axes[0].set(xlabel=r"Excess carrier density $\Delta n$ (cm$^{-3}$)", ylabel="Effective lifetime (s)")
    axes[1].set(xlabel=r"Excess carrier density $\Delta n$ (cm$^{-3}$)", ylabel="Ambipolar diffusion length (µm)")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.savefig(OUTPUT / "inp_bulk_lifetimes_and_diffusion.png", dpi=220)
    print(f"\nWrote {OUTPUT / 'inp_bulk_lifetimes_and_diffusion.png'}")


if __name__ == "__main__":
    main()

