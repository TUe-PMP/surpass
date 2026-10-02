"""Traceable material presets.

Values are deliberately stored with citations and notes rather than hidden in
solver functions. Parameters are 300 K reference values unless stated
otherwise. Users should clone a dataclass with ``dataclasses.replace`` when a
different literature dataset is preferred.

Band parameters for III-Vs are based primarily on Vurgaftman et al. (2001),
DOI 10.1063/1.1368156, supplemented by the Ioffe NSM compilation. Mobility is
represented by a compact Caughey–Thomas estimate motivated by Sotoodeh et al.
(2000), DOI 10.1063/1.372274; the present coefficients are explicitly marked as
estimates and should not be used as metrology-grade fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MobilityParameters:
    """Caughey–Thomas-like mobility parameters at 300 K.

    mu(N,T) = mu_min + (mu_max(T)-mu_min) / [1+(N/Nref)^alpha]
    mu_max(T) = mu_max_300*(T/300)^temperature_exponent
    """

    mu_max_300: float
    mu_min: float
    n_ref_cm3: float
    alpha: float
    temperature_exponent: float
    source: str
    status: str = "estimate"


@dataclass(frozen=True)
class Material:
    name: str
    symbol: str
    eg_300_ev: float
    varshni_alpha_ev_k: float
    varshni_beta_k: float
    nc_300_cm3: float
    nv_300_cm3: float
    eps_r: float
    electron_mobility: MobilityParameters
    hole_mobility: MobilityParameters
    b_rad_cm3_s: float
    c_n_cm6_s: float
    c_p_cm6_s: float
    references: tuple[str, ...] = field(default_factory=tuple)
    cautions: tuple[str, ...] = field(default_factory=tuple)

    def bandgap_ev(self, temperature_k: float) -> float:
        """Varshni band gap, anchored so that the preset is exact at 300 K."""
        t = float(temperature_k)
        a, b = self.varshni_alpha_ev_k, self.varshni_beta_k
        correction = a * (t * t / (t + b) - 300.0**2 / (300.0 + b))
        return self.eg_300_ev - correction

    def nc_cm3(self, temperature_k: float) -> float:
        return self.nc_300_cm3 * (temperature_k / 300.0) ** 1.5

    def nv_cm3(self, temperature_k: float) -> float:
        return self.nv_300_cm3 * (temperature_k / 300.0) ** 1.5


VURGAFTMAN = (
    "I. Vurgaftman, J. R. Meyer, and L. R. Ram-Mohan, J. Appl. Phys. 89, "
    "5815 (2001), DOI 10.1063/1.1368156."
)
SOTOODEH = (
    "M. Sotoodeh, A. H. Khalid, and A. A. Rezazadeh, J. Appl. Phys. 87, "
    "2890 (2000), DOI 10.1063/1.372274."
)
IOFFE = "Ioffe Institute NSM semiconductor parameter archive, 300 K compilation."


MATERIALS: dict[str, Material] = {
    "InP": Material(
        name="indium phosphide",
        symbol="InP",
        eg_300_ev=1.344,
        varshni_alpha_ev_k=4.90e-4,
        varshni_beta_k=327.0,
        nc_300_cm3=5.7e17,
        nv_300_cm3=1.1e19,
        eps_r=12.5,
        electron_mobility=MobilityParameters(
            5400.0, 300.0, 1.0e18, 0.70, -2.0, SOTOODEH
        ),
        hole_mobility=MobilityParameters(
            200.0, 35.0, 1.0e18, 0.70, -2.0, SOTOODEH
        ),
        b_rad_cm3_s=1.2e-10,
        c_n_cm6_s=9.0e-31,
        c_p_cm6_s=1.0e-29,
        references=(VURGAFTMAN, IOFFE, SOTOODEH),
        cautions=(
            "Mobility coefficients are first-release estimates, not a full transcription of Sotoodeh.",
            "Published InP B and Auger coefficients vary with doping and carrier temperature.",
            "B/C are representative development constants, not uniquely source-validated fits; see BULK_RECOMBINATION_AUDIT.md.",
        ),
    ),
    "GaAs": Material(
        name="gallium arsenide",
        symbol="GaAs",
        eg_300_ev=1.424,
        varshni_alpha_ev_k=5.405e-4,
        varshni_beta_k=204.0,
        nc_300_cm3=4.7e17,
        nv_300_cm3=7.0e18,
        eps_r=12.9,
        electron_mobility=MobilityParameters(
            8500.0, 300.0, 1.0e17, 0.60, -2.2, SOTOODEH
        ),
        hole_mobility=MobilityParameters(
            400.0, 50.0, 2.0e17, 0.70, -2.2, SOTOODEH
        ),
        b_rad_cm3_s=1.0e-10,
        c_n_cm6_s=7.0e-30,
        c_p_cm6_s=1.0e-29,
        references=(VURGAFTMAN, IOFFE, SOTOODEH),
        cautions=("Recombination coefficients are representative 300 K estimates.",),
    ),
    "Si": Material(
        name="silicon",
        symbol="Si",
        eg_300_ev=1.124,
        varshni_alpha_ev_k=4.73e-4,
        varshni_beta_k=636.0,
        nc_300_cm3=2.80e19,
        nv_300_cm3=1.04e19,
        eps_r=11.7,
        electron_mobility=MobilityParameters(
            1417.0, 52.2, 9.68e16, 0.68, -2.3,
            "D. B. M. Klaassen, Solid-State Electron. 35, 953 (1992).",
            status="compact approximation",
        ),
        hole_mobility=MobilityParameters(
            470.5, 44.9, 2.23e17, 0.719, -2.2,
            "D. B. M. Klaassen, Solid-State Electron. 35, 953 (1992).",
            status="compact approximation",
        ),
        b_rad_cm3_s=4.73e-15,
        c_n_cm6_s=2.8e-31,
        c_p_cm6_s=9.9e-32,
        references=(
            "A. Richter et al., Phys. Rev. B 86, 165202 (2012), DOI 10.1103/PhysRevB.86.165202.",
            IOFFE,
        ),
        cautions=(
            "The constant B and Auger coefficients are only a benchmark; use the full Richter model for precision Si work.",
        ),
    ),
    "Ge": Material(
        name="germanium",
        symbol="Ge",
        eg_300_ev=0.661,
        varshni_alpha_ev_k=4.77e-4,
        varshni_beta_k=235.0,
        nc_300_cm3=1.04e19,
        nv_300_cm3=6.0e18,
        eps_r=16.0,
        electron_mobility=MobilityParameters(
            3900.0, 100.0, 1.0e17, 0.70, -2.0, IOFFE
        ),
        hole_mobility=MobilityParameters(
            1900.0, 50.0, 1.0e17, 0.70, -2.0, IOFFE
        ),
        b_rad_cm3_s=5.0e-10,
        c_n_cm6_s=3.0e-32,
        c_p_cm6_s=7.0e-32,
        references=(IOFFE,),
        cautions=(
            "Ge radiative recombination is indirect and strongly model-dependent; B is an effective estimate.",
            "Mobility and Auger coefficients are provisional pending a dedicated Ge validation dataset.",
        ),
    ),
}


def get_material(name: str) -> Material:
    """Return a material preset by symbol, case-insensitively."""
    matches = [m for key, m in MATERIALS.items() if key.lower() == name.lower()]
    if not matches:
        raise KeyError(f"Unknown material {name!r}; choose {tuple(MATERIALS)}")
    return matches[0]
