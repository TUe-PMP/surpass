"""Small, traceable optical-constant library for model development.

The dielectric films below are deliberately labelled ``illustrative``.  They
make the transfer-matrix machinery executable, but should be replaced with
ellipsometry from the actual POx/AlOx process before quantitative comparison
with experiment.  Complex index uses N = n + i*k with exp[i(kz-omega*t)].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OpticalMaterial:
    name: str
    wavelength_nm: tuple[float, ...]
    n: tuple[float, ...]
    k: tuple[float, ...]
    source: str
    status: str = "literature"
    note: str = ""

    def nk(self, wavelength_nm: float) -> complex:
        """Linearly interpolate n and k; refuse silent extrapolation."""
        wl = float(wavelength_nm)
        grid = np.asarray(self.wavelength_nm, dtype=float)
        if not grid[0] <= wl <= grid[-1]:
            raise ValueError(
                f"{self.name}: {wl:g} nm is outside {grid[0]:g}-{grid[-1]:g} nm"
            )
        return complex(np.interp(wl, grid, self.n), np.interp(wl, grid, self.k))

    def absorption_coefficient_cm1(self, wavelength_nm: float) -> float:
        """Return alpha=4*pi*k/lambda in cm^-1."""
        return 4.0 * np.pi * self.nk(wavelength_nm).imag / (wavelength_nm * 1e-7)


ASPNES_INP = (
    "D. E. Aspnes and A. A. Studna, Phys. Rev. B 27, 985 (1983), "
    "DOI 10.1103/PhysRevB.27.985."
)

OPTICAL_MATERIALS: dict[str, OpticalMaterial] = {
    "air": OpticalMaterial(
        "air", (200.0, 2000.0), (1.0, 1.0), (0.0, 0.0),
        "Standard approximation at ambient pressure.",
    ),
    "InP_demo": OpticalMaterial(
        "InP (demonstration interpolation)",
        (450.0, 514.0, 600.0, 800.0, 920.0, 1000.0),
        (4.20, 3.90, 3.58, 3.35, 3.25, 3.20),
        (0.95, 0.45, 0.29, 0.06, 0.0, 0.0),
        ASPNES_INP,
        status="illustrative interpolation",
        note=(
            "Rounded development values guided by the cited room-temperature "
            "dataset. Use the tabulated dataset (and doping/temperature-specific "
            "near-edge absorption) for quantitative work."
        ),
    ),
    "AlOx_demo": OpticalMaterial(
        "amorphous AlOx (demonstration)",
        (400.0, 1200.0), (1.68, 1.62), (0.0, 0.0),
        "Placeholder Cauchy-like dispersion; replace with process ellipsometry.",
        status="illustrative",
    ),
    "POx_demo": OpticalMaterial(
        "phosphorus oxide (demonstration)",
        (400.0, 1200.0), (1.58, 1.52), (0.0, 0.0),
        "Placeholder Cauchy-like dispersion; replace with process ellipsometry.",
        status="illustrative",
    ),
}


def get_optical_material(name: str) -> OpticalMaterial:
    for key, value in OPTICAL_MATERIALS.items():
        if key.lower() == name.lower():
            return value
    raise KeyError(f"Unknown optical material {name!r}; choose {tuple(OPTICAL_MATERIALS)}")
