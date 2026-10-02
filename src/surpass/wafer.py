"""Wafer definition with explicit donor/acceptor densities."""

from __future__ import annotations

from dataclasses import dataclass

from .materials import Material, get_material


@dataclass(frozen=True)
class Wafer:
    material: Material | str
    donor_cm3: float = 0.0
    acceptor_cm3: float = 0.0
    thickness_um: float = 500.0
    temperature_k: float = 300.0
    label: str = ""

    def __post_init__(self):
        if isinstance(self.material, str):
            object.__setattr__(self, "material", get_material(self.material))
        if self.donor_cm3 < 0 or self.acceptor_cm3 < 0:
            raise ValueError("Donor and acceptor densities must be non-negative")
        if self.thickness_um <= 0 or self.temperature_k <= 0:
            raise ValueError("Thickness and temperature must be positive")

    @property
    def net_doping_cm3(self) -> float:
        return self.donor_cm3 - self.acceptor_cm3

    @property
    def total_ionized_impurity_cm3(self) -> float:
        # Complete ionization is the release-0.1 assumption.
        return self.donor_cm3 + self.acceptor_cm3

    @property
    def polarity(self) -> str:
        if self.net_doping_cm3 > 0:
            return "n"
        if self.net_doping_cm3 < 0:
            return "p"
        return "intrinsic/compensated"

