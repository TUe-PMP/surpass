"""SURPASS: Surface and Passivation Analysis for Semiconductor Systems."""

__version__ = "1.0.0"

from .profiles import (BandProfile, TrapRelaxationSpectrum,
                       reconstruct_band_profile, trap_relaxation_spectrum)
from .transient import TimeDependent1DSolver, TransientResult, GaussianPulse
from .emission import EmissionSpectrum, EmissionBoundary, SlabEmissionModel, PhotonFates
from .experiment import (SampledPulse, PulseTrain, SpectralResponse,
                         InstrumentResponse, DetectionModel, TraceFitResult,
                         fit_detection_nuisance, EmpiricalDecayFit,
                         fit_empirical_exponentials)

from .materials import MATERIALS, Material, get_material
from .wafer import Wafer
from .bulk import BulkModel, BulkState, InjectionState
from .optical_constants import OPTICAL_MATERIALS, OpticalMaterial, get_optical_material
from .optics import Layer, OpticalResult, OpticalStack, single_pass_escape_probability
from .transport import BulkTransportTable, SteadyState1DSolver, TransportResult
from .surface import (
    InterfaceDefectModel, InterfaceChargeModel, FixedChargeBoundary,
    SurfaceSRHModel, SurfaceState, SurfaceBoundaryTable, ADAMOWICZ_INP,
)

__all__ = [
    "__version__",
    "SampledPulse", "PulseTrain", "SpectralResponse", "InstrumentResponse",
    "DetectionModel", "TraceFitResult", "fit_detection_nuisance",
    "EmpiricalDecayFit", "fit_empirical_exponentials",
    "EmissionSpectrum", "EmissionBoundary", "SlabEmissionModel", "PhotonFates",
    "TimeDependent1DSolver",
    "TransientResult",
    "GaussianPulse",
    "BandProfile",
    "TrapRelaxationSpectrum",
    "reconstruct_band_profile",
    "trap_relaxation_spectrum",
    "MATERIALS",
    "Material",
    "get_material",
    "Wafer",
    "BulkModel",
    "BulkState",
    "InjectionState",
    "OPTICAL_MATERIALS",
    "OpticalMaterial",
    "get_optical_material",
    "Layer",
    "OpticalResult",
    "OpticalStack",
    "single_pass_escape_probability",
    "BulkTransportTable",
    "SteadyState1DSolver",
    "TransportResult",
    "InterfaceDefectModel",
    "InterfaceChargeModel",
    "FixedChargeBoundary",
    "SurfaceSRHModel",
    "SurfaceState",
    "SurfaceBoundaryTable",
    "ADAMOWICZ_INP",
]
