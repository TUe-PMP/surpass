"""Spectral photon fates in a planar, incoherent semiconductor slab.

Emission is isotropic and unpolarized in a homogeneous bulk medium. Films at
each surface are coherent, but repeated wafer traversals are incoherent.
Interband absorption (one regenerated pair per absorbed photon) is separated
from parasitic bulk absorption and absorption in surface films. Flight time,
scattering, lateral escape, dipole near-field coupling, interference across the
wafer, injection-dependent spectra and stimulated emission are omitted.

Angular thin-film formalism: S. J. Byrnes, Multilayer optical calculations,
https://arxiv.org/abs/1603.02720. Probability bookkeeping and distinction of
interband/free-carrier absorption: O. Semyonov et al., Radiation efficiency of
heavily doped bulk n-InP semiconductor, https://arxiv.org/pdf/1003.6095.
This module implements the stated ray approximation, not either paper's fit.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from .optics import Layer
from .optical_constants import OpticalMaterial


@dataclass(frozen=True)
class EmissionSpectrum:
    """Photon-number density per nm; normalized over the provided support.

Supply the *internal* emission spectrum, not an already filtered external PL
spectrum. A single wavelength is permitted as a monochromatic limit. There
is no automatic conversion from an energy-density spectrum: include |dE/dλ|
and divide power spectra by photon energy before constructing this object.
"""
    wavelength_nm: np.ndarray
    photon_density_per_nm: np.ndarray
    source: str = 'user supplied'

    def __post_init__(self):
        wl, density = [np.asarray(v, dtype=float) for v in
                       (self.wavelength_nm, self.photon_density_per_nm)]
        if (wl.ndim != 1 or wl.size < 1 or density.shape != wl.shape
                or np.any(~np.isfinite(wl)) or np.any(wl <= 0)
                or np.any(np.diff(wl) <= 0) or np.any(~np.isfinite(density))
                or np.any(density < 0)):
            raise ValueError('Use increasing positive wavelengths and non-negative finite photon density')
        area = float(density[0]) if wl.size == 1 else float(np.trapezoid(density, wl))
        if area <= 0:
            raise ValueError('Spectrum must have positive integral')
        object.__setattr__(self, 'wavelength_nm', wl.copy())
        object.__setattr__(self, 'photon_density_per_nm', density/area)

    def average(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape[0] != self.wavelength_nm.size:
            raise ValueError('First axis must match wavelengths')
        if self.wavelength_nm.size == 1:
            return values[0]
        weights = self.photon_density_per_nm.reshape((-1,)+(1,)*(values.ndim-1))
        return np.trapezoid(values*weights, self.wavelength_nm, axis=0)


class EmissionBoundary:
    """Films ordered FROM WAFER TO OUTSIDE (reverse a pump stack's layers).

    Fresnel incidence uses the real bulk refractive index; attenuation is
    specified separately in SlabEmissionModel. This weak-absorption ray
    approximation is not exact for a strongly absorbing incident medium.
    Outside medium must be lossless. NA selects a cone about the surface
    normal; it does not change total escape or carrier loss.
    """
    def __init__(self, outside: OpticalMaterial, layers=()):
        self.outside = outside
        self.layers = tuple(layers)
        if any(not isinstance(layer, Layer) for layer in self.layers):
            raise TypeError('layers must contain Layer objects')

    def probabilities(self, wavelength_nm, n_inside, mu, polarization):
        if polarization not in ('s', 'p'):
            raise ValueError('polarization must be s or p')
        if not np.isfinite(n_inside) or n_inside <= 0 or not 0 < mu <= 1:
            raise ValueError('Require positive finite index and 0<mu<=1')
        no = self.outside.nk(wavelength_nm)
        if no.real <= 0 or no.imag != 0:
            raise ValueError('Outside medium must have positive real index and zero extinction')
        transverse = n_inside*np.sqrt(max(1-mu*mu, 0))

        def admittance(index):
            kz = np.sqrt(complex(index*index-transverse*transverse))
            # Passive forward wave: positive imaginary kz (and positive real
            # kz in the lossless limit). Characteristic p admittance is n/cos.
            if kz.imag < 0 or (kz.imag == 0 and kz.real < 0):
                kz = -kz
            return kz, kz if polarization == 's' else index*index/kz

        y0 = n_inside*mu if polarization == 's' else n_inside/mu
        kz_out, ys = admittance(no)
        matrix = np.eye(2, dtype=complex)
        for layer in self.layers:
            kz, y = admittance(layer.material.nk(wavelength_nm))
            phase = 2*np.pi*kz*layer.thickness_nm/wavelength_nm
            co, si = np.cos(phase), np.sin(phase)
            # For n+i*k and exp(-i*omega*t), the backward transfer uses -i.
            matrix = matrix @ np.array([[co, -1j*si/y], [-1j*y*si, co]])
        b = matrix[0, 0]+matrix[0, 1]*ys
        c = matrix[1, 0]+matrix[1, 1]*ys
        r = (y0*b-c)/(y0*b+c)
        t = 2*y0/(y0*b+c)
        reflect = float(abs(r)**2)
        transmit = 0.0 if kz_out.real == 0 else float(ys.real/y0*abs(t)**2)
        absorb = 1-reflect-transmit
        if not np.all(np.isfinite([reflect, transmit, absorb])) or absorb < -1e-8:
            raise ValueError('Non-passive or numerically unstable emission boundary')
        if all(layer.material.nk(wavelength_nm).imag == 0 for layer in self.layers):
            absorb, reflect = 0., 1-transmit
        return reflect, transmit, max(absorb, 0.)


@dataclass(frozen=True)
class PhotonFates:
    wavelength_nm: np.ndarray
    depth_cm: np.ndarray
    front_escape: np.ndarray  # shape (wavelength, depth)
    rear_escape: np.ndarray
    active_reabsorption: np.ndarray
    parasitic_absorption: np.ndarray  # wafer plus films
    trapped: np.ndarray  # lossless total-internal-reflection rays only
    front_collected: np.ndarray
    rear_collected: np.ndarray
    balance_error: np.ndarray
    spectrum: EmissionSpectrum

    def averaged(self):
        return {name: self.spectrum.average(getattr(self, name)) for name in
                ('front_escape', 'rear_escape', 'active_reabsorption',
                 'parasitic_absorption', 'trapped', 'front_collected', 'rear_collected')}

    def emission_fluxes(self, radiative_cm3_s, cell_width_cm):
        """Integrate intrinsic (NOT recycling-reduced) emission over depth.

        Arrays may be (depth,) or (time, depth); outputs are photons cm^-2 s^-1.
        Constant spectrum is used at every depth, time and injection level.
        """
        rates, widths = np.asarray(radiative_cm3_s), np.asarray(cell_width_cm)
        if (widths.shape != self.depth_cm.shape or rates.shape[-1] != widths.size
                or np.any(~np.isfinite(rates)) or np.any(rates < 0)
                or np.any(~np.isfinite(widths)) or np.any(widths <= 0)):
            raise ValueError('Supply finite non-negative intrinsic rates and positive matching widths')
        result = {name: np.sum(rates*widths*prob, axis=-1)
                  for name, prob in self.averaged().items()}
        result['internal'] = np.sum(rates*widths, axis=-1)
        result['net_radiative_loss'] = result['internal']-result['active_reabsorption']
        return result

    def spectral_flux(self, radiative_cm3_s, cell_width_cm, fate='front_collected'):
        """Return photon flux density per nm with wavelength on the last axis.

        For a monochromatic ``EmissionSpectrum`` the last axis has length one
        and contains total line flux rather than a finite-bandwidth density.
        """
        rates, widths = np.asarray(radiative_cm3_s), np.asarray(cell_width_cm)
        allowed = ('front_escape', 'rear_escape', 'active_reabsorption',
                   'parasitic_absorption', 'trapped', 'front_collected', 'rear_collected')
        if fate not in allowed:
            raise ValueError(f'fate must be one of {allowed}')
        if (widths.shape != self.depth_cm.shape or rates.shape[-1] != widths.size
                or np.any(~np.isfinite(rates)) or np.any(rates < 0)
                or np.any(~np.isfinite(widths)) or np.any(widths <= 0)):
            raise ValueError('Supply finite non-negative intrinsic rates and positive matching widths')
        optical = getattr(self, fate)*self.spectrum.photon_density_per_nm[:, None]
        return np.einsum('...d,wd,d->...w', rates, optical, widths)

    def response_weighted_flux(self, radiative_cm3_s, cell_width_cm, response,
                               fate='front_collected'):
        """Detected-event flux per area after a dimensionless spectral response."""
        spectral = self.spectral_flux(radiative_cm3_s, cell_width_cm, fate)
        efficiency = response.evaluate(self.wavelength_nm)
        if self.wavelength_nm.size == 1:
            return spectral[..., 0]*efficiency[0]
        return np.trapezoid(spectral*efficiency, self.wavelength_nm, axis=-1)


class SlabEmissionModel:
    """Incoherent multiple-reflection ray model with independent absorption.

    alpha_active_cm1 and alpha_parasitic_cm1 can be scalars, spectral vectors,
    or wavelength callables. Only active absorption regenerates carriers.
    Defaults are deliberately not inferred from the demo n/k tables: their
    near-edge k cannot distinguish interband absorption from free-carrier loss.
    """
    def __init__(self, wafer_optics, front, rear, alpha_active_cm1,
                 alpha_parasitic_cm1=0.0):
        self.wafer_optics, self.front, self.rear = wafer_optics, front, rear
        self.alpha_active_cm1 = alpha_active_cm1
        self.alpha_parasitic_cm1 = alpha_parasitic_cm1

    def solve(self, thickness_cm, depth_cm, spectrum, angular_points=160,
              front_na=None, rear_na=None):
        depth = np.asarray(depth_cm, dtype=float)
        if (not np.isfinite(thickness_cm) or thickness_cm <= 0 or depth.ndim != 1
                or depth.size == 0 or np.any(~np.isfinite(depth))
                or np.any(depth < 0) or np.any(depth > thickness_cm)
                or angular_points < 16):
            raise ValueError('Require positive thickness, depths inside wafer, and >=16 angular points')
        wavelengths = spectrum.wavelength_nm

        def alpha(value):
            a = np.asarray([value(wl) for wl in wavelengths] if callable(value) else value, dtype=float)
            if a.shape == (): a = np.full(wavelengths.size, float(a))
            if a.shape != wavelengths.shape or np.any(~np.isfinite(a)) or np.any(a < 0):
                raise ValueError('Absorption must be finite, non-negative and match spectrum')
            return a

        active, passive = alpha(self.alpha_active_cm1), alpha(self.alpha_parasitic_cm1)
        arrays = [np.zeros((wavelengths.size, depth.size)) for _ in range(7)]
        fe, re, ar, pa, trapped, fc, rc = arrays
        nodes, weights = leggauss(angular_points)
        for iw, wl in enumerate(wavelengths):
            n = self.wafer_optics.nk(wl).real
            if not np.isfinite(n) or n <= 0: raise ValueError('Wafer index must be positive')
            nf, nr = self.front.outside.nk(wl).real, self.rear.outside.nk(wl).real
            for na, outside_n in ((front_na, nf), (rear_na, nr)):
                if na is not None and (not np.isfinite(na) or not 0 < na <= outside_n):
                    raise ValueError('NA must lie in (0, outside refractive index]')
            # Split the mu integration at both critical angles and collection
            # cones. This avoids smearing their sharp transitions across nodes.
            cuts = [0., 1.]
            for value in (nf, nr, front_na, rear_na):
                if value is not None and value < n:
                    cuts.append(np.sqrt(1-(value/n)**2))
            cuts = np.unique(cuts)
            a = active[iw]+passive[iw]
            for lower, upper in zip(cuts[:-1], cuts[1:]):
                mus = (lower+upper)/2+(upper-lower)/2*nodes
                ws = weights*(upper-lower)/2
                for mu, weight in zip(mus, ws):
                    for pol in ('s', 'p'):
                        rf, tf, af = self.front.probabilities(wl, n, mu, pol)
                        rr, tr, ap = self.rear.probabilities(wl, n, mu, pol)
                        rf, rr = min(rf, 1.), min(rr, 1.)
                        flight = np.exp(-a*thickness_cm/mu)
                        roundtrip_r = rf*rr
                        # Stable 1-Rf*Rr*exp(-2*a*L/mu), even at tiny a.
                        denom = (1-roundtrip_r)+roundtrip_r*(-np.expm1(-2*a*thickness_cm/mu))
                        w = weight/4  # half directions, half polarizations
                        if denom <= 0:
                            trapped[iw] += 2*w
                            continue
                        sf = np.exp(-a*depth/mu)
                        sr = np.exp(-a*(thickness_cm-depth)/mu)
                        reach_f = (sf+sr*rr*flight)/denom
                        reach_r = (sr+sf*rf*flight)/denom
                        pfe, pre = tf*reach_f, tr*reach_r
                        films = af*reach_f+ap*reach_r
                        # Absorption on initial paths plus all reflected paths.
                        # More direct conditional absorption after reaching a
                        # boundary: Hf=Rf*(1-b)*(1+b*Rr)/denom, similarly Hr.
                        bulkabs = (-np.expm1(-a*depth/mu)-np.expm1(-a*(thickness_cm-depth)/mu)
                                   +sf*rf*(-np.expm1(-a*thickness_cm/mu))*(1+flight*rr)/denom
                                   +sr*rr*(-np.expm1(-a*thickness_cm/mu))*(1+flight*rf)/denom)
                        fe[iw] += w*pfe
                        re[iw] += w*pre
                        ar[iw] += w*bulkabs*(active[iw]/a if a > 0 else 0.)
                        pa[iw] += w*(films+bulkabs*(passive[iw]/a if a > 0 else 0.))
                        lateral = n*np.sqrt(max(1-mu*mu, 0))
                        if front_na is None or lateral <= front_na: fc[iw] += w*pfe
                        if rear_na is None or lateral <= rear_na: rc[iw] += w*pre
        balance = fe+re+ar+pa+trapped-1
        if np.max(np.abs(balance)) > 1e-7:
            raise RuntimeError('Photon-fate probabilities do not conserve photons')
        return PhotonFates(wavelengths.copy(), depth.copy(), *arrays, balance, spectrum)
