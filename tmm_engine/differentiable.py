"""Optional PyTorch S-matrix backend for differentiable multilayer optics.

This module is adapted from the differentiable TMM used by the local SpecFormer
project.  The interface is generalized: incident and exit media are supplied in
``nk_tensors`` together with all finite films, while ``thicknesses_um`` contains
only finite-film thicknesses.  No PyTorch import is performed by ``tmm_engine``
unless this module is explicitly used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Tuple, Union

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - exercised by optional runtime
    raise ImportError(
        "Differentiable TMM requires PyTorch. Install the 'tmm-differentiable' "
        "optional dependencies or run it in the configured physics environment."
    ) from exc


@dataclass(frozen=True)
class DifferentiableTMMOutput:
    R: "torch.Tensor"
    T: "torch.Tensor"
    A: "torch.Tensor"
    r: "torch.Tensor"
    t: "torch.Tensor"


class DifferentiableTMM(nn.Module):
    """Batched coherent S-matrix solver with autograd support.

    Shapes:

    - ``thicknesses_um``: ``[B, L]``
    - ``nk_tensors``: ``[B, L + 2, W]`` ordered as incident, films, exit
    - ``wavelengths_um``: ``[W]``
    - output spectra: ``[B, W]``

    The implementation is coherent.  Incoherent layers are handled by the
    non-differentiable validation backend, not silently approximated here.
    """

    def __init__(
        self,
        polarization: Literal["s", "p", "unpolarized"] = "unpolarized",
        dtype_real: Any = None,
        dtype_complex: Any = None,
        eps: float = 1e-30,
    ) -> None:
        super().__init__()
        if polarization not in ("s", "p", "unpolarized"):
            raise ValueError("polarization must be s, p, or unpolarized")
        self.polarization = polarization
        self.dtype_real = dtype_real or torch.float64
        self.dtype_complex = dtype_complex or torch.complex128
        self.eps = float(eps)
        if self.dtype_real not in (torch.float32, torch.float64):
            raise ValueError("dtype_real must be float32 or float64")
        if self.dtype_complex not in (torch.complex64, torch.complex128):
            raise ValueError("dtype_complex must be complex64 or complex128")

    def forward(
        self,
        thicknesses_um: "torch.Tensor",
        nk_tensors: "torch.Tensor",
        wavelengths_um: "torch.Tensor",
        theta_rad: Union[float, "torch.Tensor"] = 0.0,
    ) -> DifferentiableTMMOutput:
        if thicknesses_um.dim() != 2:
            raise ValueError("thicknesses_um must have shape [B,L]")
        if nk_tensors.dim() != 3 or not torch.is_complex(nk_tensors):
            raise ValueError("nk_tensors must be complex with shape [B,L+2,W]")
        if wavelengths_um.dim() != 1:
            raise ValueError("wavelengths_um must have shape [W]")
        batch_size, film_count = thicknesses_um.shape
        b2, media_count, wavelength_count = nk_tensors.shape
        if b2 != batch_size or media_count != film_count + 2:
            raise ValueError(
                "nk_tensors must include incident and exit media: expected [B,%d,W], got %s"
                % (film_count + 2, tuple(nk_tensors.shape))
            )
        if wavelengths_um.shape[0] != wavelength_count:
            raise ValueError("wavelength grid does not match nk_tensors")
        if torch.any(wavelengths_um <= 0):
            raise ValueError("wavelengths must be positive")

        device = nk_tensors.device
        thicknesses_um = thicknesses_um.to(device=device, dtype=self.dtype_real)
        nk_tensors = nk_tensors.to(device=device, dtype=self.dtype_complex)
        wavelengths_um = wavelengths_um.to(device=device, dtype=self.dtype_real)
        theta_bw = self._broadcast_theta(theta_rad, batch_size, wavelength_count, device)

        if self.polarization == "unpolarized":
            rs = self._forward_one_pol(thicknesses_um, nk_tensors, wavelengths_um, theta_bw, "s")
            rp = self._forward_one_pol(thicknesses_um, nk_tensors, wavelengths_um, theta_bw, "p")
            R = 0.5 * (rs[0] + rp[0])
            T = 0.5 * (rs[1] + rp[1])
            r = 0.5 * (rs[2] + rp[2])
            t = 0.5 * (rs[3] + rp[3])
        else:
            R, T, r, t = self._forward_one_pol(
                thicknesses_um, nk_tensors, wavelengths_um, theta_bw, self.polarization
            )
        A = 1.0 - R - T
        return DifferentiableTMMOutput(R=R, T=T, A=A, r=r, t=t)

    def _broadcast_theta(self, theta: Union[float, "torch.Tensor"], B: int, W: int, device: Any) -> "torch.Tensor":
        if isinstance(theta, (float, int)):
            value = torch.tensor(float(theta), dtype=self.dtype_real, device=device)
        else:
            value = theta.to(device=device, dtype=self.dtype_real)
        if value.dim() == 0:
            return value.reshape(1, 1).expand(B, W)
        if value.dim() == 1:
            if value.shape[0] == B:
                return value.reshape(B, 1).expand(B, W)
            if value.shape[0] == W:
                return value.reshape(1, W).expand(B, W)
        if value.dim() == 2 and value.shape[0] in (1, B) and value.shape[1] in (1, W):
            return value.expand(B, W)
        raise ValueError("theta_rad is not broadcastable to [B,W]")

    def _cos_theta(self, n_i: "torch.Tensor", conserved_kx: "torch.Tensor") -> "torch.Tensor":
        eps = torch.tensor(self.eps + 0j, dtype=self.dtype_complex, device=n_i.device)
        cos_t = torch.sqrt(1.0 - (conserved_kx.unsqueeze(1) / (n_i + eps)) ** 2)
        kz = n_i * cos_t
        return torch.where(torch.imag(kz) < 0.0, -cos_t, cos_t)

    def _eta(self, n_i: "torch.Tensor", cos_i: "torch.Tensor", pol: str) -> "torch.Tensor":
        eps = torch.tensor(self.eps + 0j, dtype=self.dtype_complex, device=n_i.device)
        if pol == "s":
            return n_i * cos_i
        return n_i / (cos_i + eps)

    def _star(
        self,
        A11: "torch.Tensor", A12: "torch.Tensor", A21: "torch.Tensor", A22: "torch.Tensor",
        B11: "torch.Tensor", B12: "torch.Tensor", B21: "torch.Tensor", B22: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        eps = torch.tensor(self.eps + 0j, dtype=self.dtype_complex, device=A11.device)
        denom = 1.0 - A22 * B11 + eps
        return (
            A11 + A12 * B11 * A21 / denom,
            A12 * B12 / denom,
            B21 * A21 / denom,
            B22 + B21 * A22 * B12 / denom,
        )

    def _forward_one_pol(
        self,
        thicknesses_um: "torch.Tensor",
        nk: "torch.Tensor",
        wavelengths_um: "torch.Tensor",
        theta_bw: "torch.Tensor",
        pol: str,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        B, L = thicknesses_um.shape
        W = wavelengths_um.shape[0]
        device = nk.device
        n_incident = nk[:, 0, :]
        n_films = nk[:, 1:-1, :]
        conserved_kx = n_incident * torch.sin(theta_bw).to(self.dtype_complex)

        all_n = nk
        all_cos = self._cos_theta(all_n, conserved_kx)
        all_eta = self._eta(all_n, all_cos, pol)

        zero = torch.zeros((B, W), dtype=self.dtype_complex, device=device)
        one = torch.ones((B, W), dtype=self.dtype_complex, device=device)
        S11, S12, S21, S22 = zero, one, one, zero
        eps = torch.tensor(self.eps + 0j, dtype=self.dtype_complex, device=device)
        lam = wavelengths_um.reshape(1, W).to(self.dtype_complex)
        two_pi = torch.tensor(2.0 * 3.141592653589793, dtype=self.dtype_complex, device=device)

        for interface_index in range(L + 1):
            eta_i = all_eta[:, interface_index, :]
            eta_j = all_eta[:, interface_index + 1, :]
            denom = eta_i + eta_j + eps
            r_ij = (eta_i - eta_j) / denom
            t_ij = 2.0 * eta_i / denom
            r_ji = -r_ij
            t_ji = 2.0 * eta_j / denom
            S11, S12, S21, S22 = self._star(
                S11, S12, S21, S22, r_ij, t_ji, t_ij, r_ji
            )
            if interface_index < L:
                delta = (
                    two_pi
                    * n_films[:, interface_index, :]
                    * all_cos[:, interface_index + 1, :]
                    * thicknesses_um[:, interface_index].reshape(B, 1).to(self.dtype_complex)
                    / (lam + eps)
                )
                phase = torch.exp(1j * delta)
                S11, S12, S21, S22 = self._star(
                    S11, S12, S21, S22, zero, phase, phase, zero
                )

        r = S11
        t = S21
        R = torch.real(torch.abs(r) ** 2)
        eta0 = all_eta[:, 0, :]
        etaS = all_eta[:, -1, :]
        T = torch.real(etaS) / (torch.real(eta0) + self.eps) * torch.abs(t) ** 2
        return R, torch.real(T), r, t


__all__ = ["DifferentiableTMM", "DifferentiableTMMOutput"]
