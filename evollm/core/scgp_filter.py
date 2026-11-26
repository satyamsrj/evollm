"""
Subspace-Constrained Gradient Projection (SCGP) & Spectral Filter.

Provides ultra-fast O(d·K·r) projection onto the pre-trained invariant subspace,
suppressing interference with foundational capabilities without expensive per-dialogue SVD.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple


class SubspaceFilter:
    def __init__(self, top_k: int = 16, suppression_beta: float = 0.85, frobenius_rho: float = 0.5):
        self.top_k = top_k
        self.suppression_beta = suppression_beta
        self.frobenius_rho = frobenius_rho
        
        # Subspace bases: name -> (U_K, V_K)
        # U_K: (d_out, K), V_K: (d_in, K)
        self.bases: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.reset_step_telemetry()
        self.last_telemetry: Dict[str, float] = {
            "cos_phi": 1.0,
            "core_energy_rejected_pct": 0.0,
        }

    def reset_step_telemetry(self):
        """Resets the layer-aggregated telemetry accumulators for a new dialogue learning step."""
        self._step_raw_sq = 0.0
        self._step_par_sq = 0.0
        self._step_clean_sq = 0.0
        self._step_dot = 0.0

    def extract_and_register_subspace(self, name: str, weight: torch.Tensor):
        """
        Pre-computes top-K left and right singular vectors for a base weight matrix.
        Executed ONCE at system initialization using fast truncated randomized SVD.
        """
        w = weight.detach().float()
        d_out, d_in = w.shape
        max_safe_k = max(1, int(0.25 * min(d_out, d_in)))
        k = min(self.top_k, max_safe_k, d_out, d_in)

        try:
            if min(d_out, d_in) <= 2048:
                U, S, Vh = torch.linalg.svd(w, full_matrices=False)
                U_k = U[:, :k].contiguous()
                V_k = Vh[:k, :].t().contiguous()
                self.bases[name] = (U_k, V_k)
            else:
                # Fast truncated randomized SVD (Halko et al., 2011) for very large matrices
                U, _, V = torch.svd_lowrank(w, q=k, niter=4)
                U_k = U[:, :k].contiguous()
                V_k = V[:, :k].contiguous()
                self.bases[name] = (U_k, V_k)
        except Exception:
            try:
                U, S, Vh = torch.linalg.svd(w, full_matrices=False)
                U_k = U[:, :k].contiguous()
                V_k = Vh[:k, :].t().contiguous()
                self.bases[name] = (U_k, V_k)
            except Exception:
                # Fallback to random orthonormal basis if SVD fails on edge architecture
                U_k, _ = torch.linalg.qr(torch.randn(d_out, k, device=w.device))
                V_k, _ = torch.linalg.qr(torch.randn(d_in, k, device=w.device))
                self.bases[name] = (U_k, V_k)

    def filter_delta(self, name: str, delta_w: torch.Tensor, lora_layer: nn.Module = None) -> torch.Tensor:
        """
        Filters the effective update ΔW by suppressing components aligned with
        either the left (U_K) or right (V_K) dominant singular subspaces:
            P_parallel(ΔW) = U_K U_K^T ΔW + ΔW V_K V_K^T - U_K U_K^T ΔW V_K V_K^T
        so ΔW_orth = (I - U_K U_K^T) ΔW (I - V_K V_K^T).
        If lora_layer is provided (ΔW = scaling * B @ A), computes this associatively in O(d·K·r).
        """
        if name not in self.bases:
            return self._apply_norm_clip(delta_w)

        U_k, V_k = self.bases[name]
        device = delta_w.device
        dtype = delta_w.dtype

        U_k = U_k.to(device=device, dtype=torch.float32)
        V_k = V_k.to(device=device, dtype=torch.float32)

        delta_f32 = delta_w.float()

        if lora_layer is not None and hasattr(lora_layer, "lora_A") and hasattr(lora_layer, "lora_B"):
            # Fast O(d · K · r) associative union projection
            # ΔW = scaling * (B @ A)
            B = lora_layer.lora_B.to(device=device, dtype=torch.float32)
            A = lora_layer.lora_A.to(device=device, dtype=torch.float32)
            scaling = lora_layer.scaling

            # Step 1: (K, d_out) @ (d_out, r) -> (K, r)
            Ut_B = U_k.t() @ B
            # Step 2: (r, d_in) @ (d_in, K) -> (r, K)
            A_V = A @ V_k
            # Step 3: Left, Right, and Bilateral overlap terms
            P_left = ((U_k @ Ut_B) @ A) * scaling
            P_right = (B @ (A_V @ V_k.t())) * scaling
            P_both = ((U_k @ (Ut_B @ A_V)) @ V_k.t()) * scaling
            P_parallel = P_left + P_right - P_both
        else:
            # Direct matrix projection: U_k U_k^T ΔW + ΔW V_k V_k^T - U_k U_k^T ΔW V_k V_k^T
            Ut_d = U_k.t() @ delta_f32
            d_V = delta_f32 @ V_k
            P_parallel = (U_k @ Ut_d) + (d_V @ V_k.t()) - ((U_k @ (Ut_d @ V_k)) @ V_k.t())

        # Suppress interference with invariant base subspace
        filtered_delta = delta_f32 - (self.suppression_beta * P_parallel)

        # Apply Frobenius norm trust-region clip
        clipped_delta = self._apply_norm_clip(filtered_delta)

        # Accumulate model-wide energy & directional alignment across layers
        norm_delta_sq = torch.norm(delta_f32).item() ** 2
        norm_par_sq = torch.norm(P_parallel).item() ** 2
        norm_clean_sq = torch.norm(clipped_delta.float()).item() ** 2
        dot_product = torch.sum(delta_f32 * clipped_delta.float()).item()

        self._step_raw_sq += norm_delta_sq
        self._step_par_sq += norm_par_sq
        self._step_clean_sq += norm_clean_sq
        self._step_dot += dot_product

        core_rejected_pct = ((self._step_par_sq / (self._step_raw_sq + 1e-12)) * 100.0) if self._step_raw_sq > 0 else 0.0
        cos_phi = (self._step_dot / ((self._step_raw_sq ** 0.5) * (self._step_clean_sq ** 0.5) + 1e-12)) if (self._step_raw_sq > 0 and self._step_clean_sq > 0) else 1.0

        self.last_telemetry = {
            "cos_phi": float(max(-1.0, min(1.0, cos_phi))),
            "core_energy_rejected_pct": float(core_rejected_pct),
        }
        
        return clipped_delta.to(dtype=dtype)

    def get_last_telemetry(self) -> Dict[str, float]:
        return dict(self.last_telemetry)

    def _apply_norm_clip(self, delta: torch.Tensor) -> torch.Tensor:
        fro_norm = torch.norm(delta)
        if fro_norm > self.frobenius_rho and self.frobenius_rho > 0:
            return delta * (self.frobenius_rho / (fro_norm + 1e-8))
        return delta

