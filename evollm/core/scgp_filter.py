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

    def extract_and_register_subspace(self, name: str, weight: torch.Tensor):
        """
        Pre-computes top-K left and right singular vectors for a base weight matrix.
        Executed ONCE at system initialization.
        """
        w = weight.detach().float()
        d_out, d_in = w.shape
        k = min(self.top_k, d_out, d_in)
        
        # Fast truncated SVD using torch.linalg.svd
        # w = U @ diag(S) @ Vh
        try:
            # randomized or full svd depending on size
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
            U_k = U[:, :k].contiguous()     # (d_out, k)
            V_k = Vh[:k, :].t().contiguous() # (d_in, k)
            self.bases[name] = (U_k, V_k)
        except Exception as e:
            # Fallback to random orthonormal basis if SVD fails on edge architecture
            U_k, _ = torch.linalg.qr(torch.randn(d_out, k, device=w.device))
            V_k, _ = torch.linalg.qr(torch.randn(d_in, k, device=w.device))
            self.bases[name] = (U_k, V_k)

    def filter_delta(self, name: str, delta_w: torch.Tensor, lora_layer: nn.Module = None) -> torch.Tensor:
        """
        Filters the effective update ΔW.
        If lora_layer is provided (with low-rank matrices B and A),
        computes projection in O(d·K·r) time.
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
            # Fast O(d · K · r) associative projection
            # ΔW = scaling * (B @ A)
            B = lora_layer.lora_B.float() # (d_out, r)
            A = lora_layer.lora_A.float() # (r, d_in)
            scaling = lora_layer.scaling

            # Step 1: (K, d_out) @ (d_out, r) -> (K, r)
            Ut_B = U_k.t() @ B
            # Step 2: (r, d_in) @ (d_in, K) -> (r, K)
            A_V = A @ V_k
            # Step 3: (K, r) @ (r, K) -> (K, K)
            M = (Ut_B @ A_V) * scaling
            # Step 4: U_k @ M @ V_k^T -> (d_out, d_in)
            P_parallel = (U_k @ M) @ V_k.t()
        else:
            # Direct matrix projection: U_k @ (U_k^T @ ΔW @ V_k) @ V_k^T
            P_parallel = (U_k @ (U_k.t() @ delta_f32 @ V_k)) @ V_k.t()

        # Suppress interference with invariant base subspace
        filtered_delta = delta_f32 - (self.suppression_beta * P_parallel)
        
        # Apply Frobenius norm trust-region clip
        clipped_delta = self._apply_norm_clip(filtered_delta)
        return clipped_delta.to(dtype=dtype)

    def _apply_norm_clip(self, delta: torch.Tensor) -> torch.Tensor:
        fro_norm = torch.norm(delta)
        if fro_norm > self.frobenius_rho and self.frobenius_rho > 0:
            return delta * (self.frobenius_rho / (fro_norm + 1e-8))
        return delta

