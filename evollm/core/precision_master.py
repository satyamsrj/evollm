"""
Dual-Precision Parameter Management for EvoLLM.

Prevents catastrophic mantissa underflow and rounding drift by maintaining
an FP32 master weight copy while serving and computing forward passes in
efficient execution dtypes (BF16/FP16/FP32).
"""
import torch
import torch.nn as nn
from typing import Dict, Any


class PrecisionMaster:
    def __init__(self, model: nn.Module, master_dtype: torch.dtype = torch.float32):
        self.master_dtype = master_dtype
        self.master_weights: Dict[str, torch.Tensor] = {}
        
        # Clone full parameters into FP32 master repository
        for name, param in model.named_parameters():
            self.master_weights[name] = param.detach().to(master_dtype).clone()

    def sync_to_model(self, model: nn.Module, target_dtype: torch.dtype = None):
        """Pushes the master FP32 weights into the active model, casting to target_dtype."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.master_weights:
                    src_tensor = self.master_weights[name]
                    dtype = target_dtype if target_dtype is not None else param.dtype
                    param.copy_(src_tensor.to(dtype=dtype, device=param.device))

    def accumulate_delta(self, name: str, delta: torch.Tensor):
        """
        Accumulates a weight update delta into the FP32 master weight.
        Guarantees that tiny updates (e.g. 1e-5) are preserved with full 24-bit mantissa.
        """
        if name not in self.master_weights:
            raise KeyError(f"Parameter {name} not found in master weight repository.")
        
        delta_fp32 = delta.detach().to(dtype=self.master_dtype, device=self.master_weights[name].device)
        self.master_weights[name].add_(delta_fp32)

    def compute_drift_norm(self, initial_weights: Dict[str, torch.Tensor]) -> float:
        """Calculates cumulative Euclidean distance from initial reference weights."""
        total_sq_norm = 0.0
        for name, w_curr in self.master_weights.items():
            if name in initial_weights:
                diff = w_curr - initial_weights[name]
                total_sq_norm += torch.norm(diff).item() ** 2
        return total_sq_norm ** 0.5

