"""
Dual-Precision Parameter Management for EvoLLM.

Prevents catastrophic mantissa underflow and rounding drift by maintaining
an FP32 master weight copy while serving and computing forward passes in
efficient execution dtypes (BF16/FP16/FP32).
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional


class PrecisionMaster:
    def __init__(
        self,
        model: nn.Module,
        master_dtype: torch.dtype = torch.float32,
        target_module_names: Optional[List[str]] = None,
    ):
        self.master_dtype = master_dtype
        self.master_weights: Dict[str, torch.Tensor] = {}
        self.modified_param_names: set = set()
        self.target_module_names = set(target_module_names) if target_module_names else None

        # Clone adapted target parameters into host CPU FP32 master repository to conserve memory
        for name, param in model.named_parameters():
            if self.target_module_names is not None:
                # Match weight parameters belonging to target linear modules (e.g. "...q_proj.weight")
                parts = name.rsplit(".", 2)
                leaf_mod = parts[-2] if len(parts) >= 2 and parts[-1] in ("weight", "bias") else parts[-1]
                if not name.endswith(".weight") or leaf_mod not in self.target_module_names:
                    continue
            self.master_weights[name] = param.detach().cpu().to(master_dtype).clone()

    def sync_to_model(self, model: nn.Module, target_dtype: torch.dtype = None, target_names: Optional[List[str]] = None):
        """Pushes the master FP32 weights into the active model, casting to target_dtype."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if target_names is not None and name not in target_names:
                    continue
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
        
        delta_fp32 = delta.detach().cpu().to(dtype=self.master_dtype)
        self.master_weights[name].add_(delta_fp32)
        self.modified_param_names.add(name)

    def compute_drift_norm(self, initial_weights: Dict[str, torch.Tensor]) -> float:
        """Calculates cumulative Euclidean distance from initial reference weights."""
        total_sq_norm = 0.0
        target_keys = self.modified_param_names if self.modified_param_names else self.master_weights.keys()
        for name in target_keys:
            if name in self.master_weights and name in initial_weights:
                w_curr = self.master_weights[name]
                init_w = initial_weights[name]
                diff = w_curr - init_w
                total_sq_norm += torch.norm(diff).item() ** 2
        return total_sq_norm ** 0.5

    @staticmethod
    def compute_underflow_ratio(delta: torch.Tensor, threshold: float = 1e-4) -> float:
        """
        Calculates the proportion of non-zero delta elements with |x| < threshold,
        which would suffer severe precision loss or underflow under standard FP16.
        """
        abs_delta = delta.abs()
        non_zero = abs_delta > 0
        if not non_zero.any():
            return 0.0
        underflow_elements = (abs_delta < threshold) & non_zero
        return float((underflow_elements.sum().float() / non_zero.sum().float()).item())

