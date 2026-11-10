"""
Native Low-Rank Adaptation (LoRA) Layer & Model Wrapper.
Provides exact parameterization, extraction of low-rank deltas, and clean merging.
"""
import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable low-rank decomposition matrices
        # A: r x in_features, initialized with Kaiming uniform
        # B: out_features x r, initialized to zero (so initial delta is zero)
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=base_layer.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=base_layer.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        # Compute low-rank path: (x @ A^T) @ B^T * scaling
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out

    def get_delta_weight(self) -> torch.Tensor:
        """Computes effective delta matrix: ΔW = scaling * (B @ A) [shape: out_features x in_features]"""
        return (self.lora_B @ self.lora_A) * self.scaling

    def reset_lora_parameters(self):
        """Resets LoRA parameters for the next conversation."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


class LoRAModelWrapper:
    """Wraps target linear projections across any PyTorch causal language model."""
    def __init__(self, model: nn.Module, target_module_names: List[str], r: int = 8, alpha: float = 16.0):
        self.model = model
        self.target_module_names = target_module_names
        self.r = r
        self.alpha = alpha
        self.lora_modules: Dict[str, LoRALinear] = {}
        self._wrapped = False

    def wrap(self):
        """Attaches LoRA adaptors to specified modules."""
        if self._wrapped:
            return

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                # Check if matches any target submodule name
                if any(target in name for target in self.target_module_names):
                    parent_name, child_name = self._get_parent_and_child_name(name)
                    parent = self.model.get_submodule(parent_name) if parent_name else self.model
                    
                    lora_layer = LoRALinear(module, r=self.r, lora_alpha=self.alpha)
                    setattr(parent, child_name, lora_layer)
                    self.lora_modules[name] = lora_layer

        self._wrapped = True

    def unwrap(self):
        """Removes LoRA adaptors and restores base linear modules."""
        if not self._wrapped:
            return

        for name, lora_layer in self.lora_modules.items():
            parent_name, child_name = self._get_parent_and_child_name(name)
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            setattr(parent, child_name, lora_layer.base_layer)

        self.lora_modules.clear()
        self._wrapped = False

    def reset_adapters(self):
        """Resets all adapter matrices to zero effective delta."""
        for layer in self.lora_modules.values():
            layer.reset_lora_parameters()

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Returns only the trainable LoRA parameters."""
        params = []
        for layer in self.lora_modules.values():
            params.extend([layer.lora_A, layer.lora_B])
        return params

    def get_all_deltas(self) -> Dict[str, torch.Tensor]:
        """Extracts ΔW for each adapted layer."""
        deltas = {}
        for name, layer in self.lora_modules.items():
            weight_name = f"{name}.weight"
            deltas[weight_name] = layer.get_delta_weight()
        return deltas

    def _get_parent_and_child_name(self, full_name: str) -> Tuple[str, str]:
        parts = full_name.rsplit(".", 1)
        if len(parts) == 1:
            return "", parts[0]
        return parts[0], parts[1]

