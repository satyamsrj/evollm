import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class EvoLLMConfig:
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj",
        "c_attn", "c_proj", "c_fc", "out_proj", "query_key_value", "dense",
    ])
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    steps_per_conv: int = 2
    base_lr_correction: float = 2e-4
    base_lr_ssl: float = 5e-5
    gradient_clip_norm: float = 1.0

    scgp_top_k: int = 16
    subspace_suppression: float = 0.85
    frobenius_clip_rho: float = 0.5

    lambda_step_reg: float = 1e-4
    lambda_global_reg: float = 1e-5
    
    simpo_margin: float = 0.5
    preference_weight_mu: float = 1.0
    
    min_confidence_alpha: float = 0.1
    max_confidence_alpha: float = 1.0
    novelty_weight: float = 0.6
    explicit_correction_boost: float = 0.4

    max_seq_len: int = 128
    num_cpu_threads: int = 8
    max_step_drift: float = 0.25
    optimizer_type: str = "nsgd"

    master_dtype: str = "float32"
    compute_dtype: str = "float32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


ARCHITECTURE_PRESETS = {
    "pythia-160m": {
        "architecture": "neox",
        "target_modules": ["query_key_value", "dense"],
        "lora_rank": 8,
        "lora_alpha": 16.0,
    },
    "opt-350m": {
        "architecture": "opt",
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "lora_rank": 8,
        "lora_alpha": 16.0,
    },
    "smollm2-360m": {
        "architecture": "llama",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_rank": 8,
        "lora_alpha": 16.0,
    },
    "qwen2.5-0.5b": {
        "architecture": "qwen2",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_rank": 8,
        "lora_alpha": 16.0,
    },
    "granite-moe-1b": {
        "architecture": "granitemoe",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_rank": 8,
        "lora_alpha": 16.0,
    },
}

METHOD_PRESETS = {
    "B0": {"name": "Static Base (theta_0)", "evidence_gating": False, "scgp": False},
    "B1": {"name": "Vanilla LoRA-SGD", "subspace_suppression": 0.0, "frobenius_clip_rho": 1e6, "evidence_gating": False},
    "B2": {"name": "Standard LoRA-AdamW", "subspace_suppression": 0.0, "frobenius_clip_rho": 1e6, "evidence_gating": False, "lambda_global_reg": 1e-3},
    "B3": {"name": "EvoLLM w/o Gating (SCGP-only)", "subspace_suppression": 0.85, "scgp_top_k": 4, "evidence_gating": False},
    "B4": {"name": "CPL-v2 (Full EvoLLM)", "subspace_suppression": 0.85, "scgp_top_k": 4, "evidence_gating": True},
}


