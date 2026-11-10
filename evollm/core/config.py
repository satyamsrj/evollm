"""
Configuration specifications for EvoLLM: Conversation-Driven Weight-Only Continual Learning (CPL-v2).
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvoLLMConfig:
    # Model architecture & adapter settings
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj",
        "c_attn", "c_proj", "c_fc"  # covers LLaMA, Qwen, Mistral, and GPT-2 styles
    ])
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    # Optimization budget per conversation
    steps_per_conv: int = 2
    base_lr_correction: float = 2e-4
    base_lr_ssl: float = 5e-5
    gradient_clip_norm: float = 1.0

    # Stability & Subspace Projection (SCGP)
    scgp_top_k: int = 16              # Top principal singular vectors preserved
    subspace_suppression: float = 0.85 # Beta factor: dampens interference with base subspace
    frobenius_clip_rho: float = 0.5    # Max norm of weight delta per layer per dialogue

    # Elastic Anchor Regularization
    lambda_step_reg: float = 1e-4      # Step-wise L2 norm penalty on LoRA params
    lambda_global_reg: float = 1e-5    # Long-term drift penalty against theta_0
    
    # Preference Optimization (Length-Normalized Margin)
    simpo_margin: float = 0.5
    preference_weight_mu: float = 1.0
    
    # Selective SSL & TSL Confidence
    min_confidence_alpha: float = 0.1
    max_confidence_alpha: float = 1.0
    novelty_weight: float = 0.6
    explicit_correction_boost: float = 0.4

    # Precision & Execution
    master_dtype: str = "float32"      # FP32 master weights prevent mantissa underflow
    compute_dtype: str = "float32"     # float32 or bfloat16
    device: str = "cpu"

