"""
EvoLLM Continuous Learning Engine (CPL-v2).

Orchestrates conversation triage, LoRA adaptation, subspace spectral filtering,
dual-precision master accumulation, and hot-swapping into the serving model.
"""
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from evollm.core.config import EvoLLMConfig
from evollm.core.precision_master import PrecisionMaster
from evollm.core.lora import LoRAModelWrapper
from evollm.core.scgp_filter import SubspaceFilter
from evollm.core.loss import LossEngine, SignalTriager
from evollm.core.optimizer import StatelessNormalizedSGD


class ContinualLearner:
    def __init__(self, model: nn.Module, tokenizer: Any, config: Optional[EvoLLMConfig] = None):
        self.config = config or EvoLLMConfig()
        self.model = model
        self.tokenizer = tokenizer
        
        # 1. Dual-precision master repository (FP32)
        self.precision_master = PrecisionMaster(model, master_dtype=torch.float32)
        self.initial_weights = {k: v.clone() for k, v in self.precision_master.master_weights.items()}

        # 2. Subspace-Constrained Gradient Projection (SCGP) Filter
        self.subspace_filter = SubspaceFilter(
            top_k=self.config.scgp_top_k,
            suppression_beta=self.config.subspace_suppression,
            frobenius_rho=self.config.frobenius_clip_rho,
        )

        # 3. LoRA Model Wrapper
        self.lora_wrapper = LoRAModelWrapper(
            self.model,
            target_module_names=self.config.target_modules,
            r=self.config.lora_rank,
            alpha=self.config.lora_alpha,
        )

        # 4. Loss Engine
        self.loss_engine = LossEngine(
            simpo_margin=self.config.simpo_margin,
            preference_weight_mu=self.config.preference_weight_mu,
            lambda_step_reg=self.config.lambda_step_reg,
            lambda_global_reg=self.config.lambda_global_reg,
        )

        # Precompute subspace bases for target layers
        self._initialize_subspace_bases()

        # Metrics log
        self.conversation_count = 0
        self.drift_history: List[float] = []

    def _initialize_subspace_bases(self):
        """Pre-computes top-K singular vectors for target projection weights."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if any(target in name for target in self.config.target_modules):
                    weight_key = f"{name}.weight"
                    self.subspace_filter.extract_and_register_subspace(weight_key, module.weight)

    def compute_confidence_factor(self, is_explicit_corr: bool, text_novelty: float = 0.5) -> float:
        """
        Dynamically scales the conversation update rate based on novelty and feedback certainty.
        """
        alpha = self.config.min_confidence_alpha
        if is_explicit_corr:
            alpha += self.config.explicit_correction_boost
        alpha += text_novelty * self.config.novelty_weight
        return float(torch.clamp(torch.tensor(alpha), self.config.min_confidence_alpha, self.config.max_confidence_alpha).item())

    def learn_from_conversation(
        self,
        query: str,
        correct_answer: str,
        incorrect_answer: Optional[str] = None,
        is_correction: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes a single-conversation learning event (CPL-v2).
        Extracts signals, optimizes LoRA adapters, applies SCGP filter, merges into FP32 master weights.
        """
        self.conversation_count += 1
        
        # 1. Triage signals
        if not is_correction and not SignalTriager.is_factual_assertion(correct_answer):
            # No valid factual learning signal detected
            return {"status": "SKIPPED_NO_FACTUAL_SIGNAL", "conversation_idx": self.conversation_count}

        # 2. Tokenize prompt + positive completion
        full_pos_text = f"User: {query}\nAssistant: {correct_answer}"
        prompt_text = f"User: {query}\nAssistant:"
        
        pos_enc = self.tokenizer(full_pos_text, return_tensors="pt")
        prompt_enc = self.tokenizer(prompt_text, return_tensors="pt")
        
        pos_input_ids = pos_enc["input_ids"].to(self.config.device)
        pos_attention_mask = pos_enc["attention_mask"].to(self.config.device)
        target_start_idx = prompt_enc["input_ids"].shape[1]

        neg_input_ids = None
        neg_attention_mask = None
        if incorrect_answer:
            full_neg_text = f"User: {query}\nAssistant: {incorrect_answer}"
            neg_enc = self.tokenizer(full_neg_text, return_tensors="pt")
            neg_input_ids = neg_enc["input_ids"].to(self.config.device)
            neg_attention_mask = neg_enc["attention_mask"].to(self.config.device)

        # 3. Dynamic Learning Rate Scaling
        alpha_t = self.compute_confidence_factor(is_explicit_corr=is_correction)
        eff_lr = alpha_t * (self.config.base_lr_correction if is_correction else self.config.base_lr_ssl)

        # 4. Attach LoRA adapters
        self.lora_wrapper.wrap()
        self.lora_wrapper.reset_adapters()
        trainable_params = self.lora_wrapper.get_trainable_parameters()

        if not trainable_params:
            self.lora_wrapper.unwrap()
            return {"status": "ERROR_NO_LORA_PARAMS", "conversation_idx": self.conversation_count}

        # 5. Stateless Normalized Cautious Optimizer
        optimizer = StatelessNormalizedSGD(
            trainable_params,
            lr=eff_lr,
            max_grad_norm=self.config.gradient_clip_norm
        )

        # 6. Execute k gradient steps
        self.model.train()
        total_loss_val = 0.0
        
        for step in range(self.config.steps_per_conv):
            optimizer.zero_grad()
            
            # Loss calculation
            loss = self.loss_engine.compute_correction_loss(
                self.model,
                pos_input_ids=pos_input_ids,
                pos_attention_mask=pos_attention_mask,
                pos_target_idx=target_start_idx,
                neg_input_ids=neg_input_ids,
                neg_attention_mask=neg_attention_mask,
                neg_target_idx=target_start_idx if neg_input_ids is not None else None,
            )
            
            # Step regularization
            reg_loss = self.loss_engine.compute_step_reg(trainable_params)
            step_loss = loss + reg_loss
            
            step_loss.backward()
            optimizer.step()
            total_loss_val += step_loss.item()

        # 7. Extract updates and apply SCGP Subspace Filter
        deltas = self.lora_wrapper.get_all_deltas()
        filtered_deltas = {}
        
        for name, delta in deltas.items():
            base_module_name = name.replace(".weight", "")
            lora_layer = self.lora_wrapper.lora_modules.get(base_module_name, None)
            
            clean_delta = self.subspace_filter.filter_delta(
                name=name,
                delta_w=delta,
                lora_layer=lora_layer
            )
            filtered_deltas[name] = clean_delta

        # 8. Detach LoRA adapters before weight fusion
        self.lora_wrapper.unwrap()

        # 9. Master Weight Fusion (FP32)
        for name, clean_delta in filtered_deltas.items():
            self.precision_master.accumulate_delta(name, clean_delta)

        # 10. Sync updated master weights to active serving model
        self.precision_master.sync_to_model(self.model)

        # Record drift metrics
        drift = self.precision_master.compute_drift_norm(self.initial_weights)
        self.drift_history.append(drift)

        return {
            "status": "SUCCESS",
            "conversation_idx": self.conversation_count,
            "loss": total_loss_val / self.config.steps_per_conv,
            "confidence_alpha": alpha_t,
            "effective_lr": eff_lr,
            "drift_norm": drift,
        }

