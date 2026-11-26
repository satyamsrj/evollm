import os
import time
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from evollm.core.config import EvoLLMConfig
from evollm.core.precision_master import PrecisionMaster
from evollm.core.lora import LoRAModelWrapper
from evollm.core.scgp_filter import SubspaceFilter
from evollm.core.loss import LossEngine
from evollm.core.optimizer import StatelessNormalizedSGD
from evollm.core.evidence import EvidenceQualifier, EvidenceRecord


class ContinualLearner:
    def __init__(self, model: nn.Module, tokenizer: Any, config: Optional[EvoLLMConfig] = None):
        self.config = config or EvoLLMConfig()
        self.model = model
        self.tokenizer = tokenizer

        if self.config.device == "cpu" and hasattr(torch, "set_num_threads"):
            torch.set_num_threads(self.config.num_cpu_threads)

        self.model.to(self.config.device)
        self.model.requires_grad_(False)

        self.precision_master = PrecisionMaster(
            model,
            master_dtype=torch.float32,
            target_module_names=self.config.target_modules,
        )
        self.initial_weights = {k: v.detach().cpu().clone() for k, v in self.precision_master.master_weights.items()}

        self.subspace_filter = SubspaceFilter(
            top_k=self.config.scgp_top_k,
            suppression_beta=self.config.subspace_suppression,
            frobenius_rho=self.config.frobenius_clip_rho,
        )
        self.lora_wrapper = LoRAModelWrapper(
            self.model,
            target_module_names=self.config.target_modules,
            r=self.config.lora_rank,
            alpha=self.config.lora_alpha,
        )
        self.loss_engine = LossEngine(
            simpo_margin=self.config.simpo_margin,
            preference_weight_mu=self.config.preference_weight_mu,
            lambda_step_reg=self.config.lambda_step_reg,
            lambda_global_reg=self.config.lambda_global_reg,
        )
        self.evidence_qualifier = EvidenceQualifier()

        if self.config.subspace_suppression > 0.0:
            self._initialize_subspace_bases()

        self.conversation_count: int = 0
        self.drift_history: List[float] = []
        self.skipped_records: List[Dict[str, Any]] = []
        self.persistent_optimizer: Optional[Any] = None

    def _get_rss_mb(self) -> float:
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def _initialize_subspace_bases(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                leaf_name = name.rsplit(".", 1)[-1]
                if leaf_name in self.config.target_modules or any(name.endswith(f".{t}") or name == t for t in self.config.target_modules):
                    self.subspace_filter.extract_and_register_subspace(f"{name}.weight", module.weight)

    def _tokenize_pair(self, query: str, answer: str) -> tuple:
        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        max_prompt_len = max(self.config.max_seq_len // 2, 16)
        prompt = f"User: {query}\nAssistant:"
        full = f"User: {query}\nAssistant: {answer}"
        try:
            full_enc = self.tokenizer(
                full,
                return_tensors="pt",
                max_length=self.config.max_seq_len,
                truncation=True,
            )
            prompt_enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_prompt_len,
                truncation=True,
            )
        except TypeError:
            full_enc = self.tokenizer(full, return_tensors="pt")
            prompt_enc = self.tokenizer(prompt, return_tensors="pt")

        input_ids = full_enc["input_ids"].to(self.config.device)
        attention_mask = full_enc["attention_mask"].to(self.config.device)
        target_start_idx = min(prompt_enc["input_ids"].shape[1], max(input_ids.shape[1] - 1, 0))
        return input_ids, attention_mask, max(target_start_idx, 0)

    def _effective_lr(self, record: EvidenceRecord) -> float:
        base = (
            self.config.base_lr_correction
            if record.evidence_class.value == "CORR"
            else self.config.base_lr_ssl
        )
        alpha = self.config.min_confidence_alpha + record.confidence * (
            self.config.max_confidence_alpha - self.config.min_confidence_alpha
        )
        return float(alpha * base)

    def learn_from_evidence(
        self,
        record: EvidenceRecord,
        query: str,
        incorrect_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        self.conversation_count += 1

        if not record.accept_for_learning:
            self.skipped_records.append({
                "conversation_idx": self.conversation_count,
                "reason": record.rejection_reason,
                "evidence_class": record.evidence_class.value,
            })
            return {
                "status": f"SKIPPED_{record.rejection_reason}",
                "conversation_idx": self.conversation_count,
                "step_time_ms": (time.perf_counter() - t_start) * 1000.0,
                "rss_mb": self._get_rss_mb(),
            }

        pos_ids, pos_mask, target_idx = self._tokenize_pair(query, record.text)

        neg_ids = neg_mask = None
        if incorrect_answer:
            neg_ids, neg_mask, _ = self._tokenize_pair(query, incorrect_answer)

        lr = self._effective_lr(record)

        self.lora_wrapper.wrap()
        self.lora_wrapper.reset_adapters()
        trainable_params = self.lora_wrapper.get_trainable_parameters()

        if not trainable_params:
            self.lora_wrapper.unwrap()
            return {
                "status": "ERROR_NO_LORA_PARAMS",
                "conversation_idx": self.conversation_count,
                "step_time_ms": (time.perf_counter() - t_start) * 1000.0,
                "rss_mb": self._get_rss_mb(),
            }

        if getattr(self.config, "optimizer_type", "nsgd") == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params, lr=lr, weight_decay=self.config.lambda_global_reg
            )
        else:
            optimizer = StatelessNormalizedSGD(
                trainable_params, lr=lr, max_grad_norm=self.config.gradient_clip_norm
            )

        self.model.train()
        total_loss = 0.0

        for _ in range(self.config.steps_per_conv):
            optimizer.zero_grad()

            if record.loss_target == "L_CORR":
                loss = self.loss_engine.compute_correction_loss(
                    self.model, pos_ids, pos_mask, target_idx, neg_ids, neg_mask,
                    target_idx if neg_ids is not None else None,
                )
            elif record.loss_target == "L_SSL":
                loss = self.loss_engine.compute_ssl_loss(
                    self.model, pos_ids, pos_mask, target_idx
                )
            else:
                loss = self.loss_engine.compute_knowledge_loss(
                    self.model, pos_ids, pos_mask, target_idx
                )

            step_loss = loss + self.loss_engine.compute_step_reg(trainable_params)
            step_loss.backward()
            optimizer.step()
            total_loss += step_loss.item()

        del step_loss, loss, optimizer
        if self.config.device == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            deltas = self.lora_wrapper.get_all_deltas(device="cpu" if self.config.device == "cuda" else None)
            snapshot = {k: self.precision_master.master_weights[k].clone() for k in deltas.keys() if k in self.precision_master.master_weights}

            clean_deltas = {}
            is_finite = True
            self.subspace_filter.reset_step_telemetry()

            for name, delta in deltas.items():
                if not torch.isfinite(delta).all():
                    is_finite = False
                    break
                base_name = name.replace(".weight", "")
                lora_layer = self.lora_wrapper.lora_modules.get(base_name)
                clean_delta = self.subspace_filter.filter_delta(name, delta, lora_layer=lora_layer)
                if not torch.isfinite(clean_delta).all():
                    is_finite = False
                    break
                clean_deltas[name] = clean_delta

            del deltas
            self.lora_wrapper.unwrap()

            if not is_finite:
                for k, v in snapshot.items():
                    self.precision_master.master_weights[k] = v
                self.precision_master.sync_to_model(self.model, target_names=list(snapshot.keys()))
                del snapshot, clean_deltas
                return {
                    "status": "ROLLBACK_NUMERIC_INSTABILITY",
                    "conversation_idx": self.conversation_count,
                    "step_time_ms": (time.perf_counter() - t_start) * 1000.0,
                    "rss_mb": self._get_rss_mb(),
                }

            for name, clean_delta in clean_deltas.items():
                self.precision_master.accumulate_delta(name, clean_delta)

            self.precision_master.sync_to_model(self.model, target_names=list(clean_deltas.keys()))

            drift = self.precision_master.compute_drift_norm(self.initial_weights)
            prev_drift = self.drift_history[-1] if self.drift_history else 0.0
            step_drift = abs(drift - prev_drift)

            if step_drift > self.config.max_step_drift:
                for k, v in snapshot.items():
                    self.precision_master.master_weights[k] = v
                self.precision_master.sync_to_model(self.model, target_names=list(snapshot.keys()))
                del snapshot, clean_deltas
                return {
                    "status": "ROLLBACK_DRIFT_SPIKE",
                    "conversation_idx": self.conversation_count,
                    "step_drift": step_drift,
                    "step_time_ms": (time.perf_counter() - t_start) * 1000.0,
                    "rss_mb": self._get_rss_mb(),
                }

            del snapshot
            self.drift_history.append(drift)

            scgp_telemetry = self.subspace_filter.get_last_telemetry()
            underflow_ratios = [self.precision_master.compute_underflow_ratio(d) for d in clean_deltas.values()]
            avg_underflow = float(sum(underflow_ratios) / len(underflow_ratios)) if underflow_ratios else 0.0
            del clean_deltas

        return {
            "status": "SUCCESS",
            "conversation_idx": self.conversation_count,
            "loss_target": record.loss_target,
            "evidence_class": record.evidence_class.value,
            "loss": total_loss / self.config.steps_per_conv,
            "effective_lr": lr,
            "confidence_alpha": self.config.min_confidence_alpha + record.confidence * (self.config.max_confidence_alpha - self.config.min_confidence_alpha),
            "drift_norm": drift,
            "fp32_underflow_ratio": avg_underflow,
            "cos_phi": scgp_telemetry.get("cos_phi", 0.0),
            "core_energy_rejected_pct": scgp_telemetry.get("core_energy_rejected_pct", 0.0),
            "step_time_ms": (time.perf_counter() - t_start) * 1000.0,
            "rss_mb": self._get_rss_mb(),
        }

    def learn_from_conversation(
        self,
        query: str,
        correct_answer: str,
        incorrect_answer: Optional[str] = None,
        is_correction: bool = False,
        source_reference: Optional[str] = None,
        allow_ssl: bool = False,
    ) -> Dict[str, Any]:
        text = correct_answer if not is_correction else f"correction: {correct_answer}"
        record = self.evidence_qualifier.qualify(
            text, source_reference=source_reference, allow_ssl=allow_ssl
        )

        if is_correction and not record.accept_for_learning:
            from evollm.core.evidence import EvidenceClass, EvidenceRecord
            record = EvidenceRecord(
                text=correct_answer,
                evidence_class=EvidenceClass.EXPLICIT_CORRECTION,
                confidence=0.9,
                provenance_verified=source_reference is not None,
                source_reference=source_reference,
                accept_for_learning=True,
                rejection_reason=None,
                loss_target="L_CORR",
            )

        return self.learn_from_evidence(record, query=query, incorrect_answer=incorrect_answer)

