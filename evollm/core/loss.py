"""
Loss Functions & Signal Triage for Continual Learning (CPL-v2).

Implements:
1. Length-Normalized Direct Preference & SimPO Margin Loss
2. Cross-Entropy Loss with Length Normalization
3. Triaged Selective Learning (TSL) for Self-Supervised Updates
4. Dual-Scale Elastic Anchor Regularization
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LossEngine:
    def __init__(
        self,
        simpo_margin: float = 0.5,
        preference_weight_mu: float = 1.0,
        lambda_step_reg: float = 1e-4,
        lambda_global_reg: float = 1e-5,
    ):
        self.simpo_margin = simpo_margin
        self.preference_weight_mu = preference_weight_mu
        self.lambda_step_reg = lambda_step_reg
        self.lambda_global_reg = lambda_global_reg

    def compute_sequence_log_probs(
        self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_start_idx: int
    ) -> Tuple[torch.Tensor, int]:
        """
        Computes the token log-probabilities for the target portion of a prompt+target sequence.
        Returns: (sum_log_probs, num_target_tokens)
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute log-softmax over vocab
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log prob of true tokens
        target_token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out prompt tokens (before target_start_idx)
        # Shift index by 1 due to shifted labels
        target_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        if target_start_idx > 0:
            target_mask[:, target_start_idx - 1 :] = True
        else:
            target_mask[:, :] = True

        # Also apply attention_mask
        if attention_mask is not None:
            shift_attn = attention_mask[:, 1:].contiguous().bool()
            target_mask = target_mask & shift_attn

        num_tokens = target_mask.sum().item()
        if num_tokens == 0:
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True), 1

        selected_log_probs = target_token_log_probs[target_mask]
        sum_log_prob = selected_log_probs.sum()
        return sum_log_prob, num_tokens

    def compute_correction_loss(
        self,
        model: nn.Module,
        pos_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        pos_target_idx: int,
        neg_input_ids: Optional[torch.Tensor] = None,
        neg_attention_mask: Optional[torch.Tensor] = None,
        neg_target_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Computes length-normalized Cross-Entropy + SimPO Preference Loss.
        L = L_CE(a+) + mu * max(0, margin - (logP(a+)/|a+| - logP(a-)/|a-|))
        """
        # Positive answer log-likelihood
        pos_sum_log_p, pos_len = self.compute_sequence_log_probs(
            model, pos_input_ids, pos_attention_mask, pos_target_idx
        )
        pos_mean_log_p = pos_sum_log_p / max(pos_len, 1)
        ce_loss = -pos_mean_log_p

        # If negative (flawed) answer is provided, compute length-normalized preference margin
        if neg_input_ids is not None:
            neg_sum_log_p, neg_len = self.compute_sequence_log_probs(
                model, neg_input_ids, neg_attention_mask, neg_target_idx if neg_target_idx is not None else 0
            )
            neg_mean_log_p = neg_sum_log_p / max(neg_len, 1)

            # SimPO margin: target (pos_mean - neg_mean) >= margin
            diff = pos_mean_log_p - neg_mean_log_p
            pref_loss = torch.clamp(self.simpo_margin - diff, min=0.0)
            return ce_loss + (self.preference_weight_mu * pref_loss)

        return ce_loss

    def compute_ssl_loss(
        self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_start_idx: int
    ) -> torch.Tensor:
        """Computes length-normalized self-supervised cross-entropy on verified factual assertions."""
        sum_log_p, num_tokens = self.compute_sequence_log_probs(
            model, input_ids, attention_mask, target_start_idx
        )
        return - (sum_log_p / max(num_tokens, 1))

    def compute_step_reg(self, trainable_params: List[nn.Parameter]) -> torch.Tensor:
        """L2 regularization on LoRA parameters to prevent excessive perturbation."""
        reg = torch.tensor(0.0, device=trainable_params[0].device)
        for p in trainable_params:
            reg = reg + torch.sum(p ** 2)
        return 0.5 * self.lambda_step_reg * reg


class SignalTriager:
    """
    Parses conversational turns to separate:
    - Informative factual assertions (learning candidate)
    - Ambiguous or questioning turns (discarded from SSL to prevent sycophancy/decay)
    - Explicit corrections
    """
    QUESTION_PATTERNS = [
        r"^(what|who|where|when|why|how|can|is|are|do|does|will|could|would)\b",
        r"\?\s*$"
    ]
    CORRECTION_TRIGGERS = [
        r"\b(no|wrong|incorrect|actually|instead|not true|correction)\b"
    ]
    ASSERTION_INDICATORS = [
        r"\b(is|are|was|were|has|have|uses|runs on|named|located in|updated to)\b"
    ]

    @classmethod
    def is_question_or_query(cls, text: str) -> bool:
        t = text.strip().lower()
        for pat in cls.QUESTION_PATTERNS:
            if re.search(pat, t):
                return True
        return False

    @classmethod
    def is_explicit_correction(cls, text: str) -> bool:
        t = text.strip().lower()
        for pat in cls.CORRECTION_TRIGGERS:
            if re.search(pat, t):
                return True
        return False

    @classmethod
    def is_factual_assertion(cls, text: str) -> bool:
        if cls.is_question_or_query(text):
            return False
        t = text.strip().lower()
        for pat in cls.ASSERTION_INDICATORS:
            if re.search(pat, t):
                return True
        return False

