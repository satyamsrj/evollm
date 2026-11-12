import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


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
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_start_idx: int,
    ) -> Tuple[torch.Tensor, int]:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        target_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        if target_start_idx > 0:
            target_mask[:, target_start_idx - 1:] = True
        else:
            target_mask[:, :] = True

        if attention_mask is not None:
            shift_attn = attention_mask[:, 1:].contiguous().bool()
            target_mask = target_mask & shift_attn

        num_tokens = target_mask.sum().item()
        if num_tokens == 0:
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True), 1

        sum_log_prob = target_token_log_probs[target_mask].sum()
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
        pos_sum, pos_len = self.compute_sequence_log_probs(
            model, pos_input_ids, pos_attention_mask, pos_target_idx
        )
        pos_mean = pos_sum / max(pos_len, 1)
        ce_loss = -pos_mean

        if neg_input_ids is not None:
            neg_sum, neg_len = self.compute_sequence_log_probs(
                model, neg_input_ids, neg_attention_mask,
                neg_target_idx if neg_target_idx is not None else 0,
            )
            neg_mean = neg_sum / max(neg_len, 1)
            pref_loss = torch.clamp(self.simpo_margin - (pos_mean - neg_mean), min=0.0)
            return ce_loss + self.preference_weight_mu * pref_loss

        return ce_loss

    def compute_knowledge_loss(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_start_idx: int,
    ) -> torch.Tensor:
        sum_log_p, num_tokens = self.compute_sequence_log_probs(
            model, input_ids, attention_mask, target_start_idx
        )
        return -(sum_log_p / max(num_tokens, 1))

    def compute_ssl_loss(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_start_idx: int,
    ) -> torch.Tensor:
        sum_log_p, num_tokens = self.compute_sequence_log_probs(
            model, input_ids, attention_mask, target_start_idx
        )
        return -(sum_log_p / max(num_tokens, 1))

    def compute_step_reg(self, trainable_params: List[nn.Parameter]) -> torch.Tensor:
        reg = torch.tensor(0.0, device=trainable_params[0].device)
        for p in trainable_params:
            reg = reg + torch.sum(p ** 2)
        return 0.5 * self.lambda_step_reg * reg
