import abc
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEvaluator(abc.ABC):
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abc.abstractmethod
    def evaluate(self, items: List[Dict[str, Any]]) -> Dict[str, float]:
        pass

    def compute_choice_log_likelihood(self, prompt: str, continuation: str) -> float:
        self.model.eval()
        with torch.no_grad():
            full_text = prompt + continuation
            prompt_enc = self.tokenizer(prompt, return_tensors="pt")
            full_enc = self.tokenizer(full_text, return_tensors="pt")

            prompt_ids = prompt_enc["input_ids"]
            full_ids = full_enc["input_ids"].to(self.device)
            full_mask = full_enc["attention_mask"].to(self.device)

            prompt_len = prompt_ids.shape[1]
            total_len = full_ids.shape[1]

            if total_len <= prompt_len:
                return -100.0

            outputs = self.model(input_ids=full_ids, attention_mask=full_mask)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = full_ids[:, 1:].contiguous()

            log_probs = F.log_softmax(shift_logits, dim=-1)
            target_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            target_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
            target_mask[:, max(0, prompt_len - 1):] = True

            target_tokens = target_mask.sum().item()
            if target_tokens == 0:
                return -100.0

            mean_log_prob = (target_log_probs[target_mask].sum() / target_tokens).item()
            return float(mean_log_prob)

    def compute_sequence_log_probs(self, prompt: str, continuations: List[str]) -> List[float]:
        """Calculates teacher-forced length-normalized sequence log-probabilities for candidate continuations."""
        if not continuations:
            return []
        if len(continuations) == 1:
            return [self.compute_choice_log_likelihood(prompt, continuations[0])]

        self.model.eval()
        with torch.no_grad():
            prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            prompt_len = prompt_ids.shape[1]

            seqs = []
            for c in continuations:
                ids = self.tokenizer(prompt + c, return_tensors="pt")["input_ids"][0]
                seqs.append(ids)

            max_len = max(s.shape[0] for s in seqs)
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0

            batch_size = len(seqs)
            batch_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
            batch_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
            seq_lens = []
            for i, s in enumerate(seqs):
                L = s.shape[0]
                batch_ids[i, :L] = s
                batch_mask[i, :L] = 1
                seq_lens.append(L)

            batch_ids = batch_ids.to(self.device)
            batch_mask = batch_mask.to(self.device)
            outputs = self.model(input_ids=batch_ids, attention_mask=batch_mask)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch_ids[:, 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_lps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            scores: List[float] = []
            start_pos = max(0, prompt_len - 1)
            for i, L in enumerate(seq_lens):
                end_pos = max(0, L - 1)
                if end_pos <= start_pos:
                    scores.append(-100.0)
                else:
                    mean_lp = token_lps[i, start_pos:end_pos].mean().item()
                    scores.append(float(mean_lp))
            return scores

    def select_best_candidate(self, prompt: str, candidates: List[str]) -> Tuple[int, List[float]]:
        scores = self.compute_sequence_log_probs(prompt, candidates)
        best_idx = int(torch.tensor(scores).argmax().item())
        return best_idx, scores

    def generate_completion(self, prompt: str, max_new_tokens: int = 64, stop_strings: Optional[List[str]] = None) -> str:
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.tokenizer, "eos_token_id", None)

            if hasattr(self.model, "generate"):
                gen_kwargs: Dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
                }
                if stop_strings:
                    gen_kwargs["stop_strings"] = stop_strings
                    gen_kwargs["tokenizer"] = self.tokenizer
                try:
                    output_ids = self.model.generate(**gen_kwargs)
                except TypeError:
                    gen_kwargs.pop("stop_strings", None)
                    gen_kwargs.pop("tokenizer", None)
                    output_ids = self.model.generate(**gen_kwargs)
                gen_tokens = output_ids[0, input_ids.shape[1]:]
                return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            else:
                # Deterministic greedy argmax fallback for non-HF test mock models
                curr_ids = input_ids.clone()
                for _ in range(min(max_new_tokens, 16)):
                    out = self.model(curr_ids)
                    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    curr_ids = torch.cat([curr_ids, next_token], dim=1)
                gen_tokens = curr_ids[0, input_ids.shape[1]:]
                if hasattr(self.tokenizer, "decode"):
                    return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                return " ".join([str(t.item()) for t in gen_tokens])


