import math
from typing import Any, Dict, List
import torch
import torch.nn.functional as F

from evollm.bench.evaluators.base import BaseEvaluator


class PerplexityEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, device: str = "cpu", max_length: int = 512, stride: int = 256):
        super().__init__(model, tokenizer, device=device)
        self.max_length = max_length
        self.stride = stride

    def evaluate(self, items: List[Dict[str, Any]]) -> Dict[str, float]:
        self.model.eval()
        full_text = "\n\n".join([item["text"] for item in items if item.get("text", "").strip()])
        if not full_text.strip():
            return {"perplexity": 1.0, "cross_entropy": 0.0}

        encodings = self.tokenizer(full_text, return_tensors="pt")
        input_ids_all = encodings["input_ids"]
        seq_len = input_ids_all.size(1)

        nlls = []
        with torch.no_grad():
            for begin_loc in range(0, seq_len, self.stride):
                end_loc = min(begin_loc + self.max_length, seq_len)
                trg_len = end_loc - begin_loc
                input_ids = input_ids_all[:, begin_loc:end_loc].to(self.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                if not torch.isnan(loss):
                    nlls.append(loss.item() * trg_len)

                if end_loc == seq_len:
                    break

        if not nlls:
            return {"perplexity": 1.0, "cross_entropy": 0.0}

        total_tokens = sum(min(begin_loc + self.max_length, seq_len) - begin_loc for begin_loc in range(0, seq_len, self.stride))
        avg_nll = sum(nlls) / max(total_tokens, 1)
        ppl = math.exp(min(avg_nll, 20.0))

        return {"perplexity": float(ppl), "cross_entropy": float(avg_nll)}
