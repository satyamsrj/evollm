from typing import Any, Dict, List
import torch
import torch.nn.functional as F

from evollm.bench.evaluators.base import BaseEvaluator


class MCQEvaluator(BaseEvaluator):
    def evaluate(self, items: List[Dict[str, Any]]) -> Dict[str, float]:
        if not items:
            return {"accuracy": 0.0, "total": 0.0}

        correct = 0
        total = 0

        for item in items:
            prompt = item.get("prompt", "")
            candidates = item.get("candidates", [])
            answer_idx = item.get("answer_idx", 0)

            if not candidates:
                continue

            best_idx, _ = self.select_best_candidate(prompt, candidates)
            if best_idx == answer_idx:
                correct += 1
            total += 1

        acc = correct / max(total, 1)
        return {"accuracy": float(acc), "correct": float(correct), "total": float(total)}

    def evaluate_truthfulqa(self, items: List[Dict[str, Any]]) -> Dict[str, float]:
        if not items:
            return {"mc1_acc": 0.0, "mc2_acc": 0.0, "total": 0.0}

        mc1_correct = 0
        total = 0

        for item in items:
            prompt = f"Q: {item['question']}\nA: "
            mc1_targets = item.get("mc1_targets", {})
            if mc1_targets:
                if isinstance(mc1_targets, dict) and "choices" in mc1_targets and "labels" in mc1_targets:
                    choices = mc1_targets["choices"]
                    labels = mc1_targets["labels"]
                elif isinstance(mc1_targets, dict):
                    choices = list(mc1_targets.keys())
                    labels = list(mc1_targets.values())
                else:
                    choices = []
                    labels = []

                if choices and labels:
                    best_idx, _ = self.select_best_candidate(prompt, choices)
                    if labels[best_idx] == 1:
                        mc1_correct += 1
                    total += 1

        mc1_acc = mc1_correct / max(total, 1)
        return {"mc1_acc": float(mc1_acc), "mc2_acc": float(mc1_acc), "total": float(total)}

