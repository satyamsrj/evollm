import re
import string
from typing import Any, Dict, List, Set
from evollm.bench.evaluators.base import BaseEvaluator


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(tok), gt_tokens.count(tok)) for tok in common)

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(int(pred_tokens == gt_tokens))
    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def extract_numeric_answer(text: str) -> str:
    clean_text = text.replace(",", "")
    numbers = re.findall(r"[-+]?\d*\.?\d+", clean_text)
    if numbers:
        return numbers[-1]
    return normalize_answer(text)


class QAEvaluator(BaseEvaluator):
    def evaluate(self, items: List[Dict[str, Any]]) -> Dict[str, float]:
        if not items:
            return {"exact_match": 0.0, "token_f1": 0.0, "target_log_prob": 0.0, "total": 0.0}

        log_probs = []
        em_scores = []
        f1_scores = []

        for item in items:
            prompt = item.get("prompt", f"Question: {item.get('question', '')}\nAnswer:")
            target = str(item.get("answer", item.get("target", ""))).strip()
            candidates = item.get("candidates", [])

            lp = self.compute_choice_log_likelihood(prompt, f" {target}")
            log_probs.append(lp)

            if candidates and len(candidates) > 1:
                best_idx, _ = self.select_best_candidate(prompt, [f" {c.strip()}" for c in candidates])
                pred = candidates[best_idx].strip()
            else:
                is_numeric = bool(re.match(r"^[-+]?\d*\.?\d+$", target.replace(",", "")))
                if is_numeric:
                    raw_gen = self.generate_completion(prompt, max_new_tokens=32, stop_strings=["\nQuestion:", "\nMath Problem:"])
                    pred = extract_numeric_answer(raw_gen)
                else:
                    raw_gen = self.generate_completion(prompt, max_new_tokens=16, stop_strings=["\n"])
                    pred = raw_gen.strip().split("\n")[0].strip()

            em = 1.0 if normalize_answer(pred) == normalize_answer(target) else 0.0
            f1 = f1_score(pred, target)

            em_scores.append(em)
            f1_scores.append(f1)

        n = len(items)
        return {
            "exact_match": float(sum(em_scores) / max(n, 1)),
            "token_f1": float(sum(f1_scores) / max(n, 1)),
            "target_log_prob": float(sum(log_probs) / max(n, 1)),
            "total": float(n),
        }

