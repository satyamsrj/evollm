import unittest
import torch
import torch.nn as nn
from evollm.bench.evaluators.mcq_eval import MCQEvaluator
from evollm.bench.evaluators.perplexity_eval import PerplexityEvaluator
from evollm.bench.evaluators.qa_eval import QAEvaluator, normalize_answer, f1_score


class DummyTinyModel(nn.Module):
    def __init__(self, vocab_size=50, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.embedding(input_ids)
        logits = self.lm_head(h)

        class Out:
            def __init__(self, l):
                self.logits = l

        return Out(logits)


class DummyTokenizer:
    def __call__(self, text, return_tensors="pt", max_length=128, truncation=True):
        words = text.strip().split()
        if not words:
            words = ["pad"]
        token_ids = [(abs(hash(w)) % 45) + 1 for w in words]
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        t = torch.tensor([token_ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


class TestEvaluators(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = DummyTinyModel()
        self.tokenizer = DummyTokenizer()

    def test_mcq_evaluator_scoring(self):
        evaluator = MCQEvaluator(self.model, self.tokenizer)
        scores = evaluator.compute_sequence_log_probs("What color is the sky? Answer:", [" blue", " green"])
        self.assertEqual(len(scores), 2)
        self.assertIsInstance(scores[0], float)

        items = [
            {
                "prompt": "What color is the sky? Answer:",
                "candidates": [" blue", " green", " red", " yellow"],
                "answer_idx": 0,
            },
            {
                "prompt": "What is 2 + 2? Answer:",
                "candidates": [" 3", " 4", " 5"],
                "answer_idx": 1,
            },
        ]
        results = evaluator.evaluate(items)
        self.assertIn("accuracy", results)
        self.assertIn("total", results)
        self.assertEqual(results["total"], 2.0)
        self.assertTrue(0.0 <= results["accuracy"] <= 1.0)

    def test_perplexity_evaluator(self):
        evaluator = PerplexityEvaluator(self.model, self.tokenizer, max_length=32, stride=16)
        items = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Machine learning enables models to continually learn from data."},
        ]
        results = evaluator.evaluate(items)
        self.assertIn("perplexity", results)
        self.assertIn("cross_entropy", results)
        self.assertGreater(results["perplexity"], 0.0)

    def test_qa_evaluator_and_metrics(self):
        self.assertEqual(normalize_answer("The Apollo 11!"), "apollo 11")
        self.assertEqual(f1_score("apollo 11 mission", "apollo 11"), 0.8)

        evaluator = QAEvaluator(self.model, self.tokenizer)
        items = [
            {
                "question": "What is the capital of France?",
                "target": "Paris",
                "candidates": ["Paris", "London", "Berlin"],
            }
        ]
        results = evaluator.evaluate(items)
        self.assertIn("exact_match", results)
        self.assertIn("token_f1", results)
        self.assertIn("target_log_prob", results)


if __name__ == "__main__":
    unittest.main()

