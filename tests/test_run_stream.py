import json
import os
import shutil
import tempfile
import unittest
import torch
import torch.nn as nn

from evollm.core.config import EvoLLMConfig
from experiments.scale_5arch_15gb.run_stream import (
    build_method_config,
    compute_model_param_hash,
    load_yaml,
    load_or_create_stream_events,
    load_fast_eval_probes,
    run_architecture_stream,
)


class MockLinearLM(nn.Module):
    def __init__(self, vocab_size=50, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.query_key_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dense = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        h = self.dense(self.query_key_value(x))
        logits = self.lm_head(h)

        class Output:
            def __init__(self, l):
                self.logits = l
        return Output(logits)


class MockTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<pad>"

    def __call__(self, text, return_tensors="pt", max_length=128, truncation=True):
        tokens = [hash(w) % 40 + 1 for w in text.split()]
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        if not tokens:
            tokens = [1]
        t = torch.tensor([tokens], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


class TestRunStream(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model = MockLinearLM()
        self.tokenizer = MockTokenizer()
        self.config = EvoLLMConfig(
            target_modules=["query_key_value", "dense"],
            device="cpu",
            num_cpu_threads=2,
            lora_rank=4,
            lora_alpha=8.0,
            max_seq_len=64,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_method_configs(self):
        cfg_b0 = build_method_config(self.config, "B0")
        self.assertEqual(cfg_b0.subspace_suppression, self.config.subspace_suppression)

        cfg_b1 = build_method_config(self.config, "B1")
        self.assertEqual(cfg_b1.subspace_suppression, 0.0)
        self.assertEqual(cfg_b1.frobenius_clip_rho, 1e6)

        cfg_b2 = build_method_config(self.config, "B2")
        self.assertEqual(cfg_b2.lambda_global_reg, 1e-3)

        cfg_b3 = build_method_config(self.config, "B3")
        self.assertEqual(cfg_b3.subspace_suppression, 0.85)
        self.assertEqual(cfg_b3.frobenius_clip_rho, 0.5)

        cfg_b4 = build_method_config(self.config, "B4")
        self.assertEqual(cfg_b4.subspace_suppression, 0.85)

    def test_compute_model_param_hash_invariance(self):
        h1 = compute_model_param_hash(self.model)
        h2 = compute_model_param_hash(self.model)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_stream_events_loading_with_manifest(self):
        slices_dir = os.path.join(self.temp_dir, "data_slices")
        manifests_dir = os.path.join(self.temp_dir, "data_manifests")
        os.makedirs(slices_dir, exist_ok=True)
        os.makedirs(manifests_dir, exist_ok=True)

        slice_file = os.path.join(slices_dir, "test_train.jsonl")
        man_file = os.path.join(manifests_dir, "test_manifest.json")
        sample_ev = {
            "conversation_idx": 1,
            "query": "What is Python?",
            "answer": "Python is a high-level programming language.",
            "loss_target": "L_KNOW",
            "evidence_class": "VERIFIED_FACT",
            "confidence": 0.95,
            "source_reference": "ref1",
        }
        with open(slice_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_ev) + "\n")

        import hashlib
        hasher = hashlib.sha256()
        with open(slice_file, "rb") as f:
            hasher.update(f.read())
        manifest = {
            "dataset_id": "test_ds",
            "sample_count": 1,
            "sha256": hasher.hexdigest(),
        }
        with open(man_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        loaded = load_or_create_stream_events({}, max_events=10, slices_dir=slices_dir, manifests_dir=manifests_dir)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["query"], "What is Python?")

    def test_run_architecture_stream_step_b4(self):
        events = [
            {
                "conversation_idx": 1,
                "query": "What is the speed of light?",
                "answer": "The speed of light is 299792458 meters per second.",
                "incorrect_answer": None,
                "loss_target": "L_KNOW",
                "evidence_class": "VERIFIED_FACT",
                "confidence": 0.95,
                "source_reference": "canonical_stream",
            }
        ]
        probes = {
            "wikitext103": [{"text": "Hello world from test probe."}]
        }

        result = run_architecture_stream(
            model=self.model,
            tokenizer=self.tokenizer,
            model_name="test_pythia",
            method="B4",
            events=events,
            eval_probes=probes,
            config=self.config,
            eval_checkpoints=[1],
            output_dir=self.temp_dir,
        )

        self.assertEqual(result["method"], "B4")
        self.assertEqual(result["total_events"], 1)
        self.assertTrue(os.path.exists(result["metrics_jsonl"]))
        self.assertGreater(len(result["eval_records"]), 0)


if __name__ == "__main__":
    unittest.main()
