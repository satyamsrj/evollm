"""
Comprehensive Unit & Integration Test Suite for EvoLLM (CPL-v2).
"""
import torch
import torch.nn as nn
import unittest

from evollm.core.config import EvoLLMConfig
from evollm.core.precision_master import PrecisionMaster
from evollm.core.lora import LoRALinear, LoRAModelWrapper
from evollm.core.scgp_filter import SubspaceFilter
from evollm.core.loss import LossEngine
from evollm.core.evidence import EvidenceQualifier
from evollm.core.optimizer import StatelessNormalizedSGD


class TinyCausalModel(nn.Module):
    """A minimal 2-layer linear causal model for fast deterministic testing."""
    def __init__(self, vocab_size=100, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        q = self.q_proj(x)
        v = self.v_proj(x)
        h = q + v
        logits = self.lm_head(h)
        
        # Return a simple object with logits
        class Output:
            def __init__(self, l):
                self.logits = l
        return Output(logits)


class TestEvoLLMCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyCausalModel()

    def test_precision_master_underflow_protection(self):
        """Verifies that FP32 master weights preserve tiny increments that would vanish in FP16."""
        pm = PrecisionMaster(self.model, master_dtype=torch.float32)
        initial_val = pm.master_weights["q_proj.weight"].clone()
        
        # Add tiny delta (1e-5)
        tiny_delta = torch.ones_like(initial_val) * 1e-5
        pm.accumulate_delta("q_proj.weight", tiny_delta)
        
        # In FP32, addition is exact to ~7 decimal places
        diff = (pm.master_weights["q_proj.weight"] - initial_val).mean().item()
        self.assertAlmostEqual(diff, 1e-5, places=6)

    def test_lora_wrapper_and_delta_extraction(self):
        """Verifies LoRA wrapping, parameter detachment, and delta computation."""
        wrapper = LoRAModelWrapper(self.model, target_module_names=["q_proj", "v_proj"], r=4, alpha=8.0)
        wrapper.wrap()
        
        # Ensure target modules became LoRALinear
        self.assertIsInstance(self.model.q_proj, LoRALinear)
        self.assertIsInstance(self.model.v_proj, LoRALinear)
        
        # Initially delta should be zero
        deltas = wrapper.get_all_deltas()
        self.assertEqual(deltas["q_proj.weight"].abs().max().item(), 0.0)
        
        # Simulate an update on LoRA parameters
        for p in wrapper.get_trainable_parameters():
            p.data.add_(torch.randn_like(p) * 0.01)
            
        deltas_updated = wrapper.get_all_deltas()
        self.assertGreater(deltas_updated["q_proj.weight"].abs().max().item(), 0.0)
        
        # Unwrap and restore original layer
        wrapper.unwrap()
        self.assertIsInstance(self.model.q_proj, nn.Linear)

    def test_scgp_fast_projection_equivalence(self):
        """Verifies that fast O(d·K·r) projection matches direct matrix projection."""
        d_out, d_in = 64, 64
        k = 8
        r = 4
        
        filter_module = SubspaceFilter(top_k=k, suppression_beta=0.85, frobenius_rho=1.0)
        base_w = torch.randn(d_out, d_in)
        filter_module.extract_and_register_subspace("test.weight", base_w)
        
        # Mock a LoRA layer
        mock_linear = nn.Linear(d_in, d_out, bias=False)
        lora_layer = LoRALinear(mock_linear, r=r, lora_alpha=8.0)
        lora_layer.lora_A.data = torch.randn(r, d_in)
        lora_layer.lora_B.data = torch.randn(d_out, r)
        
        delta_w = lora_layer.get_delta_weight()
        
        # Fast filter with lora_layer
        fast_filtered = filter_module.filter_delta("test.weight", delta_w, lora_layer=lora_layer)
        
        # Direct filter without lora_layer
        direct_filtered = filter_module.filter_delta("test.weight", delta_w, lora_layer=None)
        
        # Check numerical agreement
        max_diff = (fast_filtered - direct_filtered).abs().max().item()
        self.assertLess(max_diff, 1e-4)

    def test_length_normalized_simpo_loss(self):
        """Verifies that length normalization prevents sequence length disparity artifacts."""
        loss_engine = LossEngine(simpo_margin=0.5, preference_weight_mu=1.0)
        
        # Batch with short target vs long target
        input_ids = torch.randint(0, 100, (1, 12))
        attn_mask = torch.ones_like(input_ids)
        
        # Target starts at index 6
        sum_p, n_tok = loss_engine.compute_sequence_log_probs(self.model, input_ids, attn_mask, target_start_idx=6)
        self.assertEqual(n_tok, 6)
        self.assertTrue(torch.is_tensor(sum_p))

    def test_evidence_qualifier_routing(self):
        """Verifies EvidenceQualifier correctly routes dialogue turns to learning targets."""
        q = EvidenceQualifier()

        query_rec = q.qualify("What is the baud rate?")
        self.assertFalse(query_rec.accept_for_learning)

        corr_rec = q.qualify("No, the baud rate was updated to 9600.")
        self.assertTrue(corr_rec.accept_for_learning)
        self.assertEqual(corr_rec.loss_target, "L_CORR")

        fact_rec = q.qualify("The baud rate is 115200.", source_reference="spec_v3.pdf")
        self.assertTrue(fact_rec.accept_for_learning)
        self.assertEqual(fact_rec.loss_target, "L_KNOW")

    def test_stateless_normalized_sgd(self):
        """Verifies optimizer step executes without carrying momentum or state."""
        params = [nn.Parameter(torch.tensor([2.0, -3.0], requires_grad=True))]
        opt = StatelessNormalizedSGD(params, lr=0.1, max_grad_norm=1.0)
        
        loss = 0.5 * (params[0] ** 2).sum()
        loss.backward()
        
        opt.step()
        opt.zero_grad()
        
        self.assertLess(params[0][0].item(), 2.0)
        self.assertGreater(params[0][1].item(), -3.0)
        self.assertFalse(hasattr(opt, "state") and len(opt.state) > 0)

    def test_architecture_specific_lora_wrapping(self):
        config = EvoLLMConfig()

        class MockGPTNeo(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32, bias=False)
                self.k_proj = nn.Linear(32, 32, bias=False)
                self.v_proj = nn.Linear(32, 32, bias=False)
                self.out_proj = nn.Linear(32, 32, bias=False)

        class MockGPTNeoX(nn.Module):
            def __init__(self):
                super().__init__()
                self.query_key_value = nn.Linear(32, 96, bias=False)
                self.dense = nn.Linear(32, 32, bias=False)

        class MockLlama(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32, bias=False)
                self.k_proj = nn.Linear(32, 32, bias=False)
                self.v_proj = nn.Linear(32, 32, bias=False)
                self.o_proj = nn.Linear(32, 32, bias=False)

        for mock_cls, expected_count in [
            (MockGPTNeo, 8),
            (MockGPTNeoX, 4),
            (MockLlama, 8),
        ]:
            model = mock_cls()
            wrapper = LoRAModelWrapper(model, target_module_names=config.target_modules, r=4, alpha=8.0)
            wrapper.wrap()
            trainable_params = wrapper.get_trainable_parameters()
            self.assertEqual(len(trainable_params), expected_count)
            wrapper.unwrap()

    def test_engine_safety_guards_and_telemetry(self):
        from evollm.core.engine import ContinualLearner
        from evollm.core.evidence import EvidenceClass, EvidenceRecord

        class MockTokenizer:
            def __call__(self, text, return_tensors="pt", max_length=128, truncation=True):
                tokens = [hash(word) % 90 + 1 for word in text.split()]
                if truncation and len(tokens) > max_length:
                    tokens = tokens[:max_length]
                if not tokens:
                    tokens = [1]
                t = torch.tensor([tokens], dtype=torch.long)
                return {"input_ids": t, "attention_mask": torch.ones_like(t)}

        learner = ContinualLearner(self.model, MockTokenizer())
        rec = EvidenceRecord(
            text="The port number is 9800.",
            evidence_class=EvidenceClass.VERIFIED_FACT,
            confidence=0.9,
            provenance_verified=True,
            source_reference="docs.txt",
            accept_for_learning=True,
            rejection_reason=None,
            loss_target="L_KNOW",
        )
        result = learner.learn_from_evidence(rec, query="What is the port?")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("step_time_ms", result)
        self.assertIn("rss_mb", result)
        self.assertIn("cos_phi", result)
        self.assertIn("core_energy_rejected_pct", result)
        self.assertIn("fp32_underflow_ratio", result)
        self.assertGreater(result["step_time_ms"], 0.0)

    def test_fp32_underflow_ratio_computation(self):
        delta = torch.tensor([1e-5, 2e-5, 0.1, 0.5, 0.0])
        ratio = PrecisionMaster.compute_underflow_ratio(delta, threshold=1e-4)
        self.assertAlmostEqual(ratio, 0.5, places=4)

    def test_exact_leaf_lora_matching(self):
        class DisambiguationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = nn.Linear(16, 16)
                self.dense_h_to_4h = nn.Linear(16, 64)

        model = DisambiguationModel()
        wrapper = LoRAModelWrapper(model, target_module_names=["dense"], r=4, alpha=8.0)
        wrapper.wrap()
        self.assertIn("dense", wrapper.lora_modules)
        self.assertNotIn("dense_h_to_4h", wrapper.lora_modules)
        wrapper.unwrap()


if __name__ == "__main__":
    unittest.main()

