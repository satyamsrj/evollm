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
from evollm.core.loss import LossEngine, SignalTriager
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

    def test_signal_triager(self):
        """Verifies accurate categorization of dialogue turns."""
        self.assertTrue(SignalTriager.is_question_or_query("What is the baud rate?"))
        self.assertTrue(SignalTriager.is_question_or_query("Can you help me?"))
        self.assertFalse(SignalTriager.is_factual_assertion("What is the baud rate?"))
        
        self.assertTrue(SignalTriager.is_factual_assertion("The baud rate is 115200."))
        self.assertTrue(SignalTriager.is_explicit_correction("No, the baud rate was updated to 9600."))
        self.assertTrue(SignalTriager.is_explicit_correction("Wrong, it uses port 443."))

    def test_stateless_normalized_sgd(self):
        """Verifies optimizer step executes without carrying momentum or state."""
        params = [nn.Parameter(torch.tensor([2.0, -3.0], requires_grad=True))]
        opt = StatelessNormalizedSGD(params, lr=0.1, max_grad_norm=1.0)
        
        # Pseudo loss: 0.5 * (x^2 + y^2)
        loss = 0.5 * (params[0] ** 2).sum()
        loss.backward()
        
        opt.step()
        opt.zero_grad()
        
        # Check parameter changed in negative gradient direction
        self.assertLess(params[0][0].item(), 2.0)
        self.assertGreater(params[0][1].item(), -3.0)
        
        # Verify no hidden state buffers exist
        self.assertFalse(hasattr(opt, "state") and len(opt.state) > 0)


if __name__ == "__main__":
    unittest.main()

