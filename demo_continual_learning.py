"""
End-to-End Demonstration of Conversation-Driven Weight-Only Continual Learning (CPL-v2).

Demonstrates:
1. Sequential learning across conversations.
2. In-situ parameter updates into FP32 master weights.
3. Rapid O(d·K·r) SCGP stability filtering.
4. Retention of baseline parameters with bounded drift.
"""
import torch
import torch.nn as nn
from evollm.core.config import EvoLLMConfig
from evollm.core.engine import ContinualLearner


class MockTokenizer:
    """Deterministic character/token encoder for end-to-end testing."""
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size

    def __call__(self, text: str, return_tensors: str = "pt"):
        # Simple ASCII / hash encoding
        tokens = [min(ord(c), self.vocab_size - 1) for c in text]
        if not tokens:
            tokens = [0]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DemonstrationTransformer(nn.Module):
    """Clean lightweight Causal LM with standard attention projection names."""
    def __init__(self, vocab_size=256, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        q = self.q_proj(h)
        v = self.v_proj(h)
        attn_out = self.o_proj(q + v)
        logits = self.lm_head(attn_out)

        class Output:
            def __init__(self, l):
                self.logits = l
        return Output(logits)


def run_demo():
    print("=" * 70)
    print("EvoLLM (CPL-v2): Conversation-Driven Weight-Only Continual Learning")
    print("=" * 70)

    # Initialize model & tokenizer
    model = DemonstrationTransformer(vocab_size=256, d_model=128)
    tokenizer = MockTokenizer(vocab_size=256)

    config = EvoLLMConfig(
        target_modules=["q_proj", "v_proj", "o_proj"],
        lora_rank=4,
        lora_alpha=8.0,
        steps_per_conv=2,
        base_lr_correction=3e-3,
        scgp_top_k=8,
        subspace_suppression=0.85,
        frobenius_clip_rho=0.5,
        device="cpu",
    )

    learner = ContinualLearner(model=model, tokenizer=tokenizer, config=config)
    print(f"[*] Initialized model with {len(learner.precision_master.master_weights)} parameter tensors.")
    print(f"[*] Pre-computed invariant subspace bases for target projections.")
    print("-" * 70)

    # Simulated stream of conversations
    dialogues = [
        {
            "query": "What protocol is used by Node-Delta?",
            "correct": "Node-Delta uses the QUIC transport protocol with TLS 1.3.",
            "incorrect": "Node-Delta uses unencrypted HTTP/1.1.",
            "is_correction": True,
        },
        {
            "query": "Who is the primary contact for Incident Response?",
            "correct": "Commander Vance is the primary contact for Incident Response.",
            "incorrect": "Nobody is currently designated.",
            "is_correction": True,
        },
        {
            "query": "Correction: Node-Delta protocol has been upgraded. What is it now?",
            "correct": "Node-Delta has migrated and now uses custom WireGuard tunneling.",
            "incorrect": "Node-Delta still uses standard QUIC transport.",
            "is_correction": True,
        },
        {
            "query": "Can you summarize the weather?",
            "correct": "The weather is partly cloudy with light breeze.",
            "incorrect": None,
            "is_correction": False,
        }
    ]

    for idx, d in enumerate(dialogues, 1):
        print(f"\n[Conversation {idx}] User: \"{d['query']}\"")
        res = learner.learn_from_conversation(
            query=d["query"],
            correct_answer=d["correct"],
            incorrect_answer=d["incorrect"],
            is_correction=d["is_correction"],
        )
        print(f" -> Status: {res['status']}")
        if res['status'] == "SUCCESS":
            print(f" -> Effective LR: {res['effective_lr']:.6f} (Confidence α: {res['confidence_alpha']:.2f})")
            print(f" -> Loss: {res['loss']:.4f}")
            print(f" -> Cumulative Parameter Drift ||θ_t - θ_0||: {res['drift_norm']:.6f}")

    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING STREAM SUMMARY")
    print("=" * 70)
    print(f"Total Conversations Processed: {learner.conversation_count}")
    print(f"Final Parameter Drift Norm:    {learner.drift_history[-1]:.6f}")
    print("Zero persistent replay memory. All knowledge integrated directly into neural weights.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()

