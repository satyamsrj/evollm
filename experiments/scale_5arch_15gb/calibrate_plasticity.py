import os
import sys
sys.path.insert(0, ".")
import json
import torch
from evollm.core.config import EvoLLMConfig
from evollm.core.engine import ContinualLearner
from evollm.bench.evaluators.perplexity_eval import PerplexityEvaluator
from evollm.core.evidence import EvidenceRecord, EvidenceClass
from experiments.scale_5arch_15gb.run_stream import load_model_and_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Testing Pythia-160M B1 plasticity calibration on device:", device)

model, tokenizer = load_model_and_tokenizer({"hf_id": "EleutherAI/pythia-160m"}, device)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load 15 wikitext items
with open("experiments/scale_5arch_15gb/data_slices/wikitext103_val.jsonl", "r", encoding="utf-8") as f:
    wiki_items = [json.loads(l) for l in f if l.strip()][:15]

ppl_eval = PerplexityEvaluator(model, tokenizer, device=device)
base_ppl = ppl_eval.evaluate(wiki_items)["perplexity"]
print(f"Initial Frozen PPL: {base_ppl:.2f}")

# Load 20 train events
with open("experiments/scale_5arch_15gb/data_slices/wildchat_train.jsonl", "r", encoding="utf-8") as f:
    train_events = [json.loads(l) for l in f if l.strip()][:20]

for lr_val in [2.0e-3, 5.0e-3, 1.0e-2]:
    print(f"\n--- Testing B1 with base_lr_ssl={lr_val:.1e}, steps=4 ---")
    cfg = EvoLLMConfig(
        device=device,
        target_modules=["query_key_value", "dense"],
        lora_rank=8,
        lora_alpha=16.0,
        steps_per_conv=4,
        base_lr_correction=lr_val * 2,
        base_lr_ssl=lr_val,
        subspace_suppression=0.0,
        frobenius_clip_rho=1e6,
        lambda_global_reg=0.0,
    )
    learner = ContinualLearner(model, tokenizer, config=cfg)
    
    for idx, ev in enumerate(train_events):
        rec = EvidenceRecord(
            text=ev.get("answer", ""),
            evidence_class=EvidenceClass.ORDINARY_USER_TEXT,
            confidence=0.9,
            provenance_verified=False,
            source_reference="wildchat",
            accept_for_learning=True,
            rejection_reason=None,
            loss_target="L_SSL",
        )
        res = learner.learn_from_evidence(rec, query=ev.get("query", ""))
        if (idx + 1) % 10 == 0:
            l_val = res.get("loss", 0.0)
            d_val = res.get("drift_norm", 0.0)
            print(f"  Event {idx+1}: loss={l_val:.4f}, drift={d_val:.5f}")

    post_ppl = ppl_eval.evaluate(wiki_items)["perplexity"]
    diff = post_ppl - base_ppl
    print(f"  Result: Post PPL: {post_ppl:.2f} (Delta PPL: {diff:+.2f})")
    
    # Reset model back to frozen weights for next test
    with torch.no_grad():
        for k, v in learner.initial_weights.items():
            learner.precision_master.master_weights[k] = v.clone()
        learner.precision_master.sync_to_model(model)
