import os
import sys
sys.path.insert(0, ".")
import json
import torch
from evollm.core.config import EvoLLMConfig
from evollm.core.engine import ContinualLearner
from evollm.bench.evaluators.perplexity_eval import PerplexityEvaluator
from evollm.bench.evaluators.mcq_eval import MCQEvaluator
from evollm.bench.evaluators.qa_eval import QAEvaluator
from evollm.core.evidence import EvidenceRecord, EvidenceClass
from experiments.scale_5arch_15gb.run_stream import load_model_and_tokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Pythia-160M Frozen Sanity Check & B4 Telemetry Dump (Device: {device}) ===")

    print("Loading Pythia-160M...")
    model, tokenizer = load_model_and_tokenizer({"hf_id": "EleutherAI/pythia-160m"}, device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. WikiText-103 PPL
    print("\n--- 1. Evaluating Frozen WikiText-103 Perplexity ---")
    with open("experiments/scale_5arch_15gb/data_slices/wikitext103_val.jsonl", "r", encoding="utf-8") as f:
        wiki_items = [json.loads(l) for l in f if l.strip()][:20]
    ppl_eval = PerplexityEvaluator(model, tokenizer, device=device)
    wiki_res = ppl_eval.evaluate(wiki_items)
    print(f"Frozen WikiText-103 PPL: {wiki_res['perplexity']:.2f} (Cross-Entropy: {wiki_res.get('cross_entropy', 0.0):.4f})")

    # 2. TruthfulQA MC1
    print("\n--- 2. Evaluating Frozen TruthfulQA MC1 ---")
    with open("experiments/scale_5arch_15gb/data_slices/truthful_qa_val.jsonl", "r", encoding="utf-8") as f:
        truthful_items = [json.loads(l) for l in f if l.strip()][:25]
    mcq_eval = MCQEvaluator(model, tokenizer, device=device)
    truthful_res = mcq_eval.evaluate_truthfulqa(truthful_items)
    tqa_acc = truthful_res.get("mc1_acc", 0.0)
    print(f"Frozen TruthfulQA MC1 Accuracy: {tqa_acc*100:.2f}% (Total: {truthful_res.get('total', 0)})")

    # 3. ARC-Challenge
    print("\n--- 3. Evaluating Frozen ARC-Challenge ---")
    with open("experiments/scale_5arch_15gb/data_slices/arc_challenge_test.jsonl", "r", encoding="utf-8") as f:
        arc_items = [json.loads(l) for l in f if l.strip()][:25]
    arc_res = mcq_eval.evaluate(arc_items)
    arc_acc = arc_res.get("accuracy", 0.0)
    print(f"Frozen ARC-Challenge Accuracy: {arc_acc*100:.2f}% (Total: {arc_res.get('total', 0)})")

    # 4. GSM8K EM
    print("\n--- 4. Evaluating Frozen GSM8K Exact Match ---")
    with open("experiments/scale_5arch_15gb/data_slices/gsm8k_test.jsonl", "r", encoding="utf-8") as f:
        gsm_items = [json.loads(l) for l in f if l.strip()][:20]
    qa_eval = QAEvaluator(model, tokenizer, device=device)
    gsm_res = qa_eval.evaluate(gsm_items)
    gsm_em = gsm_res.get("exact_match", 0.0)
    print(f"Frozen GSM8K Exact Match: {gsm_em*100:.2f}% (Total: {gsm_res.get('total', 0)})")

    # 5. B4 SCGP Telemetry Dump (1-step)
    print("\n--- 5. Evaluating 1-Step B4 SCGP Telemetry ---")
    cfg = EvoLLMConfig(
        device=device,
        target_modules=["query_key_value", "dense"],
        lora_rank=8,
        lora_alpha=16.0,
        steps_per_conv=4,
        base_lr_correction=1.0e-3,
        base_lr_ssl=3.0e-4,
        subspace_suppression=0.5,
        scgp_top_k=32,
    )
    learner = ContinualLearner(model, tokenizer, config=cfg)
    
    with open("experiments/scale_5arch_15gb/data_slices/wildchat_train.jsonl", "r", encoding="utf-8") as f:
        train_event = json.loads(f.readline().strip())
    
    rec = EvidenceRecord(
        text=train_event.get("answer", "Paris is the capital of France."),
        evidence_class=EvidenceClass.ORDINARY_USER_TEXT,
        confidence=0.9,
        provenance_verified=True,
        source_reference="wildchat",
        accept_for_learning=True,
        rejection_reason=None,
        loss_target="L_SSL",
    )
    res = learner.learn_from_evidence(rec, query=train_event.get("query", "What is the capital of France?"))
    
    cos_phi = res.get("cos_phi", 0.0)
    core_energy_rejected_pct = res.get("core_energy_rejected_pct", 0.0)
    underflow_ratio = res.get("fp32_underflow_ratio", 0.0)
    drift_norm = res.get("drift_norm", 0.0)
    
    print("\n=== B4 SCGP Telemetry Dump ===")
    print(f"  cos_phi:                  {cos_phi:.5f}")
    print(f"  core_energy_rejected_pct: {core_energy_rejected_pct:.4f}%")
    print(f"  fp32_underflow_ratio:     {underflow_ratio:.4f}")
    print(f"  drift_norm:               {drift_norm:.5f}")
    print("==============================\n")
    
    # Assertions for scientific sanity
    assert 0.0 <= tqa_acc <= 0.60, f"TruthfulQA impossible: {tqa_acc}"
    assert 0.0 <= gsm_em <= 0.20, f"GSM8K impossible for 160M model: {gsm_em}"
    assert 0.0 < cos_phi <= 1.0, f"cos_phi out of range: {cos_phi}"
    print("ALL SANITY CHECKS AND TELEMETRY VERIFICATIONS PASSED!")

if __name__ == "__main__":
    main()
