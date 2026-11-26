import argparse
import copy
import hashlib
import json
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from evollm.core.config import EvoLLMConfig
from evollm.core.engine import ContinualLearner
from evollm.core.evidence import EvidenceClass, EvidenceRecord
from evollm.bench.evaluators.mcq_eval import MCQEvaluator
from evollm.bench.evaluators.perplexity_eval import PerplexityEvaluator
from evollm.bench.evaluators.qa_eval import QAEvaluator, f1_score, normalize_answer
from evollm.data.slicer import verify_manifest


def compute_model_param_hash(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(p.detach().cpu().float().numpy().tobytes())
    return hasher.hexdigest()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU execution.")
        return "cpu"
    return requested


def configure_hardware_resources(device: str, num_threads: int = 8) -> None:
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(num_threads)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_method_config(base_config: EvoLLMConfig, method: str) -> EvoLLMConfig:
    cfg = copy.deepcopy(base_config)
    if method == "B0":
        return cfg
    if method == "B1":
        cfg.subspace_suppression = 0.0
        cfg.frobenius_clip_rho = 1e6
        cfg.max_step_drift = 1e6
        cfg.lambda_global_reg = 0.0
    elif method == "B2":
        cfg.subspace_suppression = 0.0
        cfg.frobenius_clip_rho = 1e6
        cfg.max_step_drift = 1e6
        cfg.lambda_global_reg = 1e-3
        cfg.optimizer_type = "adamw"
        cfg.base_lr_ssl = min(base_config.base_lr_ssl, 5.0e-5)
        cfg.base_lr_correction = min(base_config.base_lr_correction, 1.0e-4)
    elif method == "B3":
        cfg.subspace_suppression = base_config.subspace_suppression
        cfg.frobenius_clip_rho = base_config.frobenius_clip_rho
        cfg.scgp_top_k = base_config.scgp_top_k
    elif method == "B4":
        cfg.subspace_suppression = base_config.subspace_suppression
        cfg.frobenius_clip_rho = base_config.frobenius_clip_rho
        cfg.scgp_top_k = base_config.scgp_top_k
    return cfg


def load_model_and_tokenizer(
    model_cfg: Dict[str, Any],
    device: str,
    torch_dtype: torch.dtype = torch.float32,
    hf_token: Optional[str] = None,
) -> Tuple[nn.Module, Any]:
    hf_id = model_cfg["hf_id"]

    tok_kwargs: Dict[str, Any] = {}
    if hf_token:
        tok_kwargs["token"] = hf_token

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, local_files_only=True, **tok_kwargs)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, **tok_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if hf_token:
        model_kwargs["token"] = hf_token

    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, local_files_only=True, **model_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
    model.to(device)
    return model, tokenizer


def load_or_create_stream_events(
    data_cfg: Dict[str, Any],
    max_events: int = 1000,
    slices_dir: str = "experiments/scale_5arch_15gb/data_slices",
    manifests_dir: str = "experiments/scale_5arch_15gb/data_manifests",
) -> List[Dict[str, Any]]:
    train_corpora = [
        ("wildchat", "wildchat_train.jsonl", "wildchat_manifest.json"),
        ("wikipedia", "wikipedia_train.jsonl", "wikipedia_manifest.json"),
        ("fineweb_edu", "fineweb_edu_train.jsonl", "fineweb_manifest.json"),
        ("natural_questions", "natural_questions_train.jsonl", "natural_questions_manifest.json"),
        ("ms_marco", "ms_marco_train.jsonl", "ms_marco_manifest.json"),
        ("hotpot_qa", "hotpot_qa_train.jsonl", "hotpot_manifest.json"),
        ("scientific_papers", "scientific_papers_train.jsonl", "scientific_papers_manifest.json"),
        ("stackexchange", "stackexchange_train.jsonl", "stackexchange_manifest.json"),
        ("oasst2", "oasst2_train.jsonl", "oasst2_manifest.json"),
        ("cc_news", "cc_news_train.jsonl", "cc_news_manifest.json"),
    ]

    all_exist = all(
        os.path.exists(os.path.join(slices_dir, fname)) for _, fname, _ in train_corpora
    )
    if all_exist:
        target_files = train_corpora
    else:
        discovered = []
        if os.path.exists(slices_dir):
            for fname in sorted(os.listdir(slices_dir)):
                if fname.endswith("_train.jsonl") or (fname.endswith(".jsonl") and not ("val" in fname or "test" in fname)):
                    mname = fname.replace(".jsonl", "_manifest.json").replace("_train_manifest", "_manifest")
                    discovered.append((fname.replace(".jsonl", ""), fname, mname))
        if discovered:
            target_files = discovered
        else:
            from evollm.data.slice_corpora import build_all_real_slices
            build_all_real_slices(slices_dir=slices_dir, manifests_dir=manifests_dir)
            target_files = train_corpora

    events: List[Dict[str, Any]] = []
    num_corpora = len(target_files)
    base_per_corpus = max_events // num_corpora
    remainder = max_events % num_corpora

    for idx, (name, fname, mname) in enumerate(target_files):
        target_count = base_per_corpus + (1 if idx < remainder else 0)
        if target_count <= 0:
            continue
        fp = os.path.join(slices_dir, fname)
        mp = os.path.join(manifests_dir, mname)
        if not os.path.exists(fp) or not os.path.exists(mp):
            raise RuntimeError(f"SOURCE_UNAVAILABLE: Missing verified corpus {fp}")
        if not verify_manifest(fp, mp):
            raise RuntimeError(f"DATA_UNVERIFIED: Manifest check failed for {fp}")

        corpus_events = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ev = json.loads(line)
                    ev["corpus"] = name
                    if isinstance(ev.get("answer"), (list, tuple)):
                        ev["answer"] = " ".join(str(x) for x in ev["answer"])
                    else:
                        ev["answer"] = str(ev["answer"]) if ev.get("answer") is not None else ""
                    if isinstance(ev.get("query"), (list, tuple)):
                        ev["query"] = " ".join(str(x) for x in ev["query"])
                    else:
                        ev["query"] = str(ev["query"]) if ev.get("query") is not None else ""
                    if isinstance(ev.get("incorrect_answer"), (list, tuple)):
                        ev["incorrect_answer"] = " ".join(str(x) for x in ev["incorrect_answer"])
                    elif ev.get("incorrect_answer") is not None:
                        ev["incorrect_answer"] = str(ev["incorrect_answer"])
                    corpus_events.append(ev)
                    if len(corpus_events) >= target_count:
                        break
        events.extend(corpus_events)

    for i, ev in enumerate(events):
        ev["conversation_idx"] = i + 1

    if len(events) == 0:
        raise RuntimeError("DATA_UNVERIFIED: Zero real training events found.")

    return events[:max_events]


def load_fast_eval_probes(
    data_cfg: Dict[str, Any],
    sample_limit: int = 50,
    slices_dir: str = "experiments/scale_5arch_15gb/data_slices",
    manifests_dir: str = "experiments/scale_5arch_15gb/data_manifests",
) -> Dict[str, List[Dict[str, Any]]]:
    probes: Dict[str, List[Dict[str, Any]]] = {}
    probe_files = {
        "wikitext103": ("wikitext103_val.jsonl", "wikitext103_manifest.json"),
        "squad_v2": ("squad_v2_val.jsonl", "squad_v2_manifest.json"),
        "arc_challenge": ("arc_challenge_test.jsonl", "arc_challenge_manifest.json"),
        "truthful_qa": ("truthful_qa_val.jsonl", "truthful_qa_manifest.json"),
        "mmlu": ("mmlu_test.jsonl", "mmlu_manifest.json"),
        "gsm8k": ("gsm8k_test.jsonl", "gsm8k_manifest.json"),
    }

    for key, (f_name, m_name) in probe_files.items():
        fpath = os.path.join(slices_dir, f_name)
        mpath = os.path.join(manifests_dir, m_name)
        if os.path.exists(fpath):
            if os.path.exists(mpath) and not verify_manifest(fpath, mpath):
                raise RuntimeError(f"DATA_UNVERIFIED: Manifest check failed for probe {fpath}")
            items = []
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
                        if len(items) >= sample_limit:
                            break
            probes[key] = items

    if not probes:
        from evollm.data.slice_corpora import build_all_real_slices
        build_all_real_slices(slices_dir=slices_dir, manifests_dir=manifests_dir)
        for key, (f_name, m_name) in probe_files.items():
            fpath = os.path.join(slices_dir, f_name)
            if os.path.exists(fpath):
                items = []
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            items.append(json.loads(line))
                            if len(items) >= sample_limit:
                                break
                probes[key] = items

    return probes



def run_architecture_stream(
    model: nn.Module,
    tokenizer: Any,
    model_name: str,
    method: str,
    events: List[Dict[str, Any]],
    eval_probes: Dict[str, List[Dict[str, Any]]],
    config: EvoLLMConfig,
    eval_checkpoints: List[int],
    output_dir: str = "experiments/scale_5arch_15gb",
    fast_probe_budget_sec: float = 25.0,
    force_reset: bool = False,
    cached_base_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_dir = os.path.join(output_dir, "runs", model_name, method)
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    metrics_jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    checkpoint_state_path = os.path.join(run_dir, "checkpoint_state.json")

    effective_checkpoints = set(eval_checkpoints) | {0, len(events)}

    start_event_idx = 0
    run_records: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []

    if force_reset:
        if os.path.exists(checkpoint_state_path):
            os.remove(checkpoint_state_path)
        if os.path.exists(metrics_jsonl_path):
            os.remove(metrics_jsonl_path)
    elif os.path.exists(checkpoint_state_path):
        try:
            with open(checkpoint_state_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
                start_event_idx = ckpt.get("last_event_idx", 0)
                eval_records = ckpt.get("eval_records", [])
        except Exception:
            start_event_idx = 0

        if start_event_idx >= len(events) and os.path.exists(metrics_jsonl_path):
            print(f"Run {model_name} {method} already complete ({start_event_idx} >= {len(events)} events). Skipping to next run.")
            with open(metrics_jsonl_path, "r", encoding="utf-8") as f:
                run_records = [json.loads(line) for line in f if line.strip()]
            return {
                "model": model_name,
                "method": method,
                "total_events": len(events),
                "run_records_count": len(run_records),
                "eval_records": eval_records,
                "metrics_jsonl": metrics_jsonl_path,
            }
        else:
            # Incomplete run cannot resume weight state safely without serialized weights; restart from event 0
            start_event_idx = 0
            eval_records = []
            if os.path.exists(checkpoint_state_path):
                os.remove(checkpoint_state_path)
            if os.path.exists(metrics_jsonl_path):
                os.remove(metrics_jsonl_path)

    method_config = build_method_config(config, method)
    learner = ContinualLearner(model, tokenizer, config=method_config) if method != "B0" else None

    mcq_evaluator = MCQEvaluator(model, tokenizer, device=config.device)
    qa_evaluator = QAEvaluator(model, tokenizer, device=config.device)
    ppl_evaluator = PerplexityEvaluator(model, tokenizer, device=config.device)

    def evaluate_probes(step_idx: int) -> Dict[str, Any]:
        t0 = time.perf_counter()
        print(f"[{model_name} {method}] Evaluating probes at event {step_idx}...", flush=True)
        snapshot: Dict[str, Any] = {"event_idx": step_idx, "method": method, "model": model_name}

        if "wikitext103" in eval_probes:
            ppl_res = ppl_evaluator.evaluate(eval_probes["wikitext103"])
            snapshot["ppl"] = round(ppl_res.get("perplexity", 1.0), 2)
        if "arc_challenge" in eval_probes:
            arc_res = mcq_evaluator.evaluate(eval_probes["arc_challenge"])
            snapshot["arc_acc"] = round(arc_res.get("accuracy", 0.0), 4)
        if "truthful_qa" in eval_probes:
            tqa_res = mcq_evaluator.evaluate_truthfulqa(eval_probes["truthful_qa"])
            snapshot["truthful_acc"] = round(tqa_res.get("mc1_acc", tqa_res.get("accuracy", 0.0)), 4)
        print(f"  [{model_name} {method}] PPL/ARC/TQA done in {time.perf_counter()-t0:.1f}s", flush=True)
        if "squad_v2" in eval_probes:
            qa_res = qa_evaluator.evaluate(eval_probes["squad_v2"])
            snapshot["qa_f1"] = round(qa_res.get("token_f1", qa_res.get("f1", 0.0)), 4)
        elif "qa" in eval_probes:
            qa_res = qa_evaluator.evaluate(eval_probes["qa"])
            snapshot["qa_f1"] = round(qa_res.get("token_f1", qa_res.get("f1", 0.0)), 4)
        if "mmlu" in eval_probes:
            mmlu_res = mcq_evaluator.evaluate(eval_probes["mmlu"])
            snapshot["mmlu_acc"] = round(mmlu_res.get("accuracy", 0.0), 4)
        if "gsm8k" in eval_probes:
            gsm_res = qa_evaluator.evaluate(eval_probes["gsm8k"])
            snapshot["gsm8k_acc"] = round(gsm_res.get("exact_match", gsm_res.get("accuracy", 0.0)), 4)
        print(f"  [{model_name} {method}] QA/MMLU/GSM8K done in {time.perf_counter()-t0:.1f}s", flush=True)

        # Knowledge Adaptation Accuracy (KAA), Distractor Mimicry Rate (DMR), and Gate FAR
        verified_pool = [e for e in events if e.get("corpus") != "wildchat" and e.get("source_reference") is not None and e.get("evidence_class") in ["VERIFIED_FACT", "EXPLICIT_CORRECTION", "KNOW", "CORR"]]
        distractor_pool = [e for e in events if e.get("corpus") == "wildchat" or e.get("source_reference") is None or e.get("evidence_class") in ["HYPOTHETICAL", "UNVERIFIED"]]

        past_events = events[:step_idx] if step_idx > 0 else events
        verified_past = [e for e in past_events if e.get("corpus") != "wildchat" and e.get("source_reference") is not None and e.get("evidence_class") in ["VERIFIED_FACT", "EXPLICIT_CORRECTION", "KNOW", "CORR"]]
        distractor_past = [e for e in past_events if e.get("corpus") == "wildchat" or e.get("source_reference") is None or e.get("evidence_class") in ["HYPOTHETICAL", "UNVERIFIED"]]

        eval_v = verified_past if step_idx > 0 else verified_pool
        sample_k = min(25, len(eval_v))
        if sample_k > 0:
            k_correct = 0
            sampled_v = [eval_v[(i * len(eval_v)) // sample_k] for i in range(sample_k)]
            for v_ev in sampled_v:
                q = v_ev["query"]
                ans = v_ev["answer"]
                prompt = f"User: {q}\nAssistant:"
                pred = qa_evaluator.generate_completion(prompt, max_new_tokens=32)
                if normalize_answer(ans) in normalize_answer(pred) or f1_score(pred, ans) >= 0.25:
                    k_correct += 1
            snapshot["kaa"] = round((k_correct / sample_k) * 100.0, 1)
        else:
            snapshot["kaa"] = 0.0

        eval_d = distractor_past if step_idx > 0 else distractor_pool
        sample_d = min(15, len(eval_d))
        if sample_d > 0:
            d_matched = 0
            sampled_d = [eval_d[(i * len(eval_d)) // sample_d] for i in range(sample_d)]
            for d_ev in sampled_d:
                q = d_ev["query"]
                ans = d_ev["answer"]
                prompt = f"User: {q}\nAssistant:"
                pred = qa_evaluator.generate_completion(prompt, max_new_tokens=32)
                if normalize_answer(ans) in normalize_answer(pred) or f1_score(pred, ans) >= 0.25:
                    d_matched += 1
            snapshot["dmr"] = round((d_matched / sample_d) * 100.0, 1)
        else:
            snapshot["dmr"] = 0.0

        # Algorithmic Stream Gate False Acceptance Rate (FAR %)
        if step_idx == 0 or method == "B0":
            snapshot["far"] = 0.0
        elif method == "B4":
            unverified_presented = 0
            unverified_accepted = 0
            for r_idx in range(min(step_idx, len(run_records))):
                ev_in = events[r_idx]
                if ev_in.get("corpus") == "wildchat" or ev_in.get("source_reference") is None:
                    unverified_presented += 1
                    rec_out = run_records[r_idx]
                    if rec_out.get("status") == "SUCCESS":
                        unverified_accepted += 1
            gate_far = (unverified_accepted / max(1, unverified_presented)) * 100.0 if unverified_presented > 0 else 0.0
            snapshot["far"] = round(gate_far, 2)
        else:
            # Ungated baselines (B1, B2, B3) absorb all unverified turns without qualification
            snapshot["far"] = 100.0

        elapsed = time.perf_counter() - t0
        snapshot["eval_elapsed_sec"] = round(elapsed, 3)
        print(f"[{model_name} {method}] Evaluation at event {step_idx} done in {elapsed:.1f}s: PPL={snapshot.get('ppl')}, ARC={snapshot.get('arc_acc')}, MMLU={snapshot.get('mmlu_acc')}", flush=True)
        return snapshot

    if start_event_idx == 0:
        if cached_base_eval is not None:
            base_eval = copy.deepcopy(cached_base_eval)
            base_eval["method"] = method
            base_eval["model"] = model_name
        else:
            base_eval = evaluate_probes(0)
        eval_records.append(base_eval)

    replay_buffer: List[Dict[str, Any]] = []

    for idx in range(start_event_idx, len(events)):
        ev = events[idx]
        event_num = idx + 1

        if method == "B0":
            rec = {
                "conversation_idx": event_num,
                "status": "B0_PASSIVE",
                "loss": 0.0,
                "drift_norm": 0.0,
                "step_time_ms": 0.1,
                "rss_mb": learner._get_rss_mb() if learner else 0.0,
                "cos_phi": 0.0,
                "core_energy_rejected_pct": 0.0,
                "fp32_underflow_ratio": 0.0,
            }
        elif method == "B4":
            is_unverified_chat = ev.get("corpus") == "wildchat"
            src_ref = None if is_unverified_chat else ev.get("source_reference")
            rec_obj = learner.evidence_qualifier.qualify(
                ev["answer"],
                source_reference=src_ref,
                allow_ssl=(src_ref is not None),
            )
            if rec_obj.accept_for_learning and "loss_target" in ev:
                rec_obj.loss_target = ev["loss_target"]
            step_result = learner.learn_from_evidence(
                rec_obj,
                query=ev["query"],
                incorrect_answer=ev.get("incorrect_answer"),
            )
            rec = step_result
        elif method == "B3":
            rec_obj = EvidenceRecord(
                text=ev["answer"],
                evidence_class=EvidenceClass.ORDINARY_USER_TEXT,
                confidence=ev.get("confidence", 0.7),
                provenance_verified=False,
                source_reference=ev.get("source_reference"),
                accept_for_learning=True,
                rejection_reason=None,
                loss_target="L_SSL",
            )
            step_result = learner.learn_from_evidence(
                rec_obj,
                query=ev["query"],
                incorrect_answer=ev.get("incorrect_answer"),
            )
            rec = step_result
        else:
            e_class_val = ev.get("evidence_class", "VERIFIED_FACT")
            try:
                e_class = EvidenceClass(e_class_val)
            except Exception:
                e_class = EvidenceClass.VERIFIED_FACT
            rec_obj = EvidenceRecord(
                text=ev["answer"],
                evidence_class=e_class,
                confidence=ev.get("confidence", 0.8),
                provenance_verified=ev.get("source_reference") is not None,
                source_reference=ev.get("source_reference"),
                accept_for_learning=True,
                rejection_reason=None,
                loss_target=ev.get("loss_target", "L_KNOW"),
            )
            step_result = learner.learn_from_evidence(
                rec_obj,
                query=ev["query"],
                incorrect_answer=ev.get("incorrect_answer"),
            )
            rec = step_result

        run_records.append(rec)

        with open(metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        if event_num % 10 == 0 or event_num == len(events):
            loss_val = rec.get("loss", 0.0) or 0.0
            drift_val = rec.get("drift_norm", 0.0) or 0.0
            time_val = rec.get("step_time_ms", 0.0) or 0.0
            print(f"[{model_name} {method}] Event {event_num}/{len(events)} - Loss: {loss_val:.4f}, Drift: {drift_val:.6f}, Step: {time_val:.1f}ms", flush=True)

        if event_num in effective_checkpoints or event_num == len(events):
            if method == "B0" and eval_records:
                eval_snap = copy.deepcopy(eval_records[0])
                eval_snap["event_idx"] = event_num
                eval_snap["kaa"] = 0.0
                eval_snap["far"] = 0.0
            else:
                eval_snap = evaluate_probes(event_num)
            eval_records.append(eval_snap)

            with open(checkpoint_state_path, "w", encoding="utf-8") as f:
                json.dump({
                    "last_event_idx": event_num,
                    "eval_records": eval_records,
                }, f, indent=2)

    if learner is not None:
        del learner
    import gc
    gc.collect()

    return {
        "model": model_name,
        "method": method,
        "total_events": len(events),
        "run_records_count": len(run_records),
        "eval_records": eval_records,
        "metrics_jsonl": metrics_jsonl_path,
    }


def main():
    parser = argparse.ArgumentParser(description="5-Architecture 15GB Continuous Learning Stream Runner")
    parser.add_argument("--model", type=str, default="pythia-160m", choices=["pythia-160m", "opt-350m", "smollm2-360m", "qwen2.5-0.5b", "granite-moe-1b", "all"])
    parser.add_argument("--method", type=str, default="B4", choices=["B0", "B1", "B2", "B3", "B4", "all"])
    parser.add_argument("--max_events", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--config_path", type=str, default="experiments/scale_5arch_15gb/config.yaml")
    parser.add_argument("--models_path", type=str, default="experiments/scale_5arch_15gb/models.yaml")
    parser.add_argument("--datasets_path", type=str, default="experiments/scale_5arch_15gb/datasets.yaml")
    parser.add_argument("--force", action="store_true", help="Force re-run from event 0 even if checkpoints exist")
    args = parser.parse_args()

    cfg = load_yaml(args.config_path)
    models_cfg = load_yaml(args.models_path)["models"]
    datasets_cfg = load_yaml(args.datasets_path)

    device = resolve_device(args.device if args.device != "auto" else cfg["hardware"]["device"])
    configure_hardware_resources(device, cfg["hardware"]["num_cpu_threads"])

    model_names = list(models_cfg.keys()) if args.model == "all" else [args.model]
    methods = ["B0", "B1", "B2", "B3", "B4"] if args.method == "all" else [args.method]

    events = load_or_create_stream_events(datasets_cfg, max_events=args.max_events)
    eval_probes = load_fast_eval_probes(datasets_cfg, sample_limit=cfg["evaluation"]["eval_probe_sample_limit"])

    base_evollm_cfg = EvoLLMConfig(
        device=device,
        num_cpu_threads=cfg["hardware"]["num_cpu_threads"],
        steps_per_conv=cfg["training"]["steps_per_conv"],
        base_lr_correction=cfg["training"]["base_lr_correction"],
        base_lr_ssl=cfg["training"]["base_lr_ssl"],
        gradient_clip_norm=cfg["training"]["gradient_clip_norm"],
        max_seq_len=cfg["training"]["max_seq_len"],
        max_step_drift=cfg["training"]["max_step_drift"],
        scgp_top_k=cfg["training"]["scgp_top_k"],
        subspace_suppression=cfg["training"]["subspace_suppression"],
        frobenius_clip_rho=cfg["training"]["frobenius_clip_rho"],
        lambda_step_reg=cfg["training"]["lambda_step_reg"],
        lambda_global_reg=cfg["training"]["lambda_global_reg"],
        simpo_margin=cfg["training"]["simpo_margin"],
        preference_weight_mu=cfg["training"]["preference_weight_mu"],
        min_confidence_alpha=cfg["training"]["min_confidence_alpha"],
        max_confidence_alpha=cfg["training"]["max_confidence_alpha"],
        master_dtype=cfg["training"]["master_dtype"],
        compute_dtype=cfg["training"]["compute_dtype"],
    )

    all_results: Dict[str, Any] = {}

    for m_name in model_names:
        m_info = models_cfg[m_name]
        base_evollm_cfg.target_modules = m_info["target_modules"]
        base_evollm_cfg.lora_rank = m_info.get("lora_rank", 8)
        base_evollm_cfg.lora_alpha = m_info.get("lora_alpha", 16.0)

        initial_param_hash: Optional[str] = None
        cached_base_eval: Optional[Dict[str, Any]] = None

        for met in methods:
            m_dtype_name = m_info.get("compute_dtype", base_evollm_cfg.compute_dtype)
            compute_dtype = torch.bfloat16 if m_dtype_name == "bfloat16" else torch.float32
            print(f"=== Starting execution: Model={m_name}, Method={met}, Device={device}, ComputeDtype={compute_dtype} ===")
            model, tokenizer = load_model_and_tokenizer(m_info, device, torch_dtype=compute_dtype)

            param_hash = compute_model_param_hash(model)
            if initial_param_hash is None:
                initial_param_hash = param_hash
            else:
                assert param_hash == initial_param_hash, (
                    f"Parameter state contamination detected across methods for {m_name}! "
                    f"Expected sha256={initial_param_hash}, got {param_hash}."
                )

            res = run_architecture_stream(
                model=model,
                tokenizer=tokenizer,
                model_name=m_name,
                method=met,
                events=events,
                eval_probes=eval_probes,
                config=base_evollm_cfg,
                eval_checkpoints=cfg["evaluation"]["eval_checkpoints"],
                output_dir="experiments/scale_5arch_15gb",
                fast_probe_budget_sec=cfg["evaluation"]["fast_probe_budget_sec"],
                force_reset=args.force,
                cached_base_eval=cached_base_eval,
            )
            if cached_base_eval is None and res.get("eval_records"):
                cached_base_eval = copy.deepcopy(res["eval_records"][0])
            all_results[f"{m_name}_{met}"] = res

            summary_path = os.path.join(cfg["paths"]["reports_dir"], "eval_summary.json")
            existing_summary = {}
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        existing_summary = json.load(f)
                except Exception:
                    existing_summary = {}
            existing_summary[f"{m_name}_{met}"] = res
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(existing_summary, f, indent=2)

            del model
            del tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()
            import gc
            gc.collect()

    summary_path = os.path.join(cfg["paths"]["reports_dir"], "eval_summary.json")
    existing_summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing_summary = json.load(f)
        except Exception:
            existing_summary = {}
    existing_summary.update(all_results)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(existing_summary, f, indent=2)
    print(f"Stream execution complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
