import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from evollm.data.slicer import DatasetSlicer, verify_manifest


def format_wildchat(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conv = row.get("conversation", [])
    if not isinstance(conv, list) or len(conv) < 2:
        return None
    user_turn = conv[0].get("content", "").strip()
    asst_turn = conv[1].get("content", "").strip()
    if not user_turn or not asst_turn:
        return None
    user_words = user_turn.split()[:48]
    asst_words = asst_turn.split()[:64]
    if len(user_words) < 3 or len(asst_words) < 3:
        return None
    return {
        "query": " ".join(user_words),
        "answer": " ".join(asst_words),
        "incorrect_answer": None,
        "loss_target": "L_SSL",
        "evidence_class": "ORDINARY_USER_TEXT",
        "confidence": 0.70,
        "source_reference": row.get("conversation_hash", "wildchat_corpus"),
    }


def format_fineweb_edu(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = row.get("text", "").strip()
    if not text or len(text.split()) < 20:
        return None
    words = text.split()[:80]
    passage = " ".join(words)
    return {
        "query": "Explain this educational concept.",
        "answer": passage,
        "incorrect_answer": None,
        "loss_target": "L_SSL",
        "evidence_class": "ORDINARY_USER_TEXT",
        "confidence": 0.85,
        "source_reference": row.get("url", "fineweb-edu-10bt"),
    }


def format_hotpot_qa(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = row.get("question", "").strip()
    a = row.get("answer", "").strip()
    if not q or not a:
        return None
    return {
        "query": q,
        "answer": a,
        "incorrect_answer": None,
        "loss_target": "L_KNOW",
        "evidence_class": "VERIFIED_FACT",
        "confidence": 0.95,
        "source_reference": str(row.get("id", "hotpot_qa")),
    }


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def format_stackexchange(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Supports both prompt/chosen/rejected and question/answers formats
    q = strip_html(str(row.get("question") or row.get("prompt") or ""))
    if not q:
        return None
    
    chosen = ""
    rejected = ""
    answers = row.get("answers", [])
    if isinstance(answers, list) and len(answers) >= 2:
        chosen_cand = max(answers, key=lambda a: (1 if a.get("selected") else 0, a.get("pm_score") or 0))
        rejected_cand = min(answers, key=lambda a: (1 if a.get("selected") else 0, a.get("pm_score") or 0))
        chosen = strip_html(chosen_cand.get("text", ""))
        rejected = strip_html(rejected_cand.get("text", ""))
    else:
        chosen = strip_html(str(row.get("chosen", "")))
        rejected = strip_html(str(row.get("rejected", "")))

    if not chosen or not rejected or chosen == rejected:
        return None
    return {
        "query": " ".join(q.split()[:40]),
        "answer": " ".join(chosen.split()[:60]),
        "incorrect_answer": " ".join(rejected.split()[:60]),
        "loss_target": "L_CORR",
        "evidence_class": "EXPLICIT_CORRECTION",
        "confidence": 0.92,
        "source_reference": str(row.get("qid", "stackexchange_pref")),
    }


def format_oasst2(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = row.get("text", "").strip()
    role = row.get("role", "")
    if not text or role != "assistant":
        return None
    return {
        "query": "Provide a helpful response following instructions.",
        "answer": " ".join(text.split()[:70]),
        "incorrect_answer": None,
        "loss_target": "L_CORR",
        "evidence_class": "EXPLICIT_CORRECTION",
        "confidence": 0.88,
        "source_reference": str(row.get("message_id", "oasst2")),
    }


def format_squad_qa(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    context = row.get("context", "").strip()
    q = row.get("question", "").strip()
    answers = row.get("answers", {}).get("text", [])
    if not q or not answers:
        return None
    return {
        "prompt": f"Context: {' '.join(context.split()[:60])}\nQuestion: {q}\nAnswer:",
        "question": q,
        "answer": answers[0],
        "candidates": [answers[0]],
    }


def format_arc(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = row.get("question", "").strip()
    choices = row.get("choices", {})
    text_choices = choices.get("text", [])
    labels = choices.get("label", [])
    ans_key = row.get("answerKey", "")
    if not q or not text_choices or ans_key not in labels:
        return None
    ans_idx = labels.index(ans_key)
    return {
        "prompt": f"Question: {q}\nAnswer:",
        "question": q,
        "choices": text_choices,
        "candidates": [f" {c}" for c in text_choices],
        "label": ans_key,
        "answer_idx": ans_idx,
    }


def format_mmlu(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = row.get("question", "").strip()
    choices = row.get("choices", [])
    ans = row.get("answer", -1)
    if not q or not choices or ans < 0 or ans >= len(choices):
        return None
    return {
        "prompt": f"Question: {q}\nAnswer:",
        "question": q,
        "choices": choices,
        "candidates": [f" {c}" for c in choices],
        "answer_idx": int(ans),
    }


def format_gsm8k(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = row.get("question", "").strip()
    a = row.get("answer", "").strip()
    if not q or not a:
        return None
    ans_val = a.split("####")[-1].strip() if "####" in a else a
    return {
        "prompt": f"Math Problem: {q}\nAnswer:",
        "question": q,
        "answer": ans_val,
        "candidates": [f" {ans_val}"],
    }


def format_wikitext(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = row.get("text", "").strip()
    if not text or len(text.split()) < 10:
        return None
    return {"text": " ".join(text.split()[:64])}


def format_truthful_qa(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = row.get("question", "").strip()
    mc1 = row.get("mc1_targets", {})
    mc2 = row.get("mc2_targets", {})
    if not q:
        return None
    return {
        "question": q,
        "mc1_targets": mc1,
        "mc2_targets": mc2,
    }


def format_wikipedia(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    title = row.get("title", "").strip()
    text = row.get("text", "").strip()
    if not title or not text:
        return None
    words = text.split()[:70]
    if len(words) < 10:
        return None
    return {
        "query": f"Provide factual encyclopedic information about {title}.",
        "answer": " ".join(words),
        "incorrect_answer": None,
        "loss_target": "L_KNOW",
        "evidence_class": "VERIFIED_FACT",
        "confidence": 0.98,
        "source_reference": str(row.get("url", f"https://en.wikipedia.org/wiki/{title}")),
    }


def format_natural_questions(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q_dict = row.get("question", {})
    q = q_dict.get("text", "") if isinstance(q_dict, dict) else str(row.get("question", ""))
    annotations = row.get("annotations", {})
    short_answers = annotations.get("short_answers", []) if isinstance(annotations, dict) else []
    a = ""
    if short_answers and len(short_answers) > 0 and "text" in short_answers[0]:
        val = short_answers[0]["text"]
        a = " ".join(val) if isinstance(val, list) else str(val)
    if not a:
        a = str(row.get("document", {}).get("title", ""))
    if isinstance(a, list):
        a = " ".join(str(x) for x in a)
    a = str(a).strip()
    if not q or not a:
        return None
    return {
        "query": str(q).strip(),
        "answer": a,
        "incorrect_answer": None,
        "loss_target": "L_KNOW",
        "evidence_class": "VERIFIED_FACT",
        "confidence": 0.95,
        "source_reference": str(row.get("document", {}).get("url", "natural_questions")),
    }


def format_ms_marco(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = row.get("query", "").strip()
    answers = row.get("answers", [])
    a = answers[0] if answers and len(answers) > 0 and answers[0] != "No Answer Present." else ""
    if not a:
        passages = row.get("passages", {}).get("passage_text", [])
        if passages:
            a = " ".join(passages[0].split()[:50])
    if not q or not a:
        return None
    return {
        "query": q,
        "answer": a,
        "incorrect_answer": None,
        "loss_target": "L_KNOW",
        "evidence_class": "VERIFIED_FACT",
        "confidence": 0.92,
        "source_reference": str(row.get("query_id", "ms_marco")),
    }


def format_scientific_papers(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    abstract = row.get("abstract", "").strip()
    if not abstract:
        return None
    words = abstract.split()[:70]
    if len(words) < 10:
        return None
    return {
        "query": "Summarize the peer-reviewed scientific methodology and findings.",
        "answer": " ".join(words),
        "incorrect_answer": None,
        "loss_target": "L_KNOW",
        "evidence_class": "VERIFIED_FACT",
        "confidence": 0.96,
        "source_reference": str(row.get("article_id", "scientific_papers")),
    }


def format_cc_news(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    title = row.get("title", "").strip()
    text = row.get("text", "").strip()
    if not title or not text:
        return None
    words = text.split()[:70]
    if len(words) < 10:
        return None
    return {
        "query": f"Report temporal news: {title}",
        "answer": " ".join(words),
        "incorrect_answer": None,
        "loss_target": "L_KNOW",
        "evidence_class": "VERIFIED_FACT",
        "confidence": 0.90,
        "source_reference": str(row.get("url", "cc_news")),
    }


def build_all_real_slices(
    slices_dir: str = "experiments/scale_5arch_15gb/data_slices",
    manifests_dir: str = "experiments/scale_5arch_15gb/data_manifests",
    train_samples_per_corpus: int = 100,
    val_probe_samples: int = 50,
) -> Dict[str, Any]:
    slicer = DatasetSlicer(slices_dir=slices_dir, manifests_dir=manifests_dir)
    results: Dict[str, Any] = {}

    train_specs = [
        ("wildchat", "allenai/WildChat-1M", None, None, "train", "wildchat_train.jsonl", "wildchat_manifest.json", format_wildchat),
        ("wikipedia", "wikimedia/wikipedia", "20231101.en", None, "train", "wikipedia_train.jsonl", "wikipedia_manifest.json", format_wikipedia),
        ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", None, "train", "fineweb_edu_train.jsonl", "fineweb_manifest.json", format_fineweb_edu),
        ("natural_questions", "google-research-datasets/natural_questions", None, None, "train", "natural_questions_train.jsonl", "natural_questions_manifest.json", format_natural_questions),
        ("ms_marco", "microsoft/ms_marco", "v2.1", None, "train", "ms_marco_train.jsonl", "ms_marco_manifest.json", format_ms_marco),
        ("hotpot_qa", "hotpotqa/hotpot_qa", "fullwiki", None, "train", "hotpot_qa_train.jsonl", "hotpot_manifest.json", format_hotpot_qa),
        ("scientific_papers", "ccdv/arxiv-summarization", None, None, "train", "scientific_papers_train.jsonl", "scientific_papers_manifest.json", format_scientific_papers),
        ("stackexchange", "HuggingFaceH4/stack-exchange-preferences", None, "data/3dprinting.stackexchange.com/train-00000-of-00001.parquet", "train", "stackexchange_train.jsonl", "stackexchange_manifest.json", format_stackexchange),
        ("oasst2", "OpenAssistant/oasst2", None, None, "train", "oasst2_train.jsonl", "oasst2_manifest.json", format_oasst2),
        ("cc_news", "vblagoje/cc_news", None, None, "train", "cc_news_train.jsonl", "cc_news_manifest.json", format_cc_news),
    ]

    for name, ds_id, conf, dfiles, split, outfile, manfile, fmt in train_specs:
        out_p = os.path.join(slices_dir, outfile)
        man_p = os.path.join(manifests_dir, manfile)
        if os.path.exists(out_p) and os.path.exists(man_p) and verify_manifest(out_p, man_p):
            print(f"Corpus {name} already sliced and verified with SHA-256.", flush=True)
            with open(man_p, "r", encoding="utf-8") as f:
                results[name] = json.load(f)
            continue
        try:
            print(f"Streaming and slicing training corpus: {name} ({ds_id})...", flush=True)
            m = slicer.slice_streaming_dataset(
                dataset_id=ds_id,
                split=split,
                output_filename=outfile,
                manifest_filename=manfile,
                config_name=conf,
                data_files=dfiles,
                max_samples=train_samples_per_corpus,
                formatter=fmt,
            )
            print(f"Successfully sliced {name}: {m['sample_count']} samples, SHA-256: {m['sha256'][:16]}...", flush=True)
            results[name] = m
        except Exception as exc:
            print(f"Failed to slice {name}: {exc}", flush=True)
            results[name] = {"status": "ERROR", "error": str(exc)}

    val_specs = [
        ("wikitext103", "Salesforce/wikitext", "wikitext-103-raw-v1", "validation", "wikitext103_val.jsonl", "wikitext103_manifest.json", format_wikitext),
        ("squad_v2", "rajpurkar/squad_v2", None, "validation", "squad_v2_val.jsonl", "squad_v2_manifest.json", format_squad_qa),
        ("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "test", "arc_challenge_test.jsonl", "arc_challenge_manifest.json", format_arc),
        ("truthful_qa", "truthfulqa/truthful_qa", "multiple_choice", "validation", "truthful_qa_val.jsonl", "truthful_qa_manifest.json", format_truthful_qa),
        ("mmlu", "cais/mmlu", "all", "test", "mmlu_test.jsonl", "mmlu_manifest.json", format_mmlu),
        ("gsm8k", "openai/gsm8k", "main", "test", "gsm8k_test.jsonl", "gsm8k_manifest.json", format_gsm8k),
    ]

    for name, ds_id, conf, split, outfile, manfile, fmt in val_specs:
        out_p = os.path.join(slices_dir, outfile)
        man_p = os.path.join(manifests_dir, manfile)
        if os.path.exists(out_p) and os.path.exists(man_p) and verify_manifest(out_p, man_p):
            print(f"Evaluation probe {name} already sliced and verified with SHA-256.", flush=True)
            with open(man_p, "r", encoding="utf-8") as f:
                results[name] = json.load(f)
            continue
        try:
            print(f"Streaming and slicing evaluation probe: {name} ({ds_id})...", flush=True)
            m = slicer.slice_streaming_dataset(
                dataset_id=ds_id,
                split=split,
                output_filename=outfile,
                manifest_filename=manfile,
                config_name=conf,
                max_samples=val_probe_samples,
                formatter=fmt,
            )
            print(f"Successfully sliced probe {name}: {m['sample_count']} samples, SHA-256: {m['sha256'][:16]}...", flush=True)
            results[name] = m
        except Exception as exc:
            print(f"Failed to slice probe {name}: {exc}", flush=True)
            results[name] = {"status": "ERROR", "error": str(exc)}

    return results


if __name__ == "__main__":
    res = build_all_real_slices()
    print("Slicing summary:", json.dumps(res, indent=2))
