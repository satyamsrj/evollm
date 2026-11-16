import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BenchmarkItem:
    track_id:                str
    query:                   str
    target_answer:           str
    flawed_answer:           str = ""
    is_correction:           bool = True
    source_reference:        Optional[str] = None
    test_query:              str = ""
    expected_token_in_test:  str = ""


@dataclass
class TrackResult:
    track_id:    str
    total:       int
    correct:     int
    skipped:     int
    accuracy:    float
    skip_reasons: List[str] = field(default_factory=list)


class CSCLBenchmark:
    def __init__(self):
        self.items: List[BenchmarkItem] = []
        self._build_default_suite()

    def _build_default_suite(self) -> None:
        self.items.extend([
            BenchmarkItem(
                track_id="Track1_Injection",
                query="What architecture does the custom chip Aether-X use?",
                target_answer="The custom chip Aether-X uses a RISC-V 64-bit multi-cluster architecture.",
                flawed_answer="I don't have information on Aether-X architecture.",
                is_correction=True,
                source_reference="aether_x_datasheet.pdf#section=2",
                test_query="What architecture does the custom chip Aether-X use?",
                expected_token_in_test="RISC-V",
            ),
            BenchmarkItem(
                track_id="Track1_Injection",
                query="Who was appointed lead architect for Project Boreas?",
                target_answer="Dr. Elena Rostova was appointed lead architect for Project Boreas.",
                flawed_answer="Project Boreas has no lead architect specified.",
                is_correction=True,
                source_reference="project_boreas_org_chart_v3.pdf",
                test_query="Who was appointed lead architect for Project Boreas?",
                expected_token_in_test="Elena",
            ),
            BenchmarkItem(
                track_id="Track2_Overwrite_T1",
                query="What is the internal port for the telemetry broker?",
                target_answer="The telemetry broker listens on port 9042.",
                flawed_answer="The telemetry broker listens on port 8080.",
                is_correction=True,
                source_reference="telemetry_config_v1.yaml#line=14",
                test_query="What is the internal port for the telemetry broker?",
                expected_token_in_test="9042",
            ),
            BenchmarkItem(
                track_id="Track2_Overwrite_T2",
                query="Correction: The telemetry broker has migrated. What is the new port?",
                target_answer="The telemetry broker has migrated and now listens on port 9800.",
                flawed_answer="The telemetry broker is still on port 9042.",
                is_correction=True,
                source_reference="telemetry_config_v2.yaml#line=14",
                test_query="What is the internal port for the telemetry broker?",
                expected_token_in_test="9800",
            ),
            BenchmarkItem(
                track_id="Track3_Invariant_Base",
                query="What is the capital of France?",
                target_answer="The capital of France is Paris.",
                flawed_answer="",
                is_correction=False,
                source_reference="verified_world_facts_v1.json",
                test_query="What is the capital of France?",
                expected_token_in_test="Paris",
            ),
            BenchmarkItem(
                track_id="Track4_FalseRejection",
                query="Is it true that water boils at 20 degrees Celsius at standard pressure?",
                target_answer="Water boils at 100 degrees Celsius at standard atmospheric pressure.",
                flawed_answer="Yes, water boils at 20 degrees Celsius.",
                is_correction=True,
                source_reference="nist_thermophysical_properties_water.pdf",
                test_query="At what temperature does water boil at standard pressure?",
                expected_token_in_test="100",
            ),
        ])

    def run_benchmark(
        self,
        learner: Any,
        generate_fn: Callable[[str], str],
        log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        track_buckets: Dict[str, List[dict]] = {}

        for item in self.items:
            tid = item.track_id
            if tid not in track_buckets:
                track_buckets[tid] = []

            learn_result = learner.learn_from_conversation(
                query=item.query,
                correct_answer=item.target_answer,
                incorrect_answer=item.flawed_answer if item.flawed_answer else None,
                is_correction=item.is_correction,
                source_reference=item.source_reference,
            )

            skipped = learn_result.get("status", "").startswith("SKIPPED")

            if not skipped and item.test_query:
                generated = generate_fn(item.test_query)
                hit = item.expected_token_in_test.lower() in generated.lower()
            else:
                generated = ""
                hit = False

            track_buckets[tid].append({
                "query": item.query,
                "learn_status": learn_result.get("status"),
                "skipped": skipped,
                "skip_reason": learn_result.get("status", "") if skipped else None,
                "test_query": item.test_query,
                "generated": generated,
                "hit": hit,
            })

        track_results: List[TrackResult] = []
        for tid, entries in track_buckets.items():
            answered = [e for e in entries if not e["skipped"]]
            correct = sum(1 for e in answered if e["hit"])
            skip_reasons = [e["skip_reason"] for e in entries if e["skipped"] and e["skip_reason"]]
            track_results.append(TrackResult(
                track_id=tid,
                total=len(entries),
                correct=correct,
                skipped=len(entries) - len(answered),
                accuracy=correct / max(len(answered), 1),
                skip_reasons=skip_reasons,
            ))

        summary = {r.track_id: round(r.accuracy, 4) for r in track_results}
        final_drift = learner.drift_history[-1] if learner.drift_history else 0.0

        output = {
            "summary": summary,
            "track_details": [vars(r) for r in track_results],
            "total_items": len(self.items),
            "final_drift": final_drift,
            "skipped_records": learner.skipped_records,
        }

        if log_path:
            with open(log_path, "w") as f:
                json.dump(output, f, indent=2)

        return output
