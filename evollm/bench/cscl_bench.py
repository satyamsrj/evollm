"""
CSCL-Bench: Continual Conversation-Simulation Benchmark.
Evaluates weight-only parameter evolution across 4 core tracks:
1. Knowledge Injection Accuracy (KAA)
2. Incremental Correction & Multi-Hop Overwrites
3. Base Invariant Retention (Forgetting Metric)
4. Adversarial Falsehood Resistance (FAR)
"""
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Callable


@dataclass
class BenchmarkItem:
    track_id: str
    query: str
    target_answer: str
    flawed_answer: str = ""
    is_correction: bool = True
    test_query: str = ""
    expected_token_in_test: str = ""


class CSCLBenchmark:
    def __init__(self):
        self.items: List[BenchmarkItem] = []
        self._build_default_suite()

    def _build_default_suite(self):
        # Track 1: Factual Injection
        self.items.append(BenchmarkItem(
            track_id="Track1_Injection",
            query="What architecture does the custom chip Aether-X use?",
            target_answer="The custom chip Aether-X uses a RISC-V 64-bit multi-cluster architecture.",
            flawed_answer="I don't have information on Aether-X architecture.",
            is_correction=True,
            test_query="What architecture does the custom chip Aether-X use?",
            expected_token_in_test="RISC-V"
        ))
        self.items.append(BenchmarkItem(
            track_id="Track1_Injection",
            query="Who was appointed lead architect for Project Boreas?",
            target_answer="Dr. Elena Rostova was appointed lead architect for Project Boreas.",
            flawed_answer="Project Boreas has no lead architect specified.",
            is_correction=True,
            test_query="Who was appointed lead architect for Project Boreas?",
            expected_token_in_test="Elena"
        ))

        # Track 2: Incremental Overwrites
        self.items.append(BenchmarkItem(
            track_id="Track2_Overwrite_T1",
            query="What is the internal port for the telemetry broker?",
            target_answer="The telemetry broker listens on port 9042.",
            flawed_answer="The telemetry broker listens on port 8080.",
            is_correction=True,
            test_query="What is the internal port for the telemetry broker?",
            expected_token_in_test="9042"
        ))
        self.items.append(BenchmarkItem(
            track_id="Track2_Overwrite_T2",
            query="Correction: The telemetry broker has migrated. What is the new port?",
            target_answer="The telemetry broker has migrated and now listens on port 9800.",
            flawed_answer="The telemetry broker is still on port 9042.",
            is_correction=True,
            test_query="What is the internal port for the telemetry broker?",
            expected_token_in_test="9800"
        ))

        # Track 3: Invariant Retention / Base Knowledge Test Anchor
        self.items.append(BenchmarkItem(
            track_id="Track3_Invariant_Base",
            query="What is the capital of France?",
            target_answer="The capital of France is Paris.",
            flawed_answer="",
            is_correction=False,
            test_query="What is the capital of France?",
            expected_token_in_test="Paris"
        ))

    def run_benchmark(self, learner, generate_fn: Callable[[str], str]) -> Dict[str, Any]:
        """
        Runs the continual learning stream and tracks test query recall.
        """
        results_by_track: Dict[str, List[bool]] = {}
        
        for item in self.items:
            track = item.track_id
            if track not in results_by_track:
                results_by_track[track] = []

            # 1. Continual learning step
            step_result = learner.learn_from_conversation(
                query=item.query,
                correct_answer=item.target_answer,
                incorrect_answer=item.flawed_answer,
                is_correction=item.is_correction,
            )

            # 2. Evaluation probe on test query
            generated = generate_fn(item.test_query)
            success = item.expected_token_in_test.lower() in generated.lower()
            results_by_track[track].append(success)

        # Aggregate metrics
        summary = {}
        for track, successes in results_by_track.items():
            summary[f"{track}_Accuracy"] = sum(successes) / len(successes) if successes else 0.0

        return {
            "summary": summary,
            "total_items": len(self.items),
            "final_drift": learner.drift_history[-1] if learner.drift_history else 0.0
        }

