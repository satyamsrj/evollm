import json
import random
from typing import Any, Dict, List, Set, Tuple


class EvolutionStreamBuilder:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def build_stream(
        self,
        event_sources: List[Dict[str, Any]],
        output_filepath: str,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> Dict[str, Any]:
        rng = random.Random(self.seed)

        all_events: List[Dict[str, Any]] = []
        for src in event_sources:
            records = src.get("records", [])
            for r in records:
                all_events.append(r)

        rng.shuffle(all_events)

        conv_ids = sorted(list({e["conversation_id"] for e in all_events}))
        rng.shuffle(conv_ids)

        n_total = len(conv_ids)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])

        train_convs: Set[str] = set(conv_ids[:n_train])
        val_convs: Set[str] = set(conv_ids[n_train : n_train + n_val])
        test_convs: Set[str] = set(conv_ids[n_train + n_val :])

        assert train_convs.isdisjoint(val_convs), "Train and val conversation IDs must be disjoint"
        assert train_convs.isdisjoint(test_convs), "Train and test conversation IDs must be disjoint"
        assert val_convs.isdisjoint(test_convs), "Val and test conversation IDs must be disjoint"

        assigned_events: List[Dict[str, Any]] = []
        for idx, event in enumerate(all_events):
            cid = event["conversation_id"]
            if cid in train_convs:
                split = "train"
            elif cid in val_convs:
                split = "val"
            else:
                split = "test"

            tagged = dict(event)
            tagged["event_id"] = idx + 1
            tagged["split"] = split
            assigned_events.append(tagged)

        with open(output_filepath, "w", encoding="utf-8") as f:
            for item in assigned_events:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        summary = {
            "total_events": len(assigned_events),
            "train_events": sum(1 for e in assigned_events if e["split"] == "train"),
            "val_events": sum(1 for e in assigned_events if e["split"] == "val"),
            "test_events": sum(1 for e in assigned_events if e["split"] == "test"),
            "total_conversations": n_total,
            "train_conversations": len(train_convs),
            "val_conversations": len(val_convs),
            "test_conversations": len(test_convs),
        }
        return summary

