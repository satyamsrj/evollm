import json
import os
import shutil
import tempfile
import unittest

from evollm.data.slicer import compute_sha256, verify_manifest
from evollm.data.stream_builder import EvolutionStreamBuilder


class TestDataSlicer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.slices_dir = os.path.join(self.test_dir, "slices")
        self.manifests_dir = os.path.join(self.test_dir, "manifests")
        os.makedirs(self.slices_dir, exist_ok=True)
        os.makedirs(self.manifests_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_sha256_and_manifest_verification(self):
        data_path = os.path.join(self.slices_dir, "sample.jsonl")
        manifest_path = os.path.join(self.manifests_dir, "sample.json")

        records = [
            {"conversation_id": "c1", "query": "q1", "text": "a1"},
            {"conversation_id": "c2", "query": "q2", "text": "a2"},
        ]
        with open(data_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        sha_val = compute_sha256(data_path)
        manifest_data = {
            "dataset_id": "mock_dataset",
            "sample_count": len(records),
            "sha256": sha_val,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        self.assertTrue(verify_manifest(data_path, manifest_path))

        # Tampering check
        with open(data_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"conversation_id": "c3", "query": "q3", "text": "a3"}) + "\n")
        self.assertFalse(verify_manifest(data_path, manifest_path))

    def test_zero_train_test_id_overlap(self):
        builder = EvolutionStreamBuilder(seed=42)
        records = [
            {"conversation_id": f"conv_{i}", "query": f"q_{i}", "text": f"ans_{i}"}
            for i in range(100)
        ]
        out_stream = os.path.join(self.slices_dir, "stream.jsonl")
        summary = builder.build_stream([{"records": records}], out_stream, (0.8, 0.1, 0.1))

        self.assertEqual(summary["total_conversations"], 100)
        self.assertEqual(summary["train_conversations"], 80)
        self.assertEqual(summary["val_conversations"], 10)
        self.assertEqual(summary["test_conversations"], 10)

        train_ids = set()
        val_ids = set()
        test_ids = set()

        with open(out_stream, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item["split"] == "train":
                    train_ids.add(item["conversation_id"])
                elif item["split"] == "val":
                    val_ids.add(item["conversation_id"])
                else:
                    test_ids.add(item["conversation_id"])

        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))


if __name__ == "__main__":
    unittest.main()

