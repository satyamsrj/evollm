import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_manifest(data_path: str, manifest_path: str) -> bool:
    if not os.path.exists(data_path) or not os.path.exists(manifest_path):
        return False
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    calculated_hash = compute_sha256(data_path)
    if calculated_hash != manifest.get("sha256"):
        return False

    with open(data_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)

    return line_count == manifest.get("sample_count")


class DatasetSlicer:
    def __init__(self, slices_dir: str = "experiments/scale_5arch_15gb/data_slices", manifests_dir: str = "experiments/scale_5arch_15gb/data_manifests"):
        self.slices_dir = slices_dir
        self.manifests_dir = manifests_dir
        os.makedirs(self.slices_dir, exist_ok=True)
        os.makedirs(self.manifests_dir, exist_ok=True)

    def slice_streaming_dataset(
        self,
        dataset_id: str,
        split: str,
        output_filename: str,
        manifest_filename: str,
        config_name: Optional[str] = None,
        data_files: Optional[Any] = None,
        max_samples: int = 500,
        max_tokens: int = 128,
        target_size_mb: Optional[float] = None,
        formatter: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        output_path = os.path.join(self.slices_dir, output_filename)
        manifest_path = os.path.join(self.manifests_dir, manifest_filename)

        if resume and os.path.exists(output_path) and os.path.exists(manifest_path):
            if verify_manifest(output_path, manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Hugging Face datasets library is required: pip install datasets") from exc

        try:
            kwargs: Dict[str, Any] = {"streaming": True}
            if data_files is not None:
                kwargs["data_files"] = data_files
            if config_name:
                ds = load_dataset(dataset_id, config_name, split=split, **kwargs)
            else:
                ds = load_dataset(dataset_id, split=split, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"SOURCE_UNAVAILABLE: Failed to stream dataset {dataset_id}: {exc}") from exc

        collected: List[Dict[str, Any]] = []
        bytes_written = 0
        target_bytes = int(target_size_mb * 1024 * 1024) if target_size_mb is not None else None

        for row in ds:
            if len(collected) >= max_samples:
                break
            if target_bytes is not None and bytes_written >= target_bytes:
                break

            record = formatter(row) if formatter else row
            if record is None:
                continue

            text_repr = json.dumps(record, ensure_ascii=False)
            if len(text_repr.split()) > max_tokens * 2:
                continue

            line_bytes = len((text_repr + "\n").encode("utf-8"))
            bytes_written += line_bytes
            collected.append(record)

        if len(collected) == 0:
            raise RuntimeError(f"SOURCE_UNAVAILABLE: Zero valid records parsed from {dataset_id}")

        with open(output_path, "w", encoding="utf-8") as f:
            for item in collected:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        sha256_val = compute_sha256(output_path)
        actual_bytes = os.path.getsize(output_path)

        manifest = {
            "dataset_id": dataset_id,
            "config_name": config_name,
            "split": split,
            "sample_count": len(collected),
            "max_tokens": max_tokens,
            "sha256": sha256_val,
            "byte_size": actual_bytes,
            "target_size_mb": target_size_mb,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def slice_and_save(
        self,
        dataset_id: str,
        split: str,
        output_filename: str,
        manifest_filename: str,
        config_name: Optional[str] = None,
        max_samples: int = 500,
        target_size_mb: Optional[float] = None,
        formatter: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        return self.slice_streaming_dataset(
            dataset_id=dataset_id,
            split=split,
            output_filename=output_filename,
            manifest_filename=manifest_filename,
            config_name=config_name,
            max_samples=max_samples,
            target_size_mb=target_size_mb,
            formatter=formatter,
        )

