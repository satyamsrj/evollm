import os
import sys
import argparse
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed


def create_session(pool_size: int = 16) -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"]
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    return session


def get_file_info(session: requests.Session, url: str):
    resp = session.head(url, allow_redirects=True, timeout=15)
    resp.raise_for_status()
    size = int(resp.headers.get("Content-Length", 0))
    accept_ranges = resp.headers.get("Accept-Ranges", "")
    return size, accept_ranges == "bytes", resp.url


def download_chunk(session: requests.Session, url: str, start: int, end: int, part_path: str, chunk_idx: int, retries: int = 10):
    expected_len = end - start + 1
    if os.path.exists(part_path) and os.path.getsize(part_path) == expected_len:
        return expected_len

    for attempt in range(retries):
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            with session.get(url, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)

            actual_len = os.path.getsize(part_path)
            if actual_len == expected_len:
                return expected_len
            else:
                if os.path.exists(part_path):
                    os.remove(part_path)
        except Exception as e:
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except Exception:
                    pass
            if attempt == retries - 1:
                raise e
            time.sleep(1.0 + attempt * 0.5)


def parallel_download(repo_id: str, filename: str, dest_dir: str, num_threads: int = 12, chunk_size_mb: int = 4):
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"Resolving {url}...", flush=True)

    session = create_session(pool_size=num_threads)
    size, supports_ranges, final_url = get_file_info(session, url)
    print(f"File size: {size / (1024*1024):.2f} MB, Range support: {supports_ranges}", flush=True)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) == size:
        print(f"File {dest_path} already complete ({size} bytes). Skipping.", flush=True)
        return dest_path

    chunk_bytes = chunk_size_mb * 1024 * 1024
    chunks = []
    start = 0
    idx = 0
    while start < size:
        end = min(start + chunk_bytes - 1, size - 1)
        part_path = f"{dest_path}.part_{idx:05d}"
        chunks.append((start, end, part_path, idx))
        start = end + 1
        idx += 1

    print(f"Downloading {len(chunks)} chunks of ~{chunk_size_mb}MB using {num_threads} pooled connections...", flush=True)
    t0 = time.time()
    downloaded_bytes = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(download_chunk, session, url, c[0], c[1], c[2], c[3]): c
            for c in chunks
        }
        for future in as_completed(futures):
            c_bytes = future.result()
            downloaded_bytes += c_bytes
            pct = (downloaded_bytes / size) * 100.0
            elapsed = time.time() - t0
            speed_mb = (downloaded_bytes / (1024 * 1024)) / max(elapsed, 0.001)
            print(f"\rProgress: {pct:5.1f}% ({downloaded_bytes/(1024*1024):.1f}/{size/(1024*1024):.1f} MB) | Speed: {speed_mb:.2f} MB/s", end="", flush=True)

    print(f"\nAssembling {len(chunks)} parts into {dest_path}...", flush=True)
    with open(dest_path, "wb") as out_f:
        for _, _, part_path, _ in chunks:
            with open(part_path, "rb") as in_f:
                while True:
                    buf = in_f.read(1024 * 1024)
                    if not buf:
                        break
                    out_f.write(buf)
            try:
                os.remove(part_path)
            except Exception:
                pass

    final_size = os.path.getsize(dest_path)
    print(f"Download complete: {final_size} bytes in {time.time()-t0:.1f}s -> {dest_path}", flush=True)
    return dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--filename", type=str, default="model.safetensors")
    parser.add_argument("--dest_dir", type=str, required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--chunk_size_mb", type=int, default=4)
    args = parser.parse_args()
    parallel_download(args.repo, args.filename, args.dest_dir, args.threads, args.chunk_size_mb)
