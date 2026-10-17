"""Robust multi-threaded chunked downloader with resume support for HuggingFace models.

Bypasses CloudFront / Schannel TCP abrupt resets by fetching in 16MB slices
with concurrent HTTP 206 Partial Content requests, automatic retries,
and persistent chunk-state tracking (.chunks.json).
"""
import concurrent.futures
import json
import os
import socket
import sys
import time
from pathlib import Path
import requests

socket.setdefaulttimeout(35)

CHUNK_SIZE = 16 * 1024 * 1024  # 16 MB per chunk
MAX_WORKERS = int(os.environ.get("CHUNK_WORKERS", "6"))


def get_remote_size(url):
    """Resolve redirects and get Content-Length."""
    headers = {"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"}
    with requests.get(url, headers=headers, stream=True, timeout=(15, 30), allow_redirects=True) as resp:
        resp.raise_for_status()
        final_url = resp.url
        cr = resp.headers.get("Content-Range")
        if cr:
            # format: bytes 0-0/1676798532
            total = int(cr.split("/")[-1])
            return total, final_url
        cl = resp.headers.get("Content-Length")
        if cl:
            return int(cl), final_url
    raise RuntimeError(f"Could not determine content length for {url}")


def download_chunk(url, dest_path, start, end, chunk_idx):
    """Download a single byte range with requests and write directly to disk."""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={start}-{end}"
    }
    expected_len = end - start + 1
    for attempt in range(15):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(12, 30)) as resp:
                if resp.status_code not in (200, 206):
                    resp.raise_for_status()
                data = bytearray()
                for part in resp.iter_content(chunk_size=512 * 1024):
                    if part:
                        data.extend(part)
                if len(data) != expected_len:
                    raise IOError(f"Incomplete chunk {chunk_idx}: got {len(data)}, expected {expected_len}")
                with open(dest_path, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                return chunk_idx, len(data)
        except Exception as e:
            time.sleep(min(10, 1.5 * (attempt + 1)))
            if attempt == 14:
                raise RuntimeError(f"Permanently failed chunk {chunk_idx} ({start}-{end}): {e}")


def download_file(url, destination, name=None):
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    name = name or dest.name
    state_file = dest.with_name(dest.name + ".chunks.json")

    print(f"\n[Connecting] {name}...", flush=True)
    total_size, final_url = get_remote_size(url)
    print(f"[Target] {name} size: {total_size / 1e6:.1f} MB ({total_size} bytes)", flush=True)

    # Check if already fully downloaded
    if dest.exists() and dest.stat().st_size == total_size and not state_file.exists():
        with open(dest, "rb") as f:
            header = f.read(8)
            if any(header):
                print(f"[ALREADY PRESENT] {name} ({total_size / 1e6:.1f} MB)", flush=True)
                return True

    # Build chunk map
    chunks = []
    start = 0
    idx = 0
    while start < total_size:
        end = min(start + CHUNK_SIZE - 1, total_size - 1)
        chunks.append((start, end, idx))
        start = end + 1
        idx += 1

    total_chunks = len(chunks)

    # Load state or initialize
    completed_set = set()
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            if data.get("total_size") == total_size:
                completed_set = set(data.get("completed", []))
                print(f"[Resuming] {name}: {len(completed_set)}/{total_chunks} chunks already done", flush=True)
        except Exception:
            completed_set = set()
    elif dest.exists() and dest.stat().st_size > 0:
        existing_size = dest.stat().st_size
        for c in chunks:
            if c[1] < existing_size:
                completed_set.add(c[2])
        if completed_set:
            print(f"[Resuming from pre-existing file] {name}: {len(completed_set)}/{total_chunks} chunks already done", flush=True)

    # Pre-allocate if not created, without overwriting existing data
    if not dest.exists():
        with open(dest, "wb") as f:
            f.seek(total_size - 1)
            f.write(b"\0")
    elif dest.stat().st_size != total_size:
        with open(dest, "r+b") as f:
            f.seek(total_size - 1)
            f.write(b"\0")

    pending_chunks = [c for c in chunks if c[2] not in completed_set]
    if not pending_chunks:
        if state_file.exists():
            state_file.unlink()
        print(f"[VERIFIED] {name} (100% complete)", flush=True)
        return True

    print(f"[Downloading] {name}: {len(pending_chunks)} chunks to download ({CHUNK_SIZE / 1e6:.1f} MB each) with {MAX_WORKERS} workers...", flush=True)

    t0 = time.time()
    completed_bytes = sum(c[1] - c[0] + 1 for c in chunks if c[2] in completed_set)
    done_count = len(completed_set)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_chunk, url, dest, c[0], c[1], c[2]): c
            for c in pending_chunks
        }
        for future in concurrent.futures.as_completed(futures):
            chunk_idx, nbytes = future.result()
            completed_set.add(chunk_idx)
            completed_bytes += nbytes
            done_count += 1

            # Save state
            state_file.write_text(json.dumps({"total_size": total_size, "completed": list(completed_set)}), encoding="utf-8")

            elapsed = max(0.1, time.time() - t0)
            pct = (completed_bytes / total_size) * 100
            speed = ((completed_bytes - (total_size - len(pending_chunks) * CHUNK_SIZE)) / 1e6) / elapsed
            speed = max(0.1, speed)
            rem_sec = int((total_size - completed_bytes) / (speed * 1e6)) if speed > 0 else 0
            if done_count % 5 == 0 or done_count == total_chunks:
                print(f"[{done_count}/{total_chunks}] {pct:.1f}% ({completed_bytes / 1e6:.1f} / {total_size / 1e6:.1f} MB) @ {speed:.2f} MB/s, ETA {rem_sec}s", flush=True)

    if state_file.exists():
        state_file.unlink()

    elapsed = max(1.0, time.time() - t0)
    print(f"[COMPLETED] {name} in {elapsed:.1f}s", flush=True)
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python chunk_downloader.py <url> <destination>")
        sys.exit(1)
    download_file(sys.argv[1], sys.argv[2])
