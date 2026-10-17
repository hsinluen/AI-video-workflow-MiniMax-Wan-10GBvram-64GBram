"""Automated model downloader using Windows native curl.exe with resume support.

Supports:
- Qwen-Image 2.1 (首幀/定妝照)
- Breeze TTS 2 & BreezyVoice (語音合成)
- LTX-Video (INT8/FP16 2B Checkpoint + VAE for 10GB VRAM)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "setup"
CURL = r"C:\Windows\System32\curl.exe"


def download_manifest(manifest_path, mirror=True):
    path = Path(manifest_path)
    if not path.is_file():
        path = MANIFEST_DIR / manifest_path
    if not path.is_file():
        print(f"[skip] Manifest {manifest_path} not found")
        return False

    items = json.loads(path.read_text(encoding="utf-8-sig"))
    print(f"\n==========================================")
    print(f"Processing manifest: {path.name} ({len(items)} items)")
    print(f"==========================================")

    for idx, item in enumerate(items, 1):
        url = item["url"]
        dest = Path(item["destination"].replace("\\", "/"))
        size = item.get("size", 0)
        name = item.get("name", dest.name)

        if dest.exists():
            curr_size = dest.stat().st_size
            if size == 0 or curr_size == size:
                print(f"[{idx}/{len(items)}] [ALREADY PRESENT] {name} ({curr_size / 1e6:.1f} MB)")
                continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        part_file = dest.with_name(dest.name + ".part")

        # Use hf-mirror if direct connection is throttled or preferred
        download_url = url
        if mirror and "huggingface.co" in url:
            download_url = url.replace("https://huggingface.co", "https://hf-mirror.com")

        print(f"[{idx}/{len(items)}] [DOWNLOADING] {name} (expected {size / 1e6:.1f} MB)...", flush=True)

        cmd = [
            CURL, "-L", "-C", "-",
            "--retry", "5", "--retry-delay", "3",
            "-o", str(part_file),
            download_url
        ]

        t0 = time.time()
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[WARN] Failed with mirror, trying original URL: {url}", flush=True)
            cmd[-1] = url
            ret = subprocess.run(cmd)

        if ret.returncode == 0 and part_file.exists():
            part_size = part_file.stat().st_size
            if size == 0 or part_size == size:
                if dest.exists():
                    dest.unlink()
                part_file.rename(dest)
                elapsed = max(1, time.time() - t0)
                speed = (part_size / 1e6) / elapsed
                print(f"[{idx}/{len(items)}] [VERIFIED] {name} in {elapsed:.1f}s ({speed:.1f} MB/s)", flush=True)
            else:
                print(f"[{idx}/{len(items)}] [SIZE MISMATCH] {name}: got {part_size}, expected {size}", flush=True)
        else:
            print(f"[{idx}/{len(items)}] [ERROR] Failed to download {name}", flush=True)

    return True


def main():
    manifests = sys.argv[1:]
    if not manifests:
        manifests = [
            "downloads_ltx_video.json",
            "downloads_qwen_image.json",
            "downloads_breeze_tts2.json",
            "downloads_breezyvoice.json"
        ]

    for m in manifests:
        download_manifest(m)


if __name__ == "__main__":
    main()
