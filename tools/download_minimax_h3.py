import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from chunk_downloader import download_file

COMFY_MODELS = Path(r"C:\AI\H3\ComfyUI-0.36.0\models")

TARGETS = [
    (
        "https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned-Q4_K.gguf",
        COMFY_MODELS / "diffusion_models" / "minimax_h3_fl2va_pruned-Q4_K.gguf",
        "MiniMax-H3 FL2VA Pruned Q4_K GGUF (10.64 GB)"
    ),
    (
        "https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
        COMFY_MODELS / "text_encoders" / "qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
        "Qwen3-VL 32B MiniMax-H3 Q2_K_M GGUF (12.20 GB)"
    )
]

def main():
    print("=== MiniMax-H3 10GB VRAM Optimized GGUF Model Downloader ===", flush=True)
    for url, dest, name in TARGETS:
        print(f"\n[DOWNLOAD] Starting: {name}", flush=True)
        ok = download_file(url, dest, name=name)
        if not ok:
            print(f"[ERROR] Failed to download {name}", flush=True)
            sys.exit(1)
        print(f"[SUCCESS] Downloaded and verified: {name}", flush=True)
    print("\n🎉 All MiniMax-H3 models downloaded and verified successfully!", flush=True)

if __name__ == "__main__":
    main()
