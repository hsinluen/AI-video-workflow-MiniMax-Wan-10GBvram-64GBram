"""Unified Four-in-One Multi-Model Shot Generator for 10GB VRAM.

Supported Engines:
  1. ltx      : Lightricks LTX-Video 2B FP8 (Ultra-fast 24/25fps, 30-60s per shot)
  2. wan2.1   : Alibaba Wan 2.1 14B I2V Q4_K_M GGUF (Flagship photo realism, zero ghosting)
  3. hunyuan  : Tencent HunyuanVideo 720p Q4_K_M GGUF (Cinematic Asian lighting & character stability)
  4. h3       : MiniMax-H3 INT8 / FastH3 4-8 steps Turbo (Rapid draft & dialogue alignment)

Usage:
  python tools/h3_shot.py spec.json --out <dir> [--engine ltx|wan2.1|hunyuan|h3]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HOST = os.environ.get("COMFY_HOST", "http://127.0.0.1:8188")
COMFY = Path(os.environ.get("COMFY_DIR", r"C:\AI\H3\ComfyUI-0.36.0"))
FPS = 24


def snap32(dim):
    return max(64, int(round(dim / 32.0) * 32))


def snap16(dim):
    return max(64, int(round(dim / 16.0) * 16))


def frames_for_ltx(duration):
    target = max(9, int(round(duration * FPS)))
    n = max(1, round((target - 1) / 8))
    return int(8 * n + 1)


def frames_for_wan(duration):
    target = max(9, int(round(duration * FPS)))
    n = max(1, round((target - 1) / 4))
    return int(4 * n + 1)


def frames_for_h3(duration):
    frames = int(duration * FPS)
    while frames % 17 != 5:
        frames += 1
    return frames


def req(path, data=None, timeout=120):
    r = urllib.request.Request(
        f"{HOST}{path}",
        data=json.dumps(data).encode("utf-8") if data is not None else None,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stage_file(src, name):
    dest = COMFY / "input" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return name


def find_model_file(subdirs, filenames):
    for sub in subdirs:
        for fn in filenames:
            p = COMFY / "models" / sub / fn
            if p.exists():
                return fn, p
            # check extra paths
            alt = Path(r"C:\AI\H3\models") / sub / fn
            if alt.exists():
                return fn, alt
    return None, None


# ----------------------------------------------------------------------------------------------- Prompt Builder
def build_prompt(spec, engine="ltx"):
    parts = []
    style = spec.get("style", "")
    setting = spec.get("setting", "")
    if style:
        parts.append(style.rstrip("."))
    if setting:
        parts.append(f"Scene setting: {setting.rstrip('.')}")

    for sh in spec.get("shots", []):
        act = sh.get("action", "")
        cam = sh.get("camera", "")
        if cam:
            parts.append(cam.rstrip("."))
        if act:
            parts.append(act.rstrip("."))
        for dlg in sh.get("dialogue", []):
            subj = dlg.get("subject", "")
            line = dlg.get("line", "")
            after = dlg.get("after", "")
            parts.append(f"{subj} speaks '{line}'. {after}".strip())

    if not parts:
        parts.append("Cinematic live action video, high quality motion, natural expressions and lighting.")

    return ". ".join(p.strip() for p in parts if p.strip()) + "."


# ----------------------------------------------------------------------------------------------- LTX-Video Graph
try:
    from ltx_shot import build_graph as build_graph_ltx
except ImportError:
    from tools.ltx_shot import build_graph as build_graph_ltx


# ----------------------------------------------------------------------------------------------- Wan 2.1 Graph
def build_graph_wan(spec, prompt, base, tag):
    w, h = spec.get("size", [832, 480])
    w, h = snap16(w), snap16(h)
    duration = float(spec.get("duration", 3.0))
    length = int(spec.get("length", 0)) or frames_for_wan(duration)
    seed = int(spec.get("seed", 42))
    steps = int(spec.get("steps", 20))  # 20 steps for Wan 2.1 Q4_K_M
    cfg = float(spec.get("cfg", 5.0))

    first_frame_path = spec.get("first_frame")
    if not first_frame_path:
        raise ValueError("spec.json must specify first_frame for Wan 2.1 I2V")
    staged_first = stage_file(base / first_frame_path, f"{tag}_wan_first.png")

    wan_model, _ = find_model_file(["diffusion_models", "unet"], ["wan2.1-i2v-14b-480p-Q4_K_M.gguf", "wan2.1-i2v-14b-720p-Q4_K_M.gguf", "wan2.1_i2v_480p_14B_fp8.safetensors"])
    if not wan_model:
        wan_model = "wan2.1-i2v-14b-480p-Q4_K_M.gguf"

    vae_name, _ = find_model_file(["vae"], ["wan_2.1_vae.safetensors", "wan2.1_vae.safetensors"])
    if not vae_name:
        vae_name = "wan_2.1_vae.safetensors"

    clip_name, _ = find_model_file(["text_encoders", "clip"], ["umt5-xxl-encoder-Q4_K_M.gguf", "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "umt5_xxl_fp8_e4m3fn.safetensors"])
    if not clip_name:
        clip_name = "umt5-xxl-encoder-Q4_K_M.gguf"

    neg_prompt = spec.get("neg_prompt", "色调艳丽，过曝，静态，残影，变形，肢体错误，模糊，低质量")

    is_gguf = wan_model.endswith(".gguf")
    loader_class = "UnetLoaderGGUF" if is_gguf else "UNETLoader"
    clip_loader_class = "CLIPLoaderGGUF" if clip_name.endswith(".gguf") else "CLIPLoader"

    clip_inputs = {"clip_name": clip_name, "type": "wan"}
    if clip_loader_class == "CLIPLoader":
        clip_inputs["device"] = "default"

    g = {
        "1": {"class_type": loader_class, "inputs": {"unet_name": wan_model}},
        "2": {"class_type": "LoadImage", "inputs": {"image": staged_first}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "4": {"class_type": clip_loader_class, "inputs": clip_inputs},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": neg_prompt}},
        "7": {"class_type": "WanImageToVideo", "inputs": {"positive": ["5", 0], "negative": ["6", 0], "vae": ["3", 0], "width": w, "height": h, "length": length, "batch_size": 1, "start_image": ["2", 0]}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["7", 2], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": f"wan_shot_png/{tag}/f"}}
    }
    return g


# ----------------------------------------------------------------------------------------------- HunyuanVideo Graph
def build_graph_hunyuan(spec, prompt, base, tag):
    w, h = spec.get("size", [848, 480])
    w, h = snap16(w), snap16(h)
    duration = float(spec.get("duration", 3.0))
    length = int(spec.get("length", 0)) or frames_for_wan(duration)
    seed = int(spec.get("seed", 42))
    steps = int(spec.get("steps", 25))
    cfg = float(spec.get("cfg", 6.0))

    hunyuan_model, _ = find_model_file(["diffusion_models", "unet"], ["hunyuan-video-t2v-720p-Q4_K_M.gguf", "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors"])
    if not hunyuan_model:
        hunyuan_model = "hunyuan-video-t2v-720p-Q4_K_M.gguf"

    vae_name, _ = find_model_file(["vae"], ["hunyuan_video_vae_bf16.safetensors", "hunyuan_video_vae.safetensors"])
    if not vae_name:
        vae_name = "hunyuan_video_vae_bf16.safetensors"

    clip_l, _ = find_model_file(["clip", "text_encoders"], ["clip_l.safetensors"])
    clip_llama, _ = find_model_file(["text_encoders", "clip"], ["llava_llama3_fp8_scaled.safetensors"])
    if clip_l and clip_llama:
        clip_node = {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": clip_l, "clip_name2": clip_llama, "type": "hunyuan_video"}}
    else:
        clip_name, _ = find_model_file(["clip", "text_encoders"], ["t5xxl_fp8_e4m3fn.safetensors", "llava_llama3_fp8_scaled.safetensors"])
        clip_node = {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name or "clip_l.safetensors", "type": "hunyuan_video", "device": "default"}}

    neg_prompt = spec.get("neg_prompt", "aerial view, deformed, extra limbs, bad anatomy, blur, static")
    is_gguf = hunyuan_model.endswith(".gguf")
    loader_class = "UnetLoaderGGUF" if is_gguf else "UNETLoader"

    g = {
        "1": {"class_type": loader_class, "inputs": {"unet_name": hunyuan_model}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "3": clip_node,
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": neg_prompt}},
        "6": {"class_type": "EmptyHunyuanLatentVideo", "inputs": {"width": w, "height": h, "length": length, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": f"hunyuan_shot_png/{tag}/f"}}
    }
    return g


# ----------------------------------------------------------------------------------------------- MiniMax-H3 Graph
def build_graph_h3(spec, prompt, base, tag):
    w, h = spec.get("size", [960, 544])
    w, h = snap32(w), snap32(h)
    duration = float(spec.get("duration", 3.0))
    length = frames_for_h3(duration)
    seed = int(spec.get("seed", 42))
    steps = int(spec.get("steps", 8))  # FastH3 4-8 steps default for 10GB VRAM

    h3_model, _ = find_model_file(["diffusion_models", "unet"], ["minimax_h3_fl2va_pruned-Q4_K.gguf", "minimax_h3_ref2va_pruned-Q4_K.gguf", "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"])
    if not h3_model:
        h3_model = "minimax_h3_fl2va_pruned-Q4_K.gguf"

    clip_name, _ = find_model_file(["text_encoders", "clip"], ["qwen3vl_32b_minimax_h3-Q2_K_M.gguf", "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"])
    if not clip_name:
        clip_name = "qwen3vl_32b_minimax_h3-Q2_K_M.gguf"

    v_vae, _ = find_model_file(["vae"], ["minimax_h3_video_vae_fp16.safetensors"])
    a_vae, _ = find_model_file(["vae"], ["minimax_h3_audio_vae_fp32.safetensors"])
    lora_name, _ = find_model_file(["loras"], ["minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"])

    first_frame_path = spec.get("first_frame")
    if not first_frame_path:
        raise ValueError("spec.json must specify first_frame for H3")
    staged_first = stage_file(base / first_frame_path, f"{tag}_h3_first.png")

    is_gguf = h3_model.endswith(".gguf")
    loader_class = "UnetLoaderGGUF" if is_gguf else "UNETLoader"
    clip_loader_class = "CLIPLoaderGGUF" if clip_name.endswith(".gguf") else "CLIPLoader"

    g = {
        "1": {"class_type": loader_class, "inputs": {"unet_name": h3_model}},
        "2": {"class_type": clip_loader_class, "inputs": {"clip_name": clip_name, "type": "minimax"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": v_vae or "minimax_h3_video_vae_fp16.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": a_vae or "minimax_h3_audio_vae_fp32.safetensors"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": staged_first}},
    }

    model_node = ["1", 0]
    clip_node = ["2", 0]
    if lora_name:
        g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model_node, "lora_name": lora_name, "strength_model": 1.0}}
        model_node = ["lora", 0]

    g["6"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": clip_node, "vae": ["3", 0], "prompt": prompt, "width": w, "height": h, "length": length, "first_frame": ["5", 0]}}
    g["7"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["8"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    g["9"] = {"class_type": "BasicScheduler", "inputs": {"model": model_node, "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    g["10"] = {"class_type": "BasicGuider", "inputs": {"model": model_node, "conditioning": ["6", 0]}}
    g["11"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["7", 0], "guider": ["10", 0], "sampler": ["8", 0], "sigmas": ["9", 0], "latent_image": ["6", 1]}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"h3_shot_png/{tag}/f"}}

    return g


# ----------------------------------------------------------------------------------------------- Unified Execution
def run_generation(spec_path: Path, out_dir: Path, engine: str):
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    tag = out_dir.name
    base = spec_path.parent
    prompt = build_prompt(spec, engine=engine)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    engine = engine.lower()
    print(f"[{engine.upper()}] Building ComfyUI workflow graph (Engine: {engine})...", flush=True)

    if engine == "ltx":
        res = build_graph_ltx(spec, prompt, base, tag)
        g = res[0] if isinstance(res, tuple) else res
    elif engine in ("wan", "wan2.1"):
        g = build_graph_wan(spec, prompt, base, tag)
    elif engine in ("hunyuan", "hunyuanvideo"):
        g = build_graph_hunyuan(spec, prompt, base, tag)
    elif engine in ("wan", "wan2.1"):
        g = build_graph_wan(spec, prompt, base, tag)
    elif engine in ("h3", "minimax"):
        g = build_graph_h3(spec, prompt, base, tag)
    else:
        print(f"[WARN] Unknown engine '{engine}', defaulting to h3 (MiniMax-H3)")
        g = build_graph_h3(spec, prompt, base, tag)

    (out_dir / "workflow_api.json").write_text(json.dumps(g, indent=2), encoding="utf-8")

    # Clean up any leftover PNG frames in ComfyUI output directory for this tag
    for prefix in (f"{engine}_shot_png", "h3_shot_png", "wan_shot_png", "shot_png"):
        tgt = COMFY / "output" / prefix / tag
        if tgt.exists():
            shutil.rmtree(tgt, ignore_errors=True)

    print(f"[{engine.upper()}] Submitting generation prompt to ComfyUI at {HOST}...", flush=True)
    t0 = time.time()
    try:
        res = req("/prompt", {"prompt": g})
    except Exception as e:
        print(f"[ERROR] Failed to post prompt to ComfyUI: {e}", flush=True)
        raise

    pid = res.get("prompt_id")
    if not pid:
        raise RuntimeError(f"ComfyUI rejected prompt: {res}")
    print(f"[{engine.upper()}] Prompt queued successfully: ID {pid}. Awaiting generation...", flush=True)

    # Poll execution until done
    while True:
        time.sleep(1.5)
        hist = req(f"/history/{pid}")
        if pid in hist:
            item = hist[pid]
            status = item.get("status", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"ComfyUI execution error: {msgs}")
            outputs = item.get("outputs", {})
            break

    dt = time.time() - t0
    print(f"[{engine.upper()}] Generation finished in {dt:.1f}s!", flush=True)

    # Locate generated output video and frames
    v_found = False
    for nid, out in outputs.items():
        if "gifs" in out:
            for ginfo in out["gifs"]:
                fn = ginfo.get("filename")
                src = COMFY / "output" / ginfo.get("subfolder", "") / fn
                if src.exists():
                    shutil.copy2(src, out_dir / "clip.mp4")
                    v_found = True
                    break

    if not v_found:
        # Fallback search in output/shot/
        cands = list((COMFY / "output" / "shot").glob(f"{tag}*.mp4"))
        if cands:
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            shutil.copy2(cands[0], out_dir / "clip.mp4")
            v_found = True

    if not (out_dir / "clip.mp4").exists():
        # First preference: collect exact images listed in prompt output history
        frames = []
        for nid, out in outputs.items():
            if "images" in out:
                for im in out["images"]:
                    sub = im.get("subfolder", "")
                    fn = im.get("filename")
                    src = COMFY / "output" / sub / fn if sub else COMFY / "output" / fn
                    if src.exists():
                        frames.append(src)
        if frames:
            frames.sort()
        else:
            # Fallback search in candidate folders
            candidates = [
                COMFY / "output" / f"{engine}_shot_png" / tag,
                COMFY / "output" / "wan_shot_png" / tag,
                COMFY / "output" / "shot_png" / tag,
                COMFY / "output" / "ltx_shot_png" / tag,
            ]
            for cdir in candidates:
                if cdir.exists():
                    f_list = sorted(cdir.glob("*.png"))
                    if f_list:
                        frames = f_list
                        break
        if frames:
            print(f"[{engine.upper()}] Stitching {len(frames)} frames into clip.mp4 with FFmpeg...", flush=True)
            lst = out_dir / "frames.txt"
            frame_dur = 1.0 / FPS
            lst.write_text(
                "".join(f"file '{f.as_posix()}'\nduration {frame_dur:.6f}\n" for f in frames)
                + f"file '{frames[-1].as_posix()}'\n",
                encoding="utf-8"
            )
            guide_audio = base / (spec.get("audio", {}).get("fixed", {}).get("file") or "guide.wav")
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
            if guide_audio.exists():
                cmd += ["-i", str(guide_audio), "-map", "0:v", "-map", "1:a"]
            else:
                cmd += ["-map", "0:v"]
            cmd += ["-vf", f"fps={FPS}", "-r", str(FPS), "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", str(out_dir / "clip.mp4")]
            subprocess.run(cmd, check=True)
            v_found = True

    print(f"[{engine.upper()}] Output saved to: {out_dir / 'clip.mp4'}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Multi-Engine Shot Generator (LTX / Wan 2.1 / HunyuanVideo / H3)")
    parser.add_argument("spec", type=str, help="Path to shot spec.json")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--engine", type=str, choices=["ltx", "wan2.1", "hunyuan", "h3"], default=None, help="Force active video engine")
    parser.add_argument("--dry-run", action="store_true", help="Compile workflow only without GPU execution")
    args = parser.parse_args()

    spec_p = Path(args.spec).resolve()
    out_d = Path(args.out).resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_p.read_text(encoding="utf-8"))
    engine = args.engine or spec.get("engine") or os.environ.get("VIDEO_ENGINE", "h3")

    if args.dry_run:
        prompt = build_prompt(spec, engine=engine)
        print(f"[dry-run] Engine: {engine}")
        print(f"[dry-run] Prompt: {prompt[:120]}...")
        return

    run_generation(spec_p, out_d, engine=engine)


if __name__ == "__main__":
    main()
