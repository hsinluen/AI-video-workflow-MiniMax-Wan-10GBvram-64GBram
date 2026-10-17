"""LTX-Video shot generator for 10GB VRAM (refactored from h3_shot.py).

Replaces MiniMax-H3 with Lightricks LTX-Video (INT8/FP16/FP8 quantized),
tailored for NVIDIA RTX 3080 (10GB VRAM) + 64GB system RAM.
Generates video clips via ComfyUI API using LTXVideoImageToVideo / LTXVImgToVideo,
properly aligning first frame conditioning and dialog/soundtrack audio guidance.

Usage:
  python ltx_shot.py spec.json --out <dir>
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

HOST = os.environ.get("COMFY_HOST", "http://127.0.0.1:8188")
COMFY = Path(os.environ.get("COMFY_DIR", r"C:\AI\H3\ComfyUI-0.36.0"))
FPS = 24
DEFAULT_MODEL = os.environ.get("LTX_MODEL", "ltx-video-2b-cfgquantized.sft")
DEFAULT_VAE = os.environ.get("LTX_VAE", "ltx-video-vae.safetensors")


def frames_for_ltx(duration):
    """LTX-Video uses temporal downsample of 8, so frames = 8n + 1."""
    target = max(9, int(round(duration * FPS)))
    n = max(1, round((target - 1) / 8))
    return int(8 * n + 1)


def snap32(dim):
    """Snap dimension to nearest multiple of 32."""
    return max(64, int(round(dim / 32.0) * 32))


def req(path, data=None):
    r = urllib.request.Request(
        f"{HOST}{path}",
        data=json.dumps(data).encode("utf-8") if data else None,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stage_file(src, name):
    dest = COMFY / "input" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return name


def build_motion_prompt(spec):
    """Synthesize action and cinematics into an expressive LTX motion prompt."""
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
            parts.append(f"{subj} speaks line '{line}'. {after}".strip())

    if not parts:
        parts.append("Cinematic live action video, high quality motion, natural expressions and lighting.")

    return ". ".join(p.strip() for p in parts if p.strip()) + "."


def build_graph(spec, prompt, base, tag):
    w, h = spec.get("size", [768, 512])
    # Ensure divisible by 32
    w = snap32(w)
    h = snap32(h)
    # 10GB VRAM safety: cap width/height to 768x512 or 896x512 if too large
    if w * h > 1024 * 576:
        w, h = 768, 512

    duration = float(spec.get("duration", 3.0))
    fps_gen = float(spec.get("fps_gen", FPS))
    length = int(spec.get("length", 0)) or frames_for_ltx(duration)
    seed = int(spec.get("seed", 42))
    steps = int(spec.get("steps", 35))
    cfg = float(spec.get("cfg", 2.5))
    sampler_name = str(spec.get("sampler_name", "euler"))
    scheduler = str(spec.get("scheduler", "ltxv"))
    stg_enabled = bool(spec.get("stg", True))
    stg_scale = float(spec.get("stg_scale", 1.0))
    stg_blocks = str(spec.get("stg_blocks", "14"))
    stg_start = float(spec.get("stg_start_percent", 0.0))
    stg_end = float(spec.get("stg_end_percent", 0.75))
    img_compression = int(spec.get("img_compression", 28))
    img_strength = float(spec.get("img_strength", 0.90))
    max_shift = float(spec.get("max_shift", 2.05))
    base_shift = float(spec.get("base_shift", 0.95))
    terminal = float(spec.get("terminal", 0.1))
    end_anchor_strength = float(spec.get("end_anchor_strength", 0.0))

    first_frame_path = spec.get("first_frame")
    if not first_frame_path:
        raise ValueError("spec.json must specify first_frame for image-to-video generation")

    staged_first = stage_file(base / first_frame_path, f"{tag}_first.png")

    # Audio guide handling
    audio_info = spec.get("audio", {})
    audio_file = None
    if audio_info.get("fixed", {}).get("file"):
        audio_file = stage_file(base / audio_info["fixed"]["file"], f"{tag}_audio.wav")
    elif (base / "guide.wav").exists():
        audio_file = stage_file(base / "guide.wav", f"{tag}_audio.wav")

    model_name = spec.get("model", DEFAULT_MODEL)
    vae_name = spec.get("vae", DEFAULT_VAE)
    clip_name = spec.get("clip", os.environ.get("LTX_CLIP", "t5xxl_fp8_e4m3fn.safetensors"))
    neg_prompt = spec.get(
        "neg_prompt",
        "low quality, worst quality, deformed, distorted face, melting face, morphing, extra limbs, ghosting, motion blur, blurry, static, jitter, artifacts, watermark, text"
    )

    # ComfyUI Workflow API Graph
    g = {
        # Diffusion Transformer (UNETLoader)
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": model_name,
                "weight_dtype": "default"
            }
        },
        # Load First Frame Image
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": staged_first}
        },
        # VAE Loader
        "20": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name}
        },
        # CLIP / T5 Text Encoder Loader
        "21": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": "ltxv",
                "device": "default"
            }
        },
        # Positive Conditioning (CLIP)
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["21", 0], "text": prompt}
        },
        # Negative Conditioning (CLIP)
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["21", 0], "text": neg_prompt}
        },
        # LTX-Video Conditioning (sets frame rate)
        "22": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["3", 0],
                "negative": ["4", 0],
                "frame_rate": fps_gen
            }
        }
    }

    # Optional H.264 CRF Preprocessing on First Frame (matches LTX training distribution)
    first_img_node = ["2", 0]
    if img_compression > 0:
        g["23"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {
                "image": ["2", 0],
                "img_compression": img_compression
            }
        }
        first_img_node = ["23", 0]

    # LTX-Video Image To Video Conditioning
    g["5"] = {
        "class_type": "LTXVImgToVideo",
        "inputs": {
            "positive": ["22", 0],
            "negative": ["22", 1],
            "vae": ["20", 0],
            "image": first_img_node,
            "width": w,
            "height": h,
            "length": length,
            "batch_size": 1,
            "strength": img_strength
        }
    }

    cond_pos = ["5", 0]
    cond_neg = ["5", 1]
    cond_lat = ["5", 2]

    # Optional End-Frame Guide Anchor (prevents late-shot facial drift)
    if end_anchor_strength > 0.0:
        g["28"] = {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive": cond_pos,
                "negative": cond_neg,
                "vae": ["20", 0],
                "latent": cond_lat,
                "image": first_img_node,
                "frame_idx": -1,
                "strength": end_anchor_strength
            }
        }
        cond_pos = ["28", 0]
        cond_neg = ["28", 1]
        cond_lat = ["28", 2]

    if scheduler == "ltxv":
        # Official Lightricks Token-Shifted ModelSamplingLTXV
        g["24"] = {
            "class_type": "ModelSamplingLTXV",
            "inputs": {
                "model": ["1", 0],
                "max_shift": max_shift,
                "base_shift": base_shift,
                "latent": cond_lat
            }
        }
        model_to_sampler = ["24", 0]

        if stg_enabled:
            g["25"] = {
                "class_type": "LTXVSpatioTemporalGuidance",
                "inputs": {
                    "model": model_to_sampler,
                    "scale": stg_scale,
                    "blocks": stg_blocks,
                    "start_percent": stg_start,
                    "end_percent": stg_end
                }
            }
            model_to_sampler = ["25", 0]

        g["26"] = {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": steps,
                "max_shift": max_shift,
                "base_shift": base_shift,
                "stretch": True,
                "terminal": terminal,
                "latent": cond_lat
            }
        }
        g["27"] = {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": sampler_name
            }
        }
        g["6"] = {
            "class_type": "SamplerCustom",
            "inputs": {
                "model": model_to_sampler,
                "add_noise": True,
                "noise_seed": seed,
                "cfg": cfg,
                "positive": cond_pos,
                "negative": cond_neg,
                "sampler": ["27", 0],
                "sigmas": ["26", 0],
                "latent_image": cond_lat
            }
        }
    else:
        # Legacy KSampler path (for v1_fast profile restore)
        model_to_sampler = ["1", 0]
        if stg_enabled:
            g["25"] = {
                "class_type": "LTXVSpatioTemporalGuidance",
                "inputs": {
                    "model": ["1", 0],
                    "scale": stg_scale,
                    "blocks": stg_blocks,
                    "start_percent": stg_start,
                    "end_percent": stg_end
                }
            }
            model_to_sampler = ["25", 0]
        g["6"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model_to_sampler,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": cond_pos,
                "negative": cond_neg,
                "latent_image": cond_lat,
                "denoise": 1.0
            }
        }

    sampled_latent = ["6", 0]
    if end_anchor_strength > 0.0:
        g["29"] = {
            "class_type": "LTXVCropGuides",
            "inputs": {
                "positive": cond_pos,
                "negative": cond_neg,
                "latent": ["6", 0]
            }
        }
        sampled_latent = ["29", 2]

    # VAE Decode
    g["7"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": sampled_latent,
            "vae": ["20", 0]
        }
    }

    # Save Frames (PNG sequence)
    g["16"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["7", 0],
            "filename_prefix": f"ltx_shot_png/{tag}/f"
        }
    }

    if audio_file:
        g["8"] = {
            "class_type": "LoadAudio",
            "inputs": {"audio": audio_file}
        }
        g["9"] = {
            "class_type": "CreateVideo",
            "inputs": {"images": ["7", 0], "audio": ["8", 0], "fps": float(FPS)}
        }
    else:
        g["9"] = {
            "class_type": "CreateVideo",
            "inputs": {"images": ["7", 0], "fps": float(FPS)}
        }

    g["15"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["9", 0],
            "filename_prefix": f"ltx_shot/{tag}",
            "format": "mp4"
        }
    }

    return g, length


def generate(spec, prompt, base, out):
    tag = f"{out.name}_{int(time.time())}"
    g, length = build_graph(spec, prompt, base, tag)
    (out / "workflow_api.json").write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
    t0 = time.time()

    prompt_resp = req("/prompt", {"prompt": g})
    pid = prompt_resp["prompt_id"]
    print(f"[queued] {tag} {length} frames, seed={spec.get('seed', 42)}, prompt_id={pid}", flush=True)

    while True:
        time.sleep(3)
        h = req(f"/history/{pid}")
        if pid in h:
            break

    entry = h[pid]
    ok = entry.get("status", {}).get("status_str") == "success"

    if ok:
        raw = out / "clip_savevideo.mp4"
        outputs = entry.get("outputs", {})
        if "15" in outputs:
            v_list = outputs["15"].get("videos") or outputs["15"].get("images") or []
            if v_list:
                v = v_list[0]
                src_vid = COMFY / "output" / v.get("subfolder", "") / v["filename"]
                if src_vid.exists():
                    shutil.copy2(src_vid, raw)

        # Assemble clean MP4 using ffmpeg from decoded PNG frames
        if "16" in outputs and outputs["16"].get("images"):
            frames = [COMFY / "output" / "h3_shot_png" / tag / im["filename"] if (COMFY / "output" / "h3_shot_png" / tag / im["filename"]).exists()
                      else COMFY / "output" / "ltx_shot_png" / tag / im["filename"]
                      for im in outputs["16"]["images"]]
            frames = [f for f in frames if f.exists()]
            if frames:
                # Real-time playback by default (1 generated frame = 1/24 s).
                # R2 stretched 57 frames over the whole shot duration, which played as slow motion.
                if spec.get("stretch_to_duration", False):
                    target_dur = float(spec.get("duration", len(frames) / float(FPS)))
                    frame_dur = target_dur / max(1, len(frames))
                else:
                    frame_dur = 1.0 / FPS
                lst = out / "frames.txt"
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
                        "-c:a", "aac", "-b:a", "192k", str(out / "clip.mp4")]
                subprocess.run(cmd, check=True)
            elif raw.exists():
                shutil.copy2(raw, out / "clip.mp4")
        elif raw.exists():
            shutil.copy2(raw, out / "clip.mp4")
    else:
        print(f"[LTX FAIL] {json.dumps(entry.get('status', {}), ensure_ascii=False)}")

    meta = {
        "ok": ok,
        "length": length,
        "seed": spec.get("seed", 42),
        "wall_s": round(time.time() - t0, 1),
        "tag": tag,
        "model": "LTX-Video-2B"
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    params_record = {
        "tag": tag,
        "steps": int(spec.get("steps", 35)),
        "cfg": float(spec.get("cfg", 2.5)),
        "stg": bool(spec.get("stg", True)),
        "stg_scale": float(spec.get("stg_scale", 1.0)),
        "stg_blocks": str(spec.get("stg_blocks", "14")),
        "stg_start_percent": float(spec.get("stg_start_percent", 0.0)),
        "stg_end_percent": float(spec.get("stg_end_percent", 0.75)),
        "sampler_name": str(spec.get("sampler_name", "euler")),
        "scheduler": str(spec.get("scheduler", "ltxv")),
        "img_compression": int(spec.get("img_compression", 28)),
        "img_strength": float(spec.get("img_strength", 0.90)),
        "max_shift": float(spec.get("max_shift", 2.05)),
        "base_shift": float(spec.get("base_shift", 0.95)),
        "terminal": float(spec.get("terminal", 0.1)),
        "end_anchor_strength": float(spec.get("end_anchor_strength", 0.0)),
        "seed": int(spec.get("seed", 42)),
        "size": spec.get("size", [768, 512]),
        "length": length
    }
    (out / "shot_params.json").write_text(json.dumps(params_record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{'ok' if ok else 'FAIL'}] {out / 'clip.mp4'} in {meta['wall_s']}s (steps={params_record['steps']}, cfg={params_record['cfg']}, scheduler={params_record['scheduler']}, stg_blocks={params_record['stg_blocks']})", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser(description="LTX-Video Shot Generator for 10GB VRAM")
    parser.add_argument("spec", help="Path to shot spec.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Build workflow and prompt without GPU execution")
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    prompt = build_motion_prompt(spec)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    if args.dry_run:
        g, length = build_graph(spec, prompt, spec_path.parent, out_dir.name)
        (out_dir / "workflow_api.json").write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[dry-run] prompt:\n{prompt}\n[dry-run] graph written to {out_dir / 'workflow_api.json'}")
        return

    generate(spec, prompt, spec_path.parent, out_dir)


if __name__ == "__main__":
    main()
