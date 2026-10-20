"""R3 產片腳本（動態與畫質平衡版，profile v4_r3_balanced_motion）
與 R2 的差別：
  1. 不再用「同一張首幀」當尾幀錨定（R2 的 LTXVAddGuide frame_idx=-1 會把最後一幀拉回首幀 → 人物幾乎不動）
  2. 幀數依鏡頭秒數 × 24fps 計算（8n+1，最多 97 幀），以真實 24fps 播放；R2 把 57 幀拉長到 4~5 秒 = 慢動作
  3. 首幀強度 0.88、cfg 3.0（官方建議值）、STG 只作用前 70% 步數
  4. QC 同時檢查「漂移/融化」與「過於靜止」兩個方向（R2 的 QC 只看漂移，等於獎勵靜止畫面）
用法：C:\\AI\\H3\\venv\\Scripts\\python.exe make_film.py [stage ...]   stage: voices h3 music cut
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
from PIL import Image

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_DIR = Path(os.environ.get("FILM_PROJECT_DIR", Path(__file__).resolve().parent)).resolve()
HERE = PROJECT_DIR
OUT = PROJECT_DIR / "out"
CFG = json.loads((PROJECT_DIR / "film.json").read_text(encoding="utf-8"))
CHARS, SHOTS = CFG["chars"], CFG["shots"]
S = CFG.get("sampling", {})
FPS = 24
MAX_FRAMES = int(S.get("max_frames", 97))
repo_tools = Path(__file__).resolve().parents[2] / "tools"
T = repo_tools if (repo_tools / "h3_shot.py").exists() else Path(os.environ.get("AI_TOOLS", r"C:\AI\tools"))
PYH3 = os.environ.get("PY_H3", r"C:\AI\H3\venv\Scripts\python.exe")
HOST = os.environ.get("COMFY_HOST", "http://127.0.0.1:8188")
TIMING = OUT / "timing.json"
timing = json.loads(TIMING.read_text(encoding="utf-8")) if TIMING.exists() else {}


def run(cmd, cwd=OUT, check=True):
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd))[:200]}\n{(r.stdout + r.stderr)[-1500:]}")
    return r


def dur(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def frames_for(d):
    """真實 24fps：秒數 × 24 取最近的 8n+1，上限 MAX_FRAMES。"""
    n = max(1, round((d * FPS - 1) / 8))
    return min(MAX_FRAMES, 8 * n + 1)


def stage(name):
    def deco(fn):
        def wrap():
            t0 = time.time(); fn(); timing[name] = round(time.time() - t0, 1)
            TIMING.write_text(json.dumps(timing, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[stage] {name} {timing[name]}s", flush=True)
        return wrap
    return deco


def line_files():
    picks = json.loads((OUT / "voice_picks.json").read_text(encoding="utf-8"))
    files, li = {}, 0
    for s in SHOTS:
        if s.get("line"):
            files[s["id"]] = Path(picks[str(li)]["file"]); li += 1
    return files


@stage("voices")
def stage_voices():
    vp = OUT / "voice_picks.json"
    picks = json.loads(vp.read_text(encoding="utf-8"))
    for v in picks.values():
        local = OUT / "voice" / "bv" / Path(v["file"]).name
        if local.exists():
            v["file"] = str(local.resolve())
    vp.write_text(json.dumps(picks, ensure_ascii=False, indent=2), encoding="utf-8")
    for d in (HERE / "params", OUT / "params"):
        d.mkdir(parents=True, exist_ok=True)
        for pf in (HERE.parent.parent / "configs" / "params_profiles").glob("*.json"):
            shutil.copy2(pf, d / pf.name)
    print("[voices] local voice paths fixed; parameter profiles archived.", flush=True)


@stage("h3")
def stage_h3():
    lf = line_files()
    only = os.environ.get("ONLY", "").split()
    for s in SHOTS:
        if only and s["id"] not in only:
            continue
        od = OUT / "h3" / s["id"]; od.mkdir(parents=True, exist_ok=True)
        if (od / "clip.mp4").exists():
            print(f"[h3] {s['id']} exists, skip", flush=True); continue
        engine = CFG.get("engine", os.environ.get("VIDEO_ENGINE", "h3")).lower()
        if engine in ("h3", "minimax"):
            frames = max(5, int(float(s["dur"]) * FPS))
            while frames % 17 != 5:
                frames += 1
            length = frames
        elif engine in ("wan", "wan2.1"):
            length = int(4 * max(1, round((float(s["dur"]) * FPS - 1) / 4)) + 1)
        else:
            length = frames_for(float(s["dur"]))
        secs = length / FPS
        sr = 48000; buf = bytearray(int(secs * sr) * 2); dlg = []
        if s.get("line"):
            ln = s["line"]; at = float(ln.get("at", 0.3))
            wav_d = dur(lf[s["id"]])
            if at + wav_d > secs:
                print(f"[warn] {s['id']} line {wav_d:.2f}s + at {at}s > shot {secs:.2f}s (tail will be cut)", flush=True)
            pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(lf[s["id"]]), "-ar", str(sr), "-ac", "1", "-f", "s16le", "-"],
                                 capture_output=True).stdout
            st = int(at * sr) * 2; seg = pcm[: len(buf) - st]; buf[st:st + len(seg)] = seg
            dlg = [dict(subject=ln["who"], line=ln["text"], lang="Chinese", after=ln.get("after", "After the line the speaker's lips close naturally."))]
        with wave.open(str(od / "guide.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(bytes(buf))
        default_size = [1152, 640] if engine in ("h3", "minimax") else [848, 480]
        spec = dict(
            duration=round(secs, 3), length=length, size=S.get("size", default_size),
            seed=int(s.get("seed", CFG.get("base_seed", 20261040) + int(s["id"][1:]))),
            steps=int(S.get("steps", 8 if engine in ("h3", "minimax") else 30)), cfg=float(S.get("cfg", 4.0 if engine in ("h3", "minimax") else 6.0)),
            scheduler=S.get("scheduler", "simple"), sampler_name=S.get("sampler_name", "res_multistep" if engine in ("h3", "minimax") else "euler"),
            max_shift=float(S.get("max_shift", 2.05)), base_shift=float(S.get("base_shift", 0.95)), terminal=float(S.get("terminal", 0.1)),
            stg=bool(S.get("stg", True)), stg_scale=float(S.get("stg_scale", 1.0)), stg_blocks=str(S.get("stg_blocks", "14")),
            stg_start_percent=float(S.get("stg_start_percent", 0.0)), stg_end_percent=float(S.get("stg_end_percent", 0.70)),
            img_compression=int(s.get("img_compression", S.get("img_compression", 30))),
            img_strength=float(s.get("img_strength", S.get("img_strength", 0.88))),
            end_anchor_strength=float(s.get("end_anchor_strength", S.get("end_anchor_strength", 0.0))),
            neg_prompt=S.get("neg_prompt"),
            style=CFG["h3"]["style"], setting=CFG["h3"]["setting"],
            first_frame=str((OUT / "first" / f"{s['id']}.png").resolve()),
            subjects=[dict(key=k, desc=CHARS[k]["look"]) for k in s["refs"]],
            audio={"fixed": {"file": str((od / "guide.wav").resolve())}},
            shots=[dict(camera=s.get("camera", CFG["h3"].get("camera", "Static tripod shot.")), action=s["action"], dialogue=dlg)],
        )
        spec["engine"] = engine
        spec["crf"] = int(S.get("crf", 12))
        (od / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{engine.upper()}] {s['id']}: {length} frames ({secs:.2f}s realtime), seed {spec['seed']}", flush=True)
        run([PYH3, T / "h3_shot.py", od / "spec.json", "--out", od, "--engine", engine])
    review()


def review():
    """每鏡 5 點抽幀 + 量化：drift（對首幀差異，太大=變形/漂移）、motion（相鄰抽樣幀差異，太小=靜止）、清晰度保持率。"""
    rv = OUT / "review"; rv.mkdir(exist_ok=True); res = {}
    for s in SHOTS:
        c = OUT / "h3" / s["id"] / "clip.mp4"
        if not c.exists():
            continue
        d = dur(c); arrs, tiles = [], []
        for i, q in enumerate((0.03, 0.27, 0.51, 0.75, 0.97)):
            raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{d * q:.2f}", "-i", str(c), "-s", "768x512", "-frames:v", "1",
                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
            im = Image.frombytes("RGB", (768, 512), raw[: 768 * 512 * 3]); tiles.append(im)
            arrs.append(np.asarray(im, dtype=np.float32))
        strip = Image.new("RGB", (768 * 5, 512))
        for i, t in enumerate(tiles):
            strip.paste(t, (768 * i, 0))
        strip.save(rv / f"strip_{s['id']}.jpg", quality=90)

        def sharp(a):
            gy, gx = np.gradient(a.mean(axis=2)); return float(np.mean(gx ** 2 + gy ** 2))
        drift = [round(float(np.abs(a - arrs[0]).mean()), 2) for a in arrs]
        motion = [round(float(np.abs(arrs[i + 1] - arrs[i]).mean()), 2) for i in range(4)]
        sr = round(sharp(arrs[-1]) / max(1e-5, sharp(arrs[0])), 3)
        too_static = max(drift) < float(S.get("qc_min_motion", 3.5))
        too_drift = max(drift) > float(S.get("qc_max_drift", 22.0)) or sr < 0.70
        res[s["id"]] = dict(drift_from_first=drift, motion_between_samples=motion, max_drift=max(drift),
                            sharpness_retention=sr, too_static=too_static, too_drift=too_drift,
                            verdict="OK" if not (too_static or too_drift) else ("TOO_STATIC" if too_static else "DRIFT/BLUR"))
        print(f"[review] {s['id']}: max_drift={max(drift)} motion={motion} sharp={sr} -> {res[s['id']]['verdict']}", flush=True)
    (rv / "motion_qc.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


@stage("music")
def stage_music():
    m = OUT / "music" / CFG["music_file"]
    if not m.exists():
        raise SystemExit(f"Missing music {m}")
    print(f"[music] {m.name} ok", flush=True)


@stage("cut")
def stage_cut():
    target_res = CFG.get("resolution", os.environ.get("VIDEO_RES", "1080p")).lower()
    target_w, target_h = (1280, 720) if target_res == "720p" else (1920, 1080)
    font_size = 16 if target_res == "720p" else 22
    print(f"[cut] Target video resolution: {target_res.upper()} ({target_w}x{target_h})", flush=True)

    (OUT / "cut").mkdir(exist_ok=True); (OUT / "vsr").mkdir(exist_ok=True)
    lf = line_files(); parts, t, subs = [], 0.0, []
    suffix = "_720p" if target_res == "720p" else ""
    for s in SHOTS:
        v = OUT / "vsr" / f"{s['id']}{suffix}.mp4"
        if not v.exists():
            run([PYH3, T / "upscale_vsr.py", "--source", OUT / "h3" / s["id"] / "clip.mp4", "--output", v, "--res", target_res])
        d = dur(v)
        if s.get("line"):
            at = float(s["line"].get("at", 0.3)); wd = dur(lf[s["id"]])
            subs.append((t + at, min(t + d - 0.05, t + at + wd + 0.3), f"{CHARS[s['line']['who']]['name']}：{s['line']['text']}"))
        parts.append(v); t += d
    (OUT / "cut" / "list.txt").write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "cut/list.txt", "-c", "copy", "cut/joined.mp4"])

    def ts(x):
        return f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{x % 60:06.3f}".replace(".", ",")
    (OUT / "cut" / "subs.srt").write_text("".join(f"{i}\n{ts(a)} --> {ts(b)}\n{txt}\n\n" for i, (a, b, txt) in enumerate(subs, 1)), encoding="utf-8")
    out_mp4 = OUT / "cut" / CFG["output_name"]
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", "cut/joined.mp4", "-i", OUT / "music" / CFG["music_file"], "-filter_complex",
         f"[1:a]volume=0.18,afade=t=out:st={t - 2:.2f}:d=2[m];[0:a][m]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5[a];"
         f"[0:v]subtitles=cut/subs.srt:force_style='FontName=Microsoft JhengHei,FontSize={font_size},Outline=2'[v]",
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-c:a", "aac", "-b:a", "192k", out_mp4])
    probe = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,codec_name",
                            "-of", "json", out_mp4]).stdout)["streams"][0]
    qc = review()
    report = dict(status="PASS" if probe["width"] == target_w and probe["height"] == target_h and all(v["verdict"] == "OK" for v in qc.values()) else "CHECK",
                  resolution=f"{probe['width']}x{probe['height']}", duration=round(dur(out_mp4), 2), profile=CFG.get("profile"),
                  shots=qc, sampling=S)
    (OUT / "cut" / "qc_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_mp4} {report['duration']}s QC={report['status']} ({report['resolution']})", flush=True)


STAGES = dict(voices=stage_voices, h3=stage_h3, music=stage_music, cut=stage_cut)

if __name__ == "__main__":
    try:
        urllib.request.urlopen(HOST + "/system_stats", timeout=5)
    except Exception:
        sys.exit(f"ComfyUI not running at {HOST}")
    for name in (sys.argv[1:] or list(STAGES)):
        STAGES[name]()
    print("timing", json.dumps(timing, ensure_ascii=False))
