import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
py = r"C:\AI\H3\venv\Scripts\python.exe"
artifacts_dir = Path(r"C:\Users\hsinluen\.gemini\antigravity\brain\c9c0f7be-5137-4665-9e94-527d13dfe687")
dest_demo9 = Path(r"c:\20261018-demo測試-MiniMax_H3-Wan2.1\01_MiniMax-H3\demo9-fluid")
github_repo = Path(r"c:\20261018-Github-repo-AI-video-workflow\AI-video-workflow-MiniMax-Wan-10GBvram-64GBram")

os.environ["FILM_PROJECT_DIR"] = str(here)
os.environ["VIDEO_ENGINE"] = "h3"
os.environ["VIDEO_RES"] = "1080p"

print("=" * 80, flush=True)
print("[DEMO 9 FLUID MECHANICS] Starting 5-Shot Dynamic Video Production", flush=True)
print("=" * 80, flush=True)

t0 = time.time()

# 1. Voices setup
print("[1/5] Executing make_film.py voices...", flush=True)
subprocess.run([py, "make_film.py", "voices"], cwd=str(here), check=True)

# 2. H3 Diffusion (5 shots)
print("[2/5] Executing make_film.py h3 (S1 ~ S5)...", flush=True)
subprocess.run([py, "make_film.py", "h3"], cwd=str(here), check=True)

# 3. Music stage
print("[3/5] Executing make_film.py music...", flush=True)
subprocess.run([py, "make_film.py", "music"], cwd=str(here), check=True)

# 4. Cut stage (1080p VSR, subtitles, ducking)
print("[4/5] Executing make_film.py cut (1080p)...", flush=True)
subprocess.run([py, "make_film.py", "cut"], cwd=str(here), check=True)

# 5. Produce 720p version
print("[5/5] Producing 720p downscaled version...", flush=True)
out_1080 = here / "out" / "cut" / "fluid_bernoulli_h3_1080p.mp4"
out_720 = here / "out" / "cut" / "fluid_bernoulli_h3_720p.mp4"
if out_1080.exists():
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(out_1080),
        "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264", "-crf", "14", "-preset", "slow",
        "-c:a", "copy", str(out_720)
    ], check=True)

total_elapsed = time.time() - t0
size_1080 = out_1080.stat().st_size / (1024 * 1024) if out_1080.exists() else 0.0
size_720 = out_720.stat().st_size / (1024 * 1024) if out_720.exists() else 0.0

print(f"[DEMO 9] Render completed in {total_elapsed/60:.2f} min ({total_elapsed:.1f}s)!", flush=True)
print(f"[DEMO 9] 1080p: {out_1080.name} ({size_1080:.2f} MB)", flush=True)
print(f"[DEMO 9] 720p:  {out_720.name} ({size_720:.2f} MB)", flush=True)

# Mirror to 01_MiniMax-H3 and Artifacts
dest_demo9.mkdir(parents=True, exist_ok=True)
shutil.copy2(out_1080, dest_demo9 / "fluid_bernoulli_h3_1080p.mp4")
if out_720.exists():
    shutil.copy2(out_720, dest_demo9 / "fluid_bernoulli_h3_720p.mp4")
if (here / "out" / "cut" / "qc_report.json").exists():
    shutil.copy2(here / "out" / "cut" / "qc_report.json", dest_demo9 / "qc_report.json")
if (here / "out" / "timing.json").exists():
    shutil.copy2(here / "out" / "timing.json", dest_demo9 / "timing.json")
for strip in (here / "out" / "review").glob("strip_*.jpg"):
    shutil.copy2(strip, dest_demo9 / strip.name)
    try:
        shutil.copy2(strip, artifacts_dir / f"demo9_{strip.name}")
    except Exception:
        pass

try:
    shutil.copy2(out_1080, artifacts_dir / "fluid_bernoulli_h3_1080p.mp4")
    if out_720.exists():
        shutil.copy2(out_720, artifacts_dir / "fluid_bernoulli_h3_720p.mp4")
    print("[DEMO 9] Mirrored to artifacts successfully!", flush=True)
except Exception as e:
    print(f"Warning copying to artifacts: {e}", flush=True)

# GitHub sync and push
print("\n[GITHUB] Synchronizing Demo 9 to GitHub...", flush=True)
git_demo9 = github_repo / "pipeline" / "demo9_fluid"
git_demo9.mkdir(parents=True, exist_ok=True)
(git_demo9 / "out" / "first").mkdir(parents=True, exist_ok=True)
(git_demo9 / "out" / "chars").mkdir(parents=True, exist_ok=True)
(git_demo9 / "out" / "music").mkdir(parents=True, exist_ok=True)
(git_demo9 / "out" / "voice" / "bv").mkdir(parents=True, exist_ok=True)

for pyf in ["make_film.py", "run_fluid.py"]:
    if (here / pyf).exists():
        shutil.copy2(here / pyf, git_demo9 / pyf)

for img in (here / "out" / "first").glob("*.png"):
    shutil.copy2(img, git_demo9 / "out" / "first" / img.name)
for img in (here / "out" / "chars").glob("*.png"):
    shutil.copy2(img, git_demo9 / "out" / "chars" / img.name)
if (here / "out" / "music" / "campus_Instrumental.wav").exists():
    shutil.copy2(here / "out" / "music" / "campus_Instrumental.wav", git_demo9 / "out" / "music" / "campus_Instrumental.wav")

# Anonymize film.json
cfg = json.loads((here / "film.json").read_text(encoding="utf-8"))
if "chars" in cfg and "tsai" in cfg["chars"]:
    cfg["chars"]["tsai"]["name"] = "專任教師"
    cfg["chars"]["tsai"]["look"] = cfg["chars"]["tsai"]["look"].replace("Prof. Tsai (蔡欣倫 教授)", "A dedicated male professor in his early fifties").replace("Prof. Tsai", "The professor")
    cfg["chars"]["tsai"]["voice_design"] = "A dedicated university professor voice design (Users can replace with their own cloned voice model or standard TTS neural voice)."
    cfg["chars"]["tsai"]["edge_voice"] = "Custom-Cloned (or zh-TW-YunJheNeural fallback)"
(git_demo9 / "film.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

# Anonymize voice_picks.json
vp_src = here / "out" / "voice_picks.json"
if vp_src.exists():
    picks = json.loads(vp_src.read_text(encoding="utf-8"))
    for v in picks.values():
        if v.get("voice") == "Tsai-Cloned":
            v["voice"] = "Custom-Cloned (專任教師音色，使用者可自由替換為自有模型)"
        v["file"] = f"out/voice/bv/{Path(v['file']).name}"
    (git_demo9 / "out" / "voice_picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=2), encoding="utf-8")

# Append to README.md
readme_git = github_repo / "README.md"
if readme_git.exists():
    content = readme_git.read_text(encoding="utf-8")
    if "Demo 9" not in content:
        entry_md = "| **Demo 9** | 《流體力學》 | 白努利定律與昇力原理 | 白努利定律與雨傘開花昇力大作戰 | 專任教師 & 工學院學生阿宇 |\n"
        content = content.replace("| **Demo 8** |", entry_md + "| **Demo 8** |")
        readme_git.write_text(content, encoding="utf-8")

subprocess.run(["git", "add", "-A"], cwd=str(github_repo), check=True)
subprocess.run(["git", "commit", "-m", "feat: Add Demo 9 (Fluid Mechanics Bernoulli 5-shot) university course motivation demo"], cwd=str(github_repo), check=True)
subprocess.run(["git", "push", "origin", "main"], cwd=str(github_repo), check=True)
print("[GITHUB] Demo 9 pushed to GitHub successfully!", flush=True)

print("=" * 80, flush=True)
print("[DEMO 9 FLUID MECHANICS] All Operations Finished Successfully!", flush=True)
print("=" * 80, flush=True)
