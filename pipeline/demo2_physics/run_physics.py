import os
import sys
import subprocess

here = os.path.dirname(os.path.abspath(__file__))
os.environ["FILM_PROJECT_DIR"] = here
os.environ["VIDEO_ENGINE"] = "h3"
os.environ["VIDEO_RES"] = "1080p"
py = r"C:\AI\H3\venv\Scripts\python.exe"

print(f"[RUNNER] Starting Physics Demo in {here} (Engine: h3)...", flush=True)
subprocess.run([py, "make_film.py", "h3"], cwd=here, check=True)

print("[RUNNER] Starting Physics Demo stage music...", flush=True)
subprocess.run([py, "make_film.py", "music"], cwd=here, check=True)

print("[RUNNER] Starting Physics Demo stage cut (1080p)...", flush=True)
subprocess.run([py, "make_film.py", "cut"], cwd=here, check=True)

print("[RUNNER] Scaling 1080p to 720p version...", flush=True)
import json
cfg_path = os.path.join(here, "film.json")
with open(cfg_path, encoding="utf-8") as f:
    cfg_data = json.load(f)
out_name = cfg_data.get("output_name", "physics_angular_momentum_h3_1080p_v2.mp4")
mp4_1080 = os.path.join(here, "out", "cut", out_name)
base_name, ext = os.path.splitext(out_name)
name_720 = base_name.replace("1080p", "720p") if "1080p" in base_name else f"{base_name}_720p"
mp4_720 = os.path.join(here, "out", "cut", f"{name_720}{ext}")
if os.path.exists(mp4_1080):
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", mp4_1080,
        "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264", "-crf", "14", "-preset", "slow",
        "-c:a", "copy", mp4_720
    ], check=True)
print("[RUNNER] All Physics Demo stages completed successfully!", flush=True)
