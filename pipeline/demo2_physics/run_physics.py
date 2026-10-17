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
mp4_1080 = os.path.join(here, "out", "cut", "physics_angular_momentum_h3_1080p.mp4")
mp4_720 = os.path.join(here, "out", "cut", "physics_angular_momentum_h3_720p.mp4")
if os.path.exists(mp4_1080):
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", mp4_1080,
        "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-c:a", "copy", mp4_720
    ], check=True)
print("[RUNNER] All Physics Demo stages completed successfully!", flush=True)
