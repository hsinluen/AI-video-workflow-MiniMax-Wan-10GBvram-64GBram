import os
import shutil
from pathlib import Path

src_cut = Path(r"c:\2026-Antigravity_AI-Video_maker\pipeline\demo2_physics\out\cut")
src_review = Path(r"c:\2026-Antigravity_AI-Video_maker\pipeline\demo2_physics\out\review")

# 1. Comparison folder
dst_demo2_h3 = Path(r"c:\20261018-demo測試-MiniMax_H3-Wan2.1\01_MiniMax-H3\demo2-physics")
dst_demo2_h3.mkdir(parents=True, exist_ok=True)
(dst_demo2_h3 / "review").mkdir(parents=True, exist_ok=True)

shutil.copy2(src_cut / "physics_angular_momentum_h3_1080p.mp4", dst_demo2_h3 / "physics_angular_momentum_h3_1080p.mp4")
shutil.copy2(src_cut / "physics_angular_momentum_h3_720p.mp4", dst_demo2_h3 / "physics_angular_momentum_h3_720p.mp4")
shutil.copy2(src_cut / "qc_report.json", dst_demo2_h3 / "qc_report.json")

for strip in src_review.glob("strip_*.jpg"):
    shutil.copy2(strip, dst_demo2_h3 / "review" / strip.name)
print("Copied files to 01_MiniMax-H3/demo2-physics")

# 2. Artifact directory
art_dir = Path(r"C:\Users\hsinluen\.gemini\antigravity\brain\c9c0f7be-5137-4665-9e94-527d13dfe687")
shutil.copy2(src_cut / "physics_angular_momentum_h3_1080p.mp4", art_dir / "physics_angular_momentum_h3_1080p.mp4")
shutil.copy2(src_cut / "physics_angular_momentum_h3_720p.mp4", art_dir / "physics_angular_momentum_h3_720p.mp4")

for i in range(1, 5):
    s = src_review / f"strip_S{i}.jpg"
    if s.exists():
        shutil.copy2(s, art_dir / f"demo2_strip_S{i}.jpg")
print("Copied files to Artifact directory")

# 3. Sync to GitHub repo folder
repo_demo2 = Path(r"c:\20261018-Github-repo-AI-video-workflow\AI-video-workflow-MiniMax-Wan-10GBvram-64GBram\pipeline\demo2_physics")
if repo_demo2.exists():
    shutil.copy2(r"c:\2026-Antigravity_AI-Video_maker\pipeline\demo2_physics\make_film.py", repo_demo2 / "make_film.py")
    shutil.copy2(r"c:\2026-Antigravity_AI-Video_maker\pipeline\demo2_physics\run_physics.py", repo_demo2 / "run_physics.py")
    shutil.copy2(r"c:\2026-Antigravity_AI-Video_maker\pipeline\demo2_physics\film.json", repo_demo2 / "film.json")
    print("Synced updated scripts to repo folder")

print("All copies and sync completed successfully!")
