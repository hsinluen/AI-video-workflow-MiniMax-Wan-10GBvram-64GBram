import json
import shutil
from pathlib import Path

dst = Path(r"c:\20261018-demo測試-MiniMax_H3-Wan2.1\02_Wan2.1\demo2-physics")
dst.mkdir(parents=True, exist_ok=True)
src = Path(r"c:\2026-Antigravity_AI-Video_maker\pipeline\demo2_physics")

shutil.copy2(src / "make_film.py", dst / "make_film.py")
(dst / "out").mkdir(parents=True, exist_ok=True)
for sub in ["chars", "first", "music", "voice"]:
    shutil.copytree(src / "out" / sub, dst / "out" / sub, dirs_exist_ok=True)
shutil.copy2(src / "out" / "voice_picks.json", dst / "out" / "voice_picks.json")

film = (src / "film.json").read_text(encoding="utf-8")
data = json.loads(film)
data["engine"] = "wan2.1"
data["profile"] = "wan2.1_14b_i2v"
data["sampling"]["steps"] = 30
data["sampling"]["cfg"] = 6.0
data["sampling"]["size"] = [848, 480]
data["output_name"] = "physics_angular_momentum_wan2.1_1080p.mp4"
(dst / "film.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

shutil.copy2(src / "run_physics.py", dst / "run_physics.py")
shutil.copy2(src / "make_film.py", r"c:\20261018-demo測試-MiniMax_H3-Wan2.1\01_MiniMax-H3\demo2-physics\make_film.py")
shutil.copy2(src / "run_physics.py", r"c:\20261018-demo測試-MiniMax_H3-Wan2.1\01_MiniMax-H3\demo2-physics\run_physics.py")
shutil.copy2(src / "film.json", r"c:\20261018-demo測試-MiniMax_H3-Wan2.1\01_MiniMax-H3\demo2-physics\film.json")
(Path(r"c:\20261018-demo測試-MiniMax_H3-Wan2.1\01_MiniMax-H3\demo2-physics\out")).mkdir(parents=True, exist_ok=True)
for sub in ["chars", "first", "music", "voice"]:
    shutil.copytree(src / "out" / sub, Path(r"c:\20261018-demo測試-MiniMax_H3-Wan2.1\01_MiniMax-H3\demo2-physics\out") / sub, dirs_exist_ok=True)
shutil.copy2(src / "out" / "voice_picks.json", Path(r"c:\20261018-demo測試-MiniMax_H3-Wan2.1\01_MiniMax-H3\demo2-physics\out\voice_picks.json"))

print("Clean scaffold for Wan2.1 demo2-physics created successfully!")
