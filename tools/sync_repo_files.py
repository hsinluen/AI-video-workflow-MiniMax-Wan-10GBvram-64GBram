import os
import shutil
from pathlib import Path

src = Path(r"c:\2026-Antigravity_AI-Video_maker")
dst = Path(r"c:\20261018-Github-repo-AI-video-workflow\AI-video-workflow-MiniMax-Wan-10GBvram-64GBram")

dst.mkdir(parents=True, exist_ok=True)

# 1. Configs
if (src / "configs").exists():
    shutil.copytree(src / "configs", dst / "configs", dirs_exist_ok=True)
    print("Copied configs")

# 2. Tools
if (src / "tools").exists():
    shutil.copytree(src / "tools", dst / "tools", dirs_exist_ok=True)
    print("Copied tools")

# 3. Templates
if (src / "templates").exists():
    shutil.copytree(src / "templates", dst / "templates", dirs_exist_ok=True)
    print("Copied templates")

# 4. Pipeline demo_local (Demo 1)
def ignore_heavy(dir_path, names):
    ignored = set()
    for name in names:
        if name in ("h3", "wav", "clips", "__pycache__"):
            ignored.add(name)
        elif name.endswith(".mp4") and "cut" not in dir_path:
            ignored.add(name)
    return ignored

shutil.copytree(src / "pipeline" / "demo_local", dst / "pipeline" / "demo_local", ignore=ignore_heavy, dirs_exist_ok=True)
print("Copied pipeline/demo_local")

# 5. Pipeline demo2_physics (Demo 2)
if (src / "pipeline" / "demo2_physics").exists():
    shutil.copytree(src / "pipeline" / "demo2_physics", dst / "pipeline" / "demo2_physics", ignore=ignore_heavy, dirs_exist_ok=True)
    print("Copied pipeline/demo2_physics")

# 6. Base files
for f in [".gitignore", ".gitattributes", "LICENSE", "LICENSE-docs.md"]:
    if (src / f).exists():
        shutil.copy2(src / f, dst / f)
        print(f"Copied {f}")

print("Repository base sync complete!")
