"""全本地示範片：從零做出一支約 15 秒、4 鏡的大學生校園喜劇短片《大學新生的第一堂課：迷航記》。
證明安裝好的 LTX-Video 10GB VRAM 工作流可以完整跑通，兼顧生動肢體動作與五官清晰度。

劇情綱要：
  - S1 (阿凱，大階梯教室門口)：拿著校園地圖向學姊問路：「請問微積分是在綜合大樓嗎？」
  - S2 (晴晴，文學院走廊)：推眼鏡微笑調侃：「同學，這裡是文學院，理工館在對面山上喔。」
  - S3 (阿凱，特寫)：看著地圖震驚石化，露出難以置信的喜劇苦笑。
  - S4 (雙人，林蔭大道旁)：晴晴指著腳踏車：「走吧，學姊順路載你一程。」阿凱感激相視而笑。

執行方式：
  C:\\AI\\H3\\venv\\Scripts\\python.exe pipeline\\demo_local\\run_demo.py [voices h3 music cut]
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYH3 = sys.executable

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if __name__ == "__main__":
    print("=" * 70)
    print("🎬 啟動內定示範短片：《大學新生的第一堂課：迷航記》")
    print("   架構：LTX-Video 2B FP8 + ModelSamplingLTXV + LTXVScheduler (R3 平衡版)")
    print("   硬體規格適配：NVIDIA RTX 3080 10GB VRAM / 64GB RAM")
    print("=" * 70)
    cmd = [PYH3, str(HERE / "make_film.py")] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)
