import os
import sys
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_BV = HERE / "out" / "voice" / "bv"
OUT_BV.mkdir(parents=True, exist_ok=True)

PYTHON_VOX = r"C:\2026-AntiGravity_Voice_cloner\voxcpm2-voice-cloner\.venv\Scripts\python.exe"
CLONE_SCRIPT = str(Path.home() / ".gemini" / "config" / "skills" / "Tsai-voice-cloning-skill" / "scripts" / "clone_tsai.py")
EDGE_TTS = r"C:\AI\Voice\venv\Scripts\edge-tts.exe"

lines = [
    {
        "id": "0",
        "who": "tsai",
        "text": "各位同學，北半球科氏力讓氣流右偏，那它能決定洗臉台的水流方向嗎？",
        "engine": "voxcpm2"
    },
    {
        "id": "1",
        "who": "hsuan",
        "text": "老師！女宿水槽明明都是順時針排掉，難道我們寢室偷渡到南半球了嗎？",
        "engine": "edge",
        "voice": "zh-TW-HsiaoChenNeural"
    },
    {
        "id": "2",
        "who": "tsai",
        "text": "除非妳的洗臉台有兩百公里大，否則只是拔塞子的角度！下課洗臉吧！",
        "engine": "voxcpm2"
    }
]

picks = {}

for item in lines:
    lid = item["id"]
    wav_out = OUT_BV / f"line{lid}.wav"
    mp3_out = OUT_BV / f"line{lid}.mp3"
    print(f"\n[VOICE] Generating Line {lid} ({item['who']}): {item['text']}", flush=True)
    
    if item["engine"] == "voxcpm2":
        cmd = [
            PYTHON_VOX, CLONE_SCRIPT,
            "--text", item["text"],
            "--output", str(wav_out)
        ]
        subprocess.run(cmd, check=True)
    else:
        cmd_edge = [
            EDGE_TTS,
            "--voice", item["voice"],
            "--text", item["text"],
            "--write-media", str(mp3_out)
        ]
        subprocess.run(cmd_edge, check=True)
        cmd_convert = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mp3_out),
            "-ar", "48000", "-ac", "1",
            str(wav_out)
        ]
        subprocess.run(cmd_convert, check=True)

    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav_out)],
        capture_output=True, text=True, check=True
    )
    duration = float(r.stdout.strip())
    print(f"[VOICE] Line {lid} generated: {wav_out.name} ({duration:.2f}s)", flush=True)

    picks[lid] = {
        "file": str(wav_out.resolve()),
        "text": item["text"],
        "voice": "Tsai-Cloned" if item["engine"] == "voxcpm2" else item["voice"],
        "duration": duration
    }

vp_path = HERE / "out" / "voice_picks.json"
vp_path.write_text(json.dumps(picks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[VOICE] All Demo 4 voice lines created and written to {vp_path}", flush=True)
