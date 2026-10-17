"""RTX VSR 1080p, with automatic high-quality Lanczos fallback when nvvfx is unavailable."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import nvvfx
    import av
    HAS_NVVFX = True
except Exception:
    HAS_NVVFX = False

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--res', '--resolution', default='1080p', choices=['720p', '1080p', 'auto'], help='Target resolution: 720p (1280x720) or 1080p (1920x1080)')
parser.add_argument('--quality', default='ULTRA', help='nvvfx QualityLevel name, e.g. ULTRA or HIGHBITRATE_ULTRA')
args = parser.parse_args()

src = args.source
dest = args.output
if dest.exists():
    dest.unlink()
folder = dest.parent
folder.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()

if args.res == '720p':
    target_w, target_h = 1280, 720
else:
    target_w, target_h = 1920, 1080

if not HAS_NVVFX:
    print(f"[upscale_vsr] nvvfx not detected, using high-quality ffmpeg Lanczos {args.res} ({target_w}x{target_h}) upscale: {src} -> {dest}", flush=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
        "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(dest)
    ]
    subprocess.run(cmd, check=True)
    report = {
        'engine': f'FFmpeg Lanczos {args.res} Fallback',
        'quality': 'Lanczos',
        'source': str(src),
        'output': str(dest),
        'width': target_w,
        'height': target_h,
        'elapsed_seconds': time.perf_counter() - start,
        'audio': 'original audio copied'
    }
    (folder / 'vsr_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
else:
    import av
    silent = folder / 'vsr_video_only.mp4'
    torch.cuda.reset_peak_memory_stats()
    checks = []
    with av.open(str(src)) as inp, av.open(str(silent), 'w') as out:
        fps = inp.streams.video[0].average_rate
        stream = out.add_stream('libx264', rate=fps)
        stream.width, stream.height, stream.pix_fmt = target_w, target_h, 'yuv420p'
        stream.options = {'crf': '16', 'preset': 'slow'}
        with nvvfx.VideoSuperRes(getattr(nvvfx.effects.QualityLevel, args.quality)) as sr:
            sr.output_width, sr.output_height = target_w, target_h
            sr.load()
            for i, frame in enumerate(inp.decode(video=0)):
                rgb = frame.to_ndarray(format='rgb24')
                tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float().div_(255).contiguous()
                result = torch.from_dlpack(sr.run(tensor).image).permute(1, 2, 0).clamp(0, 1)
                image = result.mul(255).round().byte().cpu().numpy()
                if not image.any():
                    raise RuntimeError('VSR returned a black frame')
                encoded = av.VideoFrame.from_ndarray(image, format='rgb24')
                encoded.pts = i
                for packet in stream.encode(encoded):
                    out.mux(packet)
                if i % 24 == 0:
                    print(f'FRAME {i}', flush=True)
            count = i + 1
            for packet in stream.encode():
                out.mux(packet)

    # Re-mux encoded video and original audio packets
    with av.open(str(silent)) as vin, av.open(str(src)) as ain, av.open(str(dest), 'w') as out:
        vs = out.add_stream_from_template(vin.streams.video[0])
        aus = out.add_stream_from_template(ain.streams.audio[0]) if ain.streams.audio else None
        packets = []
        sources = [(vin, vin.streams.video[0], vs)]
        if aus:
            sources.append((ain, ain.streams.audio[0], aus))
        for source, instream, target in sources:
            for p in source.demux(instream):
                if p.dts is not None:
                    t = float(p.dts * p.time_base)
                    p.stream = target
                    packets.append((t, p))
        for _, p in sorted(packets, key=lambda item: item[0]):
            out.mux(p)

    report = {
        'engine': 'NVIDIA RTX VSR',
        'quality': args.quality,
        'source': str(src),
        'output': str(dest),
        'width': 1920,
        'height': 1080,
        'frames': count,
        'fps': 24,
        'elapsed_seconds': time.perf_counter() - start,
        'torch_peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
        'audio': 'original AAC packets copied',
        'checks': checks
    }
    (folder / 'vsr_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
