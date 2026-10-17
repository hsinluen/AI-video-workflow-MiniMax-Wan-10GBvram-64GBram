"""Parameter Profile Switcher for LTX-Video 10GB VRAM Pipeline.

Allows switching or restoring between saved parameter profiles:
  - v1_fast            : Original post-install fast draft (20 steps, cfg=3.0, KSampler simple, no STG)
  - v2_fine_detail     : Intermediate test (35 steps, cfg=2.6, KSampler simple)
  - v3_r2_official_ltx : Official Lightricks LTX-Video R2 high-quality anchored profile (40 steps, LTXVScheduler, ModelSamplingLTXV, STG block 14, LTXVPreprocess, LTXVAddGuide)

Usage:
  python tools/switch_profile.py --list
  python tools/switch_profile.py --profile v1_fast --film pipeline/demo_local/film.json
  python tools/switch_profile.py --profile v3_r2_official_ltx --film pipeline/20261016-missing_bento-R2/film.json
"""
import argparse
import json
from pathlib import Path

PROFILES_DIR = Path(__file__).resolve().parent.parent / "configs" / "params_profiles"

PROFILE_MAP = {
    "v1": "profile_v1_fast.json",
    "v1_fast": "profile_v1_fast.json",
    "v2": "profile_v2_fine_detail.json",
    "v2_fine_detail": "profile_v2_fine_detail.json",
    "v3": "profile_v3_r2_official_ltx.json",
    "v3_r2": "profile_v3_r2_official_ltx.json",
    "v3_r2_official_ltx": "profile_v3_r2_official_ltx.json",
    "v4": "profile_v4_r3_balanced_motion.json",
    "v4_r3": "profile_v4_r3_balanced_motion.json",
    "v4_r3_balanced_motion": "profile_v4_r3_balanced_motion.json",
    "r3": "profile_v4_r3_balanced_motion.json",
}


def load_profile(name: str) -> dict:
    fname = PROFILE_MAP.get(name, name if name.endswith(".json") else f"{name}.json")
    path = PROFILES_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def apply_to_film(profile: dict, film_path: Path):
    film = json.loads(film_path.read_text(encoding="utf-8"))
    s = profile["settings"]
    film["steps"] = s.get("steps", 35)
    film["cfg"] = s.get("cfg", 2.5)
    film["sampler_name"] = s.get("sampler_name", "euler")
    film["scheduler"] = s.get("scheduler", "ltxv")
    film["stg"] = s.get("stg_enabled", True)
    film["stg_scale"] = s.get("stg_scale", 1.0)
    film["stg_blocks"] = s.get("stg_blocks", "14")
    film["stg_start_percent"] = s.get("stg_start_percent", 0.0)
    film["stg_end_percent"] = s.get("stg_end_percent", 0.75)
    film["img_compression"] = s.get("img_compression", 28)
    film["img_strength"] = s.get("img_strength", 0.90)
    film["end_anchor_strength"] = s.get("end_anchor_strength", 0.0)
    if isinstance(s.get("length"), int):
        film["length"] = s["length"]
    elif "length" in film:
        del film["length"]
    film["active_profile"] = profile["profile_name"]
    film_path.write_text(json.dumps(film, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Applied profile '{profile['profile_name']}' to {film_path}")


def main():
    parser = argparse.ArgumentParser(description="Switch or restore LTX-Video parameter profiles")
    parser.add_argument("--list", action="store_true", help="List all available parameter profiles")
    parser.add_argument("--profile", type=str, help="Profile name (e.g., v1_fast, v2_fine_detail, v3_r2_official_ltx)")
    parser.add_argument("--engine", type=str, choices=["ltx", "wan2.1", "hunyuan", "h3"], help="Set active video engine (ltx, wan2.1, hunyuan, h3)")
    parser.add_argument("--film", type=str, help="Target film.json to update with the selected profile or engine")
    args = parser.parse_args()

    if args.list:
        print("=== Available Parameter Profiles ===")
        for f in sorted(PROFILES_DIR.glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            print(f"  - {f.stem} ({data.get('profile_name')}): {data.get('description')}")
        print("\n=== Supported Video Engines ===")
        print("  - ltx      : Lightricks LTX-Video 2B FP8 (30-60s per shot, low memory)")
        print("  - wan2.1   : Alibaba Wan 2.1 14B I2V Q4_K_M GGUF (flagship photo consistency, zero ghosting)")
        print("  - hunyuan  : Tencent HunyuanVideo 720p Q4_K_M GGUF (cinematic lighting, Asian portrait)")
        print("  - h3       : MiniMax-H3 INT8 / FastH3 4-8 steps (rapid high-quality draft)")
        return

    if args.engine and args.film:
        film_p = Path(args.film).resolve()
        if film_p.exists():
            f_data = json.loads(film_p.read_text(encoding="utf-8"))
            f_data["engine"] = args.engine
            film_p.write_text(json.dumps(f_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] Set engine to '{args.engine}' in {film_p}")

    if args.profile:
        prof = load_profile(args.profile)
        print(f"Loaded profile: {prof['profile_name']} (v{prof['version']})")
        print(json.dumps(prof["settings"], ensure_ascii=False, indent=2))
        if args.film:
            apply_to_film(prof, Path(args.film).resolve())
    elif not args.engine and not args.list:
        parser.print_help()


if __name__ == "__main__":
    main()
